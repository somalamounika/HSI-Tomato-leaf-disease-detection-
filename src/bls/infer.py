"""Run parse → load → segment → features → model → anomaly on a folder of scans."""

from __future__ import annotations

import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from bls.config import Config, load_config
from bls.features import features_from_pixels, load_leaf_npz
from bls.filenames import parse_name
from bls.inventory import build_inventory
from bls.io import false_rgb, load_cube
from bls.leafmap import merge_leaf_map
from bls.segment import load_expected_leaves, segment_scan
from bls.train_leaf import _predict_proba, feature_columns

_LOG = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]


def extract_zip(zip_path: Path, dest: Path | None = None) -> Path:
    dest = dest or Path(tempfile.mkdtemp(prefix="bls_zip_"))
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return dest


def load_model_bundle(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if "model" not in bundle or "feature_columns" not in bundle:
        raise ValueError(f"{path} is not a leaf-model bundle (need model + feature_columns)")
    return bundle


def load_healthy_reference(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return joblib.load(path)


def infer_folder(
    cfg: Config,
    data_dir: Path,
    model_path: Path | None = None,
    reference_path: Path | None = None,
    threshold: float | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Run the pipeline on every in-scope scan under *data_dir*.

    Group labels are optional (joined from leaf_map.csv if present).
    """
    original_data = cfg.data.get("paths", {}).get("data_dir")
    cfg.data.setdefault("paths", {})["data_dir"] = str(data_dir)
    try:
        inv = build_inventory(cfg)
    finally:
        if original_data is not None:
            cfg.data["paths"]["data_dir"] = original_data

    in_scope = inv[inv["in_scope"] & inv["pair_ok"]].copy()
    model_path = model_path or (cfg.path("paths", "models_dir") / "leaf_model.joblib")
    bundle = load_model_bundle(model_path) if model_path.exists() else None
    ref_path = reference_path or (cfg.path("paths", "models_dir") / "healthy_reference.joblib")
    ref = load_healthy_reference(ref_path)
    thr = float(threshold if threshold is not None else cfg.get("ui", "default_threshold", default=0.5))
    expected = load_expected_leaves(cfg)

    overlays: dict[str, Path] = {}
    heatmaps: dict[str, Path] = {}
    overlay_dir = cfg.path("paths", "overlays_dir")
    cache = cfg.path("paths", "cache_dir")
    features_path = cfg.path("paths", "features")
    anomaly_path = cfg.path("paths", "reports_dir") / "anomaly_leaves.csv"

    # Reuse existing feature table when inferring the configured data folder.
    configured_data = cfg.path("paths", "data_dir").resolve()
    reuse = Path(data_dir).resolve() == configured_data and features_path.exists()
    if reuse:
        if progress:
            progress("Loading cached features", 0.2)
        leaves = pd.read_csv(features_path)
        if bundle is not None:
            cols = bundle["feature_columns"]
            X = leaves.reindex(columns=cols, fill_value=0.0).to_numpy(dtype=float)
            p = _predict_proba(bundle["model"], X)
            leaves["P_infected"] = p
            leaves["predicted_label"] = (p >= thr).astype(int)
        if anomaly_path.exists():
            anom = pd.read_csv(anomaly_path)
            keep = [c for c in ("stem", "leaf_no", "median_distance", "atypical_pixel_pct", "heatmap") if c in anom.columns]
            leaves = leaves.merge(anom[keep], on=["stem", "leaf_no"], how="left")
        for rec in in_scope.itertuples(index=False):
            overlay = overlay_dir / f"{rec.stem}.png"
            if overlay.exists():
                overlays[str(rec.stem)] = overlay
        try:
            leaves = merge_leaf_map(cfg, leaves)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("leaf_map merge skipped: %s", exc)
        return {
            "inventory": inv,
            "leaves": leaves,
            "overlays": overlays,
            "heatmaps": heatmaps,
            "threshold": thr,
            "model_name": bundle["name"] if bundle else None,
            "reference_mean": None if ref is None else np.asarray(ref["mean_spectrum"]),
            "reference_wl": None if ref is None else np.asarray(ref["wavelengths"]),
        }

    rows: list[dict[str, Any]] = []
    n_scans = max(len(in_scope), 1)

    for i, rec in enumerate(in_scope.itertuples(index=False)):
        stem = str(rec.stem)
        if progress:
            progress(f"Segmenting {stem}", i / n_scans)
        _LOG.info("infer scan %s", stem)
        overlay_path = overlay_dir / f"{stem}.png"
        leaves_csv = cfg.path("paths", "leaves")
        cached_rows: list[dict[str, Any]] | None = None
        if overlay_path.exists() and leaves_csv.exists():
            existing = pd.read_csv(leaves_csv)
            part = existing[existing["stem"] == stem]
            npz_ok = all((cache / f"{stem}__leaf{int(n):02d}.npz").exists() for n in part["leaf_no"]) if not part.empty else False
            if npz_ok:
                cached_rows = part.to_dict(orient="records")
                _LOG.info("reusing segmentation for %s", stem)
        if cached_rows is None:
            leaf_rows, overlay_path = segment_scan(
                cfg, stem, Path(str(rec.bil_path)), Path(str(rec.hdr_path)), expected=expected.get(stem)
            )
        else:
            leaf_rows = cached_rows
        overlays[stem] = overlay_path

        for leaf in leaf_rows:
            npz = cache / f"{stem}__leaf{int(leaf['leaf_no']):02d}.npz"
            blob = load_leaf_npz(npz)
            feats = features_from_pixels(blob["spectra"], blob["wavelengths"], cfg)
            rec_out = {**leaf, **feats}

            if bundle is not None:
                cols = bundle["feature_columns"]
                x = np.array([[float(rec_out.get(c, 0.0)) for c in cols]], dtype=float)
                p = float(_predict_proba(bundle["model"], x)[0])
                rec_out["P_infected"] = p
                rec_out["predicted_label"] = int(p >= thr)
            else:
                rec_out["P_infected"] = float("nan")
                rec_out["predicted_label"] = None

            if ref is not None and blob["spectra"].shape[0]:
                from bls.anomaly import HealthyReference
                from bls.features import snv

                spec = np.asarray(blob["spectra"], dtype=np.float64)
                spec = np.where(np.isfinite(spec), spec, 0.0)
                if bool(cfg.get("anomaly", "snv_pixels", default=True)):
                    spec = snv(spec)
                href = HealthyReference(
                    ref["pca"], ref["lw"], ref["threshold"], ref["mean_spectrum"], ref["wavelengths"]
                )
                dist = href.mahalanobis(spec)
                rec_out["median_distance"] = float(np.median(dist))
                rec_out["atypical_pixel_pct"] = float(100.0 * (dist > href.threshold).mean())
            rows.append(rec_out)

        if progress:
            progress(f"Done {stem}", (i + 1) / n_scans)

    leaves = pd.DataFrame(rows)
    if not leaves.empty:
        try:
            leaves = merge_leaf_map(cfg, leaves)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("leaf_map merge skipped: %s", exc)
    return {
        "inventory": inv,
        "leaves": leaves,
        "overlays": overlays,
        "heatmaps": heatmaps,
        "threshold": thr,
        "model_name": bundle["name"] if bundle else None,
        "reference_mean": None if ref is None else np.asarray(ref["mean_spectrum"]),
        "reference_wl": None if ref is None else np.asarray(ref["wavelengths"]),
    }


def infer_from_source(
    source: str | Path,
    cfg: Config | None = None,
    model_path: Path | None = None,
    threshold: float | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_config()
    source = Path(source)
    if source.suffix.lower() == ".zip":
        data_dir = extract_zip(source)
    else:
        data_dir = source
    return infer_folder(cfg, data_dir, model_path=model_path, threshold=threshold, progress=progress)
