"""Early-change maps: PCA + Mahalanobis distance from a healthy reference.

Everything here is labelled "exploratory: spectral difference from healthy
reference", never "disease area".
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

from bls.config import Config, ensure_dir, set_seed
from bls.features import load_leaf_npz, snv
from bls.io import false_rgb, load_cube
from bls.leafmap import merge_leaf_map

_LOG = logging.getLogger(__name__)


def is_reference_leaf(row: pd.Series) -> bool:
    """Healthy reference: bi leaves (both groups) and control leaves at all time points."""
    tp = str(row.get("time_point", "")).lower()
    group = str(row.get("group", "")).lower()
    if tp == "bi":
        return True
    return group == "control"


def _stack_reference(cfg: Config, leaves: pd.DataFrame, rng: np.random.Generator, n_max: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = cfg.path("paths", "cache_dir")
    chunks: list[np.ndarray] = []
    units: list[np.ndarray] = []
    wl = None
    ref_mask = leaves.apply(is_reference_leaf, axis=1)
    for rec in leaves[ref_mask].itertuples(index=False):
        npz = cache / f"{rec.stem}__leaf{int(rec.leaf_no):02d}.npz"
        if not npz.exists():
            continue
        blob = load_leaf_npz(npz)
        spec = np.asarray(blob["spectra"], dtype=np.float32)
        if spec.size == 0:
            continue
        if wl is None:
            wl = np.asarray(blob["wavelengths"], dtype=np.float64)
        unit = -1 if pd.isna(getattr(rec, "unit", np.nan)) else int(rec.unit)
        run = -1 if pd.isna(getattr(rec, "run", np.nan)) else int(rec.run)
        uid = run * 1000 + unit
        chunks.append(spec)
        units.append(np.full(spec.shape[0], uid, dtype=np.int64))
    if not chunks:
        raise RuntimeError("No reference pixels found (need bi leaves and/or control leaves)")
    spec = np.concatenate(chunks, axis=0)
    unit_arr = np.concatenate(units, axis=0)
    if spec.shape[0] > n_max:
        pick = rng.choice(spec.shape[0], size=n_max, replace=False)
        spec = spec[pick]
        unit_arr = unit_arr[pick]
    assert wl is not None
    return spec, unit_arr, wl


def _prepare_spectra(spec: np.ndarray, cfg: Config) -> np.ndarray:
    spec = np.asarray(spec, dtype=np.float64)
    spec = np.where(np.isfinite(spec), spec, 0.0)
    if bool(cfg.get("anomaly", "snv_pixels", default=True)):
        spec = snv(spec, eps=float(cfg.get("features", "eps", default=1e-8)))
    return spec


class HealthyReference:
    """PCA + Ledoit–Wolf covariance in PCA space, fit on healthy-reference pixels."""

    def __init__(self, pca: PCA, lw: LedoitWolf, threshold: float, mean_spectrum: np.ndarray, wavelengths: np.ndarray):
        self.pca = pca
        self.lw = lw
        self.threshold = float(threshold)
        self.mean_spectrum = mean_spectrum
        self.wavelengths = wavelengths

    def mahalanobis(self, spec: np.ndarray) -> np.ndarray:
        z = self.pca.transform(spec)
        diff = z - self.lw.location_
        prec = self.lw.precision_
        return np.sqrt(np.sum(diff @ prec * diff, axis=1))


def fit_reference(
    spec: np.ndarray,
    wavelengths: np.ndarray,
    cfg: Config,
) -> HealthyReference:
    n_comp = int(cfg.require("anomaly", "n_components"))
    pct = float(cfg.require("anomaly", "distance_percentile"))
    X = _prepare_spectra(spec, cfg)
    n_comp = min(n_comp, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_comp, random_state=cfg.seed())
    z = pca.fit_transform(X)
    lw = LedoitWolf().fit(z)
    dist = _mahalanobis(z, lw)
    threshold = float(np.percentile(dist, pct))
    mean_spectrum = np.asarray(spec, dtype=np.float64).mean(axis=0)
    return HealthyReference(pca, lw, threshold, mean_spectrum, wavelengths)


def _mahalanobis(z: np.ndarray, lw: LedoitWolf) -> np.ndarray:
    diff = z - lw.location_
    return np.sqrt(np.sum(diff @ lw.precision_ * diff, axis=1))


def unit_id(row: pd.Series) -> int:
    """Stable id for leave-one-unit-out.

    Unlabelled leaves (bi / 2hr) must not share one dummy unit. If they do,
    scoring a run-2 leaf drops every run-2 pixel from the reference and
    compares it to run-1 days 1–3 — that 86–100% 'atypical' figure is a
    lighting mismatch, not disease area.
    """
    run = -1 if pd.isna(row.get("run", np.nan)) else int(row["run"])
    unit = row.get("unit", np.nan)
    group = row.get("group", np.nan)
    missing = pd.isna(unit) or (int(unit) == 0 and pd.isna(group))
    if missing:
        stem = str(row.get("stem", ""))
        leaf = int(row.get("leaf_no", 0))
        key = f"{run}:{stem}:{leaf}".encode()
        return int(hashlib.md5(key).hexdigest()[:8], 16)
    return run * 1000 + int(unit)


def leaf_anomaly(
    cfg: Config,
    rec: pd.Series,
    ref: HealthyReference,
) -> dict[str, Any]:
    cache = cfg.path("paths", "cache_dir")
    npz = cache / f"{rec['stem']}__leaf{int(rec['leaf_no']):02d}.npz"
    blob = load_leaf_npz(npz)
    spec = _prepare_spectra(blob["spectra"], cfg)
    if spec.shape[0] == 0:
        return {
            "median_distance": 0.0,
            "atypical_pixel_pct": 0.0,
            "n_pixels": 0,
        }
    dist = ref.mahalanobis(spec)
    return {
        "median_distance": float(np.median(dist)),
        "atypical_pixel_pct": float(100.0 * (dist > ref.threshold).mean()),
        "n_pixels": int(dist.size),
        "distances": dist,
        "rows": blob["rows"],
        "cols": blob["cols"],
        "bbox": blob["bbox"],
    }


def save_heatmap(
    cfg: Config,
    rec: pd.Series,
    result: dict[str, Any],
    rgb: np.ndarray | None,
    out_dir: Path,
) -> Path | None:
    if "distances" not in result:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Use the stored rows/cols against a canvas the size of the scan RGB if given.
    if rgb is not None:
        canvas_h, canvas_w = rgb.shape[:2]
    else:
        rows = result["rows"]
        cols = result["cols"]
        canvas_h = int(rows.max()) + 1 if rows.size else 1
        canvas_w = int(cols.max()) + 1 if cols.size else 1
    heat = np.full((canvas_h, canvas_w), np.nan, dtype=np.float32)
    rows = result["rows"]
    cols = result["cols"]
    heat[rows, cols] = result["distances"].astype(np.float32)

    cmap = str(cfg.get("anomaly", "heatmap_cmap", default="inferno"))
    fig, axes = plt.subplots(1, 2 if rgb is not None else 1, figsize=(8, 5))
    if rgb is not None:
        axes = np.atleast_1d(axes)
        axes[0].imshow(rgb)
        axes[0].set_title("false colour")
        axes[0].set_axis_off()
        im = axes[1].imshow(heat, cmap=cmap)
        axes[1].set_title("Mahalanobis distance\n(exploratory: vs healthy reference)")
        axes[1].set_axis_off()
        fig.colorbar(im, ax=axes[1], fraction=0.046)
    else:
        ax = axes if not hasattr(axes, "__len__") else axes[0]
        im = ax.imshow(heat, cmap=cmap)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{rec['stem']} leaf {int(rec['leaf_no'])}  (not a disease-area map)")
    fig.tight_layout()
    path = out_dir / f"{rec['stem']}__leaf{int(rec['leaf_no']):02d}_heatmap.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def run_anomaly(cfg: Config, force: bool = False) -> pd.DataFrame:
    set_seed(cfg)
    reports = ensure_dir(cfg.path("paths", "reports_dir"))
    fig_dir = ensure_dir(cfg.path("paths", "figures_dir") / "heatmaps")
    out_path = reports / "anomaly_leaves.csv"
    if out_path.exists() and not force:
        _LOG.info("Anomaly table exists; skip (use --force to rebuild)")
        return pd.read_csv(out_path)

    from bls.features import run_features

    leaves = run_features(cfg, force=False)
    leaves = merge_leaf_map(cfg, leaves)
    # Reference uses bi leaves even when include=0 (group unknown); control uses labelled rows.
    if leaves.empty:
        raise RuntimeError("No leaves for anomaly analysis")

    rng = np.random.default_rng(cfg.seed())
    n_max = int(cfg.require("anomaly", "n_ref_pixels"))
    spec, unit_arr, wl = _stack_reference(cfg, leaves, rng, n_max)
    use_loo = bool(cfg.get("anomaly", "leave_one_unit_out_reference", default=True))

    global_ref = fit_reference(spec, wl, cfg)
    rows: list[dict[str, Any]] = []

    # Cache RGB per stem for heatmaps.
    rgb_cache: dict[str, np.ndarray] = {}
    inv_path = cfg.path("paths", "inventory")
    inv = pd.read_csv(inv_path) if inv_path.exists() else pd.DataFrame()

    for rec in leaves.itertuples(index=False):
        rec_s = pd.Series(rec._asdict()) if hasattr(rec, "_asdict") else pd.Series(rec)
        uid = unit_id(rec_s)
        if use_loo:
            keep = unit_arr != uid
            if keep.sum() < 50:
                ref = global_ref
                loo_note = "global_fallback_too_few_pixels"
            else:
                ref = fit_reference(spec[keep], wl, cfg)
                loo_note = "leave_one_unit_out"
        else:
            ref = global_ref
            loo_note = "global_reference"
        result = leaf_anomaly(cfg, rec_s, ref)
        stem = str(rec.stem)
        if stem not in rgb_cache and not inv.empty:
            match = inv[inv["stem"] == stem]
            if not match.empty and bool(match.iloc[0]["pair_ok"]):
                try:
                    cube, wlc = load_cube(match.iloc[0]["bil_path"], match.iloc[0]["hdr_path"], cfg=cfg, trim=True)
                    rgb_cache[stem] = false_rgb(cube, wlc, cfg)
                except Exception as exc:  # noqa: BLE001
                    _LOG.warning("Could not load RGB for %s: %s", stem, exc)
        heat = save_heatmap(cfg, rec_s, result, rgb_cache.get(stem), fig_dir)
        rows.append(
            {
                "stem": rec.stem,
                "leaf_no": int(rec.leaf_no),
                "time_point": rec.time_point,
                "run": rec.run,
                "unit": rec_s.get("unit"),
                "group": rec_s.get("group"),
                "median_distance": result["median_distance"],
                "atypical_pixel_pct": result["atypical_pixel_pct"],
                "n_pixels": result["n_pixels"],
                "distance_threshold": ref.threshold,
                "reference": loo_note,
                "heatmap": str(heat) if heat else "",
                "label": "exploratory: spectral difference from healthy reference",
            }
        )
        _LOG.info(
            "anomaly %s leaf %02d median_d=%.3f atypical%%=%.2f (%s)",
            rec.stem,
            int(rec.leaf_no),
            result["median_distance"],
            result["atypical_pixel_pct"],
            loo_note,
        )

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    joblib_payload = {
        "pca": global_ref.pca,
        "lw": global_ref.lw,
        "threshold": global_ref.threshold,
        "mean_spectrum": global_ref.mean_spectrum,
        "wavelengths": global_ref.wavelengths,
        "label": "exploratory: spectral difference from healthy reference",
    }
    import joblib

    joblib.dump(joblib_payload, cfg.path("paths", "models_dir") / "healthy_reference.joblib")
    return out
