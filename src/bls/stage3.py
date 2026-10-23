"""3-class leaf stage model: day 1 healthy, day 3 early infection, day 6 infected."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bls.config import Config, ensure_dir, set_seed
from bls.features import features_from_pixels, load_leaf_npz
from bls.segment import segment_scan
from bls.train_leaf import feature_columns

_LOG = logging.getLogger(__name__)

CLASS_ORDER = ["healthy", "early", "infected"]
CLASS_DISPLAY = {
    "healthy": "Healthy",
    "early": "Early infection",
    "infected": "Infected",
}
TIME_TO_CLASS = {"d1": "healthy", "d3": "early", "d6": "infected"}
STAGE_STEMS = {
    "d1": "1dai_visnir_bc_run1",
    "d3": "3dai_visnir_bc_run1",
    "d6": "6dai_visnir_bc_run1",
}
LOCAL_6DAI_CANDIDATES = [
    Path(r"C:\Users\mouni\Downloads\hyperspectral data\6dai_visnir_bc_run1.bil"),
    Path(r"C:\Users\mouni\Downloads\eda of dataset\data\visnir_run1\6dai_visnir_bc_run1.bil"),
    Path(r"C:\Users\mouni\Downloads\HSI leaf\data\raw_samples\6dai_visnir_bc_run1.bil"),
]


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.hardlink_to(src)
        _LOG.info("Hardlinked %s -> %s", src, dst)
    except OSError:
        import shutil

        shutil.copy2(src, dst)
        _LOG.info("Copied %s -> %s", src, dst)


def ensure_6dai(cfg: Config) -> tuple[Path, Path]:
    data_dir = cfg.path("paths", "data_dir")
    bil = data_dir / "6dai_visnir_bc_run1.bil"
    hdr = data_dir / "6dai_visnir_bc_run1.bil.hdr"
    if bil.exists() and hdr.exists():
        return bil, hdr
    src = next((p for p in LOCAL_6DAI_CANDIDATES if p.exists()), None)
    if src is None:
        raise FileNotFoundError(
            "Could not find 6dai_visnir_bc_run1.bil. Place it in data/raw/ or keep it under Downloads."
        )
    src_hdr = Path(str(src) + ".hdr")
    if not src_hdr.exists():
        raise FileNotFoundError(f"Missing header {src_hdr}")
    _link_or_copy(src, bil)
    _link_or_copy(src_hdr, hdr)
    return bil, hdr


def _segment_6dai(cfg: Config, force: bool = False) -> pd.DataFrame:
    stem = STAGE_STEMS["d6"]
    overlay = cfg.path("paths", "overlays_dir") / f"{stem}.png"
    cache = cfg.path("paths", "cache_dir")
    stage_dir = ensure_dir(cfg.path("paths", "output_dir") / "stage3")
    leaves_path = stage_dir / "leaves_6dai.csv"
    if leaves_path.exists() and overlay.exists() and not force:
        existing = pd.read_csv(leaves_path)
        npz_ok = all((cache / f"{stem}__leaf{int(n):02d}.npz").exists() for n in existing["leaf_no"])
        if npz_ok:
            _LOG.info("Reusing 6dai segmentation (%d leaves)", len(existing))
            return existing
    bil, hdr = ensure_6dai(cfg)
    rows, _overlay = segment_scan(cfg, stem, bil, hdr, expected=None)
    df = pd.DataFrame(rows)
    df.to_csv(leaves_path, index=False)
    return df


def build_stage_table(cfg: Config, force: bool = False) -> pd.DataFrame:
    """1dai + 3dai from existing features, 6dai newly segmented."""
    stage_dir = ensure_dir(cfg.path("paths", "output_dir") / "stage3")
    out_path = stage_dir / "features.csv"
    if out_path.exists() and not force:
        return pd.read_csv(out_path)

    feats = pd.read_csv(cfg.path("paths", "features"))
    keep_tp = {"d1", "d3"}
    early = feats[feats["time_point"].astype(str).isin(keep_tp)].copy()

    six = _segment_6dai(cfg, force=force)
    cache = cfg.path("paths", "cache_dir")
    six_rows: list[dict[str, Any]] = []
    for rec in six.itertuples(index=False):
        npz = cache / f"{rec.stem}__leaf{int(rec.leaf_no):02d}.npz"
        blob = load_leaf_npz(npz)
        leaf_feats = features_from_pixels(blob["spectra"], blob["wavelengths"], cfg)
        meta = rec._asdict() if hasattr(rec, "_asdict") else {c: getattr(rec, c) for c in six.columns}
        six_rows.append({**meta, **leaf_feats})
    six_df = pd.DataFrame(six_rows)

    table = pd.concat([early, six_df], ignore_index=True, sort=False)
    table["stage"] = table["time_point"].astype(str).map(TIME_TO_CLASS)
    table = table[table["stage"].notna()].copy()
    table["stage_id"] = table["stage"].map({c: i for i, c in enumerate(CLASS_ORDER)}).astype(int)
    num = table.select_dtypes(include=[np.number])
    table[num.columns] = num.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    table.to_csv(out_path, index=False)
    _LOG.info("Stage-3 table: %d leaves (%s)", len(table), table["stage"].value_counts().to_dict())
    return table


def _estimators(cfg: Config) -> dict[str, Any]:
    seed = cfg.seed()
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "lda": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                ),
            ]
        ),
        "lightgbm": _lgb(seed),
    }


def _lgb(seed: int) -> Any:
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        num_leaves=7,
        min_child_samples=3,
        n_estimators=200,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.6,
        max_depth=3,
        objective="multiclass",
        num_class=3,
        class_weight="balanced",
        verbosity=-1,
        random_state=seed,
    )


def _loo_predict(estimator: Any, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    pred = np.full(len(y), -1, dtype=int)
    loo = LeaveOneOut()
    for train, test in loo.split(X):
        if len(np.unique(y[train])) < 2:
            continue
        model = clone(estimator)
        model.fit(X[train], y[train])
        pred[test] = model.predict(X[test])
    return pred


def train_stage3(cfg: Config, force: bool = False) -> dict[str, Any]:
    set_seed(cfg)
    stage_dir = ensure_dir(cfg.path("paths", "output_dir") / "stage3")
    models_dir = ensure_dir(cfg.path("paths", "models_dir"))
    model_path = models_dir / "stage3_model.joblib"
    pred_path = stage_dir / "predictions.csv"
    report_path = stage_dir / "model_report.md"
    if model_path.exists() and pred_path.exists() and not force:
        return joblib.load(model_path)

    table = build_stage_table(cfg, force=force)
    feat_cols = feature_columns(table, cfg)
    X = table[feat_cols].to_numpy(dtype=float)
    y = table["stage_id"].to_numpy(dtype=int)

    results: list[dict[str, Any]] = []
    for name, est in _estimators(cfg).items():
        oof = _loo_predict(est, X, y)
        ok = oof >= 0
        ba = float(balanced_accuracy_score(y[ok], oof[ok])) if ok.any() else 0.0
        acc = float(accuracy_score(y[ok], oof[ok])) if ok.any() else 0.0
        f1 = float(f1_score(y[ok], oof[ok], average="macro", zero_division=0)) if ok.any() else 0.0
        results.append({"name": name, "oof": oof, "balanced_accuracy": ba, "accuracy": acc, "f1": f1})
        _LOG.info("%s LOO BA=%.3f acc=%.3f f1=%.3f", name, ba, acc, f1)

    results.sort(key=lambda r: (-r["balanced_accuracy"], ["logistic_regression", "lda", "lightgbm"].index(r["name"])))
    selected = results[0]
    model = clone(_estimators(cfg)[selected["name"]])
    model.fit(X, y)
    fitted_pred = np.asarray(model.predict(X), dtype=int)
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    out = table[["stem", "leaf_no", "time_point", "stage", "stage_id"]].copy()
    out["pred_id"] = fitted_pred
    out["pred_stage"] = [CLASS_ORDER[i] for i in fitted_pred]
    out["oof_id"] = selected["oof"]
    out["oof_stage"] = [CLASS_ORDER[i] if i >= 0 else "" for i in selected["oof"]]
    if proba is not None:
        for i, name in enumerate(CLASS_ORDER):
            out[f"P_{name}"] = proba[:, i]
    out.to_csv(pred_path, index=False)

    lines = [
        "# 3-class stage model",
        "",
        "Labels from the use-case (not inoculated vs control):",
        "- **1dai → Healthy**",
        "- **3dai → Early infection**",
        "- **6dai → Infected**",
        "",
        f"Selected: **{selected['name']}**",
        f"Leave-one-leaf-out balanced accuracy: {selected['balanced_accuracy']:.3f}",
        f"Accuracy: {selected['accuracy']:.3f}  macro-F1: {selected['f1']:.3f}",
        "",
        "Leaves per class: " + ", ".join(f"{k}={v}" for k, v in table['stage'].value_counts().sort_index().items()),
        "",
        "```",
        classification_report(
            y[selected["oof"] >= 0],
            selected["oof"][selected["oof"] >= 0],
            target_names=CLASS_ORDER,
            zero_division=0,
        ),
        "```",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    bundle = {
        "model": model,
        "name": selected["name"],
        "feature_columns": feat_cols,
        "class_order": CLASS_ORDER,
        "class_display": CLASS_DISPLAY,
        "time_to_class": TIME_TO_CLASS,
        "loo": {r["name"]: {k: r[k] for k in ("balanced_accuracy", "accuracy", "f1")} for r in results},
        "n": int(len(table)),
    }
    joblib.dump(bundle, model_path)
    _LOG.info("Wrote %s", model_path)
    return bundle


def run_stage3(cfg: Config, force: bool = False) -> dict[str, Any]:
    ensure_6dai(cfg)
    table = build_stage_table(cfg, force=force)
    bundle = train_stage3(cfg, force=force)
    return {"table": table, "bundle": bundle}
