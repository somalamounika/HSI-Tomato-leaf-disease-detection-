"""Leaf-level inoculated vs control models: LightGBM, logistic regression, LDA."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bls.config import Config, ensure_dir, set_seed
from bls.features import INDEX_NAMES, run_features
from bls.leafmap import UNCONFIRMED_WARNING, assert_ready_for_training, merge_leaf_map

_LOG = logging.getLogger(__name__)

MODEL_NAMES = ["logistic_regression", "lda", "lightgbm"]


def feature_columns(df: pd.DataFrame, cfg: Config) -> list[str]:
    """Indices + pixel stats; optionally the downsampled spectrum. Never time/run/unit/position."""
    cols = [c for c in INDEX_NAMES if c in df.columns]
    extra = [
        "NDVI_median",
        "GNDVI_median",
        "NDRE_median",
        "PRI_median",
        "PSSRa_median",
        "CTR2_median",
        "SIPI_median",
        "mND705_median",
        "mSR705_median",
        "ARI_median",
        "REP_median",
        "WI_median",
        "NDVI_std",
        "NDVI_p5",
        "NDVI_p95",
        "PRI_std",
        "PRI_p5",
        "PRI_p95",
        "NDVI_low_frac",
        "RE_deriv_peak_nm",
        "RE_deriv_peak_amp",
    ]
    cols += [c for c in extra if c in df.columns]
    if bool(cfg.get("features", "include_downsampled_spectrum", default=False)):
        cols += [c for c in df.columns if c.startswith("R_") and c[2:].isdigit()]
    # Drop duplicates, keep order.
    seen: set[str] = set()
    out: list[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def make_estimators(cfg: Config) -> dict[str, Any]:
    seed = cfg.seed()
    log_cfg = cfg.require("models", "logistic")
    lda_cfg = cfg.require("models", "lda")
    lgb_cfg = cfg.require("models", "lightgbm")

    logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=float(log_cfg.get("C", 1.0)),
                    max_iter=int(log_cfg.get("max_iter", 2000)),
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )
    lda = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LinearDiscriminantAnalysis(
                    solver=str(lda_cfg.get("solver", "lsqr")),
                    shrinkage=lda_cfg.get("shrinkage", "auto"),
                ),
            ),
        ]
    )
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("lightgbm is required for the leaf classifier") from exc
    lightgbm = lgb.LGBMClassifier(
        num_leaves=int(lgb_cfg.get("num_leaves", 7)),
        min_child_samples=int(lgb_cfg.get("min_child_samples", 5)),
        n_estimators=int(lgb_cfg.get("n_estimators", 300)),
        learning_rate=float(lgb_cfg.get("learning_rate", 0.03)),
        subsample=float(lgb_cfg.get("subsample", 0.8)),
        colsample_bytree=float(lgb_cfg.get("colsample_bytree", 0.5)),
        max_depth=int(lgb_cfg.get("max_depth", 3)),
        objective=str(lgb_cfg.get("objective", "binary")),
        verbosity=int(lgb_cfg.get("verbosity", -1)),
        random_state=seed,
    )
    return {
        "logistic_regression": logistic,
        "lda": lda,
        "lightgbm": lightgbm,
    }


def cv_group_id(df: pd.DataFrame) -> pd.Series:
    """Leave-one-unit-out group = run + unit (units from different runs are different groups)."""
    run = df["run"].fillna(-1).astype(int).astype(str)
    unit = df["unit"].fillna(-1).astype(int).astype(str)
    return run + "::" + unit


def _predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba.ravel()
    if hasattr(model, "decision_function"):
        s = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-s))
    return np.asarray(model.predict(X), dtype=float)


def metrics_from_scores(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    out = {
        "n": int(y.size),
        "accuracy": float(accuracy_score(y, pred)) if y.size else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if y.size else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)) if y.size else float("nan"),
    }
    if y.size and len(np.unique(y)) > 1:
        try:
            out["roc_auc"] = float(roc_auc_score(y, p))
        except ValueError:
            out["roc_auc"] = float("nan")
    else:
        out["roc_auc"] = float("nan")
    return out


def leave_one_group_out(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, list[Any]]:
    """Fit on all groups except one; return OOF probabilities and the fold models."""
    oof = np.full(len(y), np.nan, dtype=float)
    models: list[Any] = []
    for g in np.unique(groups):
        train = groups != g
        test = groups == g
        if train.sum() == 0 or test.sum() == 0:
            continue
        if len(np.unique(y[train])) < 2:
            _LOG.warning("Skipping fold group=%s: training fold has a single class", g)
            continue
        model = clone(estimator)
        model.fit(X[train], y[train])
        oof[test] = _predict_proba(model, X[test])
        models.append(model)
    return oof, models


def leave_one_time_out(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    time_points: np.ndarray,
) -> np.ndarray:
    oof = np.full(len(y), np.nan, dtype=float)
    for tp in np.unique(time_points):
        train = time_points != tp
        test = time_points == tp
        if train.sum() == 0 or test.sum() == 0:
            continue
        if len(np.unique(y[train])) < 2:
            continue
        model = clone(estimator)
        model.fit(X[train], y[train])
        oof[test] = _predict_proba(model, X[test])
    return oof


def permutation_pvalue(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    n_perm: int,
    seed: int,
    real_auc: float,
) -> float:
    """Shuffle labels at the unit (cv-group) level; p = (# perm AUC >= real + 1) / (n+1)."""
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    group_y = {g: int(np.round(y[groups == g].mean())) for g in unique_groups}
    count = 0
    valid = np.isfinite(p)
    for _ in range(n_perm):
        perm_g = unique_groups.copy()
        rng.shuffle(perm_g)
        mapping = dict(zip(unique_groups, [group_y[g] for g in perm_g]))
        y_perm = np.array([mapping[g] for g in groups])
        if len(np.unique(y_perm[valid])) < 2:
            continue
        try:
            auc = float(roc_auc_score(y_perm[valid], p[valid]))
        except ValueError:
            continue
        if auc >= real_auc:
            count += 1
    return (count + 1) / (n_perm + 1)


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_confusion(y: np.ndarray, p: np.ndarray, path: Path, threshold: float, title: str) -> None:
    plt = _mpl()
    pred = (np.asarray(p) >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="black")
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _gain_importance(model: Any, names: list[str]) -> pd.DataFrame:
    clf = model
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf", model)
    if hasattr(clf, "booster_"):
        gain = clf.booster_.feature_importance(importance_type="gain")
        return pd.DataFrame({"feature": names, "gain": gain}).sort_values("gain", ascending=False)
    if hasattr(clf, "coef_"):
        coef = np.abs(np.ravel(clf.coef_))
        return pd.DataFrame({"feature": names, "abs_coef": coef}).sort_values("abs_coef", ascending=False)
    return pd.DataFrame({"feature": names})


def evaluate_model(
    name: str,
    estimator: Any,
    train_df: pd.DataFrame,
    bi_df: pd.DataFrame,
    feat_cols: list[str],
    cfg: Config,
) -> dict[str, Any]:
    threshold = float(cfg.get("models", "probability_threshold", default=0.5))
    X = train_df[feat_cols].to_numpy(dtype=float)
    y = train_df["infected"].to_numpy(dtype=int)
    groups = cv_group_id(train_df).to_numpy()
    tps = train_df["time_point"].astype(str).to_numpy()

    oof, fold_models = leave_one_group_out(estimator, X, y, groups)
    loto = leave_one_time_out(estimator, X, y, tps)
    overall = metrics_from_scores(y[np.isfinite(oof)], oof[np.isfinite(oof)], threshold)
    overall_loto = metrics_from_scores(y[np.isfinite(loto)], loto[np.isfinite(loto)], threshold)

    per_tp: list[dict[str, Any]] = []
    for tp in sorted(set(tps), key=lambda t: str(t)):
        m = tps == tp
        row = {"time_point": tp, "split": "loo_unit", **metrics_from_scores(y[m], oof[m], threshold)}
        per_tp.append(row)
        m2 = tps == tp
        per_tp.append(
            {"time_point": tp, "split": "loo_time", **metrics_from_scores(y[m2], loto[m2], threshold)}
        )

    bi_metrics = {"n": 0, "roc_auc": float("nan")}
    bi_p = np.array([])
    if not bi_df.empty and fold_models:
        Xb = bi_df[feat_cols].to_numpy(dtype=float)
        preds = np.column_stack([_predict_proba(m, Xb) for m in fold_models])
        bi_p = preds.mean(axis=1)
        yb = (bi_df["group"].astype(str).str.lower() == "inoculated").to_numpy(dtype=int)
        if len(np.unique(yb)) > 1:
            try:
                bi_auc = float(roc_auc_score(yb, bi_p))
            except ValueError:
                bi_auc = float("nan")
        else:
            bi_auc = float("nan")
        bi_metrics = {"n": int(len(bi_df)), "roc_auc": bi_auc}

    real_auc = overall.get("roc_auc", float("nan"))
    p_perm = float("nan")
    if np.isfinite(real_auc):
        p_perm = permutation_pvalue(
            y[np.isfinite(oof)],
            oof[np.isfinite(oof)],
            groups[np.isfinite(oof)],
            int(cfg.require("models", "n_permutations")),
            cfg.seed(),
            real_auc,
        )

    return {
        "name": name,
        "overall": overall,
        "loto": overall_loto,
        "per_time_point": per_tp,
        "bi": bi_metrics,
        "bi_p": bi_p,
        "oof": oof,
        "loto_p": loto,
        "fold_models": fold_models,
        "permutation_p": p_perm,
        "y": y,
        "groups": groups,
    }


def select_model(results: list[dict[str, Any]], cfg: Config) -> dict[str, Any]:
    tie = list(cfg.get("model_tie_order", default=MODEL_NAMES))
    scored = []
    for r in results:
        ba = r["overall"].get("balanced_accuracy", float("nan"))
        ba = -1.0 if not np.isfinite(ba) else ba
        tie_rank = tie.index(r["name"]) if r["name"] in tie else 99
        scored.append((ba, -tie_rank, r))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return scored[0][2]


def write_report(
    cfg: Config,
    results: list[dict[str, Any]],
    selected: dict[str, Any],
    feat_cols: list[str],
    train_df: pd.DataFrame,
    bi_df: pd.DataFrame,
    reports: Path,
    figures: Path,
) -> None:
    warn_thr = float(cfg.require("models", "bi_auc_warn"))
    lines: list[str] = []
    lines.append("# Leaf model report (inoculated vs control)")
    lines.append("")
    lines.append("Training time points: 2hr, d1, d2, d3. **bi is never used for training.**")
    lines.append("Validation: leave-one-unit-out (group = run + unit).")
    lines.append("A leave-one-time-point-out evaluation is also reported.")
    lines.append("Runs map to different time points in this subset; no train-on-one-run/test-on-the-other split.")
    lines.append("")
    lines.append(UNCONFIRMED_WARNING)
    lines.append("")
    lines.append(
        "The source paper reported roughly 0.55–0.64 accuracy at the early/pre-symptomatic "
        "stage. Modest numbers at d1–d3 are expected, not a bug."
    )
    lines.append("")
    lines.append(f"Selected model: **{selected['name']}**")
    ba = selected["overall"].get("balanced_accuracy")
    lines.append(
        f"Chosen for highest out-of-fold balanced accuracy "
        f"({ba:.3f} if finite). Ties break toward the simpler model "
        f"(logistic regression, then LDA, then LightGBM). "
        f"No hyperparameter search was run on the test folds."
    )
    lines.append("")
    lines.append("## Overall leave-one-unit-out metrics")
    lines.append("")
    lines.append("| model | n | accuracy | balanced_accuracy | F1 | ROC AUC | permutation p (AUC) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        o = r["overall"]
        lines.append(
            f"| {r['name']} | {o['n']} | {o['accuracy']:.3f} | {o['balanced_accuracy']:.3f} | "
            f"{o['f1']:.3f} | {o['roc_auc']:.3f} | {r['permutation_p']:.4f} |"
        )
    lines.append("")
    lines.append("## Leave-one-time-point-out (overall)")
    lines.append("")
    lines.append("| model | n | accuracy | balanced_accuracy | F1 | ROC AUC |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in results:
        o = r["loto"]
        lines.append(
            f"| {r['name']} | {o['n']} | {o['accuracy']:.3f} | {o['balanced_accuracy']:.3f} | "
            f"{o['f1']:.3f} | {o['roc_auc']:.3f} |"
        )
    lines.append("")
    lines.append("## Per time point (leave-one-unit-out)")
    lines.append("")
    lines.append("| model | time_point | n | accuracy | balanced_accuracy | F1 | ROC AUC |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in results:
        for row in r["per_time_point"]:
            if row["split"] != "loo_unit":
                continue
            lines.append(
                f"| {r['name']} | {row['time_point']} | {row['n']} | {row['accuracy']:.3f} | "
                f"{row['balanced_accuracy']:.3f} | {row['f1']:.3f} | {row['roc_auc']:.3f} |"
            )
    lines.append("")
    lines.append("## bi check (fold models applied to before-inoculation leaves)")
    lines.append("")
    lines.append("Expected AUC of P(infected) between groups at bi is about 0.5.")
    lines.append("")
    lines.append("| model | n_bi | AUC |")
    lines.append("|---|---:|---:|")
    for r in results:
        auc = r["bi"]["roc_auc"]
        flag = "  **WARN > 0.65**" if np.isfinite(auc) and auc > warn_thr else ""
        lines.append(f"| {r['name']} | {r['bi']['n']} | {auc:.3f}{flag} |")
        if np.isfinite(auc) and auc > warn_thr:
            _LOG.warning("%s bi-check AUC=%.3f exceeds %.2f", r["name"], auc, warn_thr)
    lines.append("")
    lines.append("## Features used")
    lines.append("")
    lines.append(", ".join(feat_cols))
    lines.append("")
    (reports / "leaf_model.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # CSVs
    rows = []
    for r in results:
        rec = {"model": r["name"], "split": "loo_unit_overall", **r["overall"], "permutation_p": r["permutation_p"]}
        rows.append(rec)
        rec2 = {"model": r["name"], "split": "loo_time_overall", **r["loto"]}
        rows.append(rec2)
        rec3 = {"model": r["name"], "split": "bi_check", "n": r["bi"]["n"], "roc_auc": r["bi"]["roc_auc"]}
        rows.append(rec3)
        for row in r["per_time_point"]:
            rows.append({"model": r["name"], **row})
    pd.DataFrame(rows).to_csv(reports / "model_comparison.csv", index=False)

    oof_rows = train_df[["stem", "leaf_no", "time_point", "group"]].copy()
    for r in results:
        oof_rows[f"P_infected_{r['name']}"] = r["oof"]
    oof_rows["P_infected"] = selected["oof"]
    oof_rows.to_csv(reports / "oof_predictions.csv", index=False)

    threshold = float(cfg.get("models", "probability_threshold", default=0.5))
    plot_confusion(
        selected["y"][np.isfinite(selected["oof"])],
        selected["oof"][np.isfinite(selected["oof"])],
        figures / "confusion_matrix.png",
        threshold,
        f"OOF confusion ({selected['name']})",
    )


def run_train(cfg: Config, force: bool = False) -> dict[str, Any]:
    set_seed(cfg)
    reports = ensure_dir(cfg.path("paths", "reports_dir"))
    models_dir = ensure_dir(cfg.path("paths", "models_dir"))
    figures = ensure_dir(cfg.path("paths", "figures_dir"))
    selected_path = models_dir / "leaf_model.joblib"
    if selected_path.exists() and (reports / "leaf_model.md").exists() and not force:
        _LOG.info("Train outputs exist; skip (use --force to rebuild)")
        return {"selected_path": selected_path}

    df = run_features(cfg, force=False)
    df = merge_leaf_map(cfg, df)
    df = df[df.get("include", 1) == 1].copy()
    assert_ready_for_training(cfg, df)

    train_tp = set(cfg.require("training", "time_points"))
    train_df = df[df["time_point"].isin(train_tp)].copy()
    bi_df = df[df["time_point"] == "bi"].copy()
    if train_df.empty:
        raise RuntimeError("No training leaves (2hr, d1, d2, d3) after filters")

    feat_cols = feature_columns(train_df, cfg)
    estimators = make_estimators(cfg)
    results = []
    for name, est in estimators.items():
        _LOG.info("Evaluating %s on %d leaves, %d features", name, len(train_df), len(feat_cols))
        results.append(evaluate_model(name, est, train_df, bi_df, feat_cols, cfg))

    selected = select_model(results, cfg)
    _LOG.info(
        "Selected %s (OOF balanced_accuracy=%.3f)",
        selected["name"],
        selected["overall"]["balanced_accuracy"],
    )
    write_report(cfg, results, selected, feat_cols, train_df, bi_df, reports, figures)

    # Fit all three on all eligible training leaves; save each + the selected copy.
    X_all = train_df[feat_cols].to_numpy(dtype=float)
    y_all = train_df["infected"].to_numpy(dtype=int)
    snapshot = {
        "config_source": str(cfg.source),
        "feature_columns": feat_cols,
        "selected": selected["name"],
        "training_time_points": sorted(train_tp),
        "n_train": int(len(train_df)),
        "seed": cfg.seed(),
    }
    saved = {}
    for name, est in estimators.items():
        model = clone(est)
        model.fit(X_all, y_all)
        payload = {"model": model, "name": name, "feature_columns": feat_cols, "snapshot": snapshot}
        path = models_dir / f"{name}.joblib"
        joblib.dump(payload, path)
        saved[name] = path
        _LOG.info("Wrote %s", path)

        if name == selected["name"]:
            # Permutation importance on the full-fit model (descriptive, not for selection).
            try:
                perm = permutation_importance(
                    model, X_all, y_all, n_repeats=20, random_state=cfg.seed(), scoring="roc_auc"
                )
                pd.DataFrame(
                    {
                        "feature": feat_cols,
                        "perm_importance_mean": perm.importances_mean,
                        "perm_importance_std": perm.importances_std,
                    }
                ).sort_values("perm_importance_mean", ascending=False).to_csv(
                    reports / "permutation_importance.csv", index=False
                )
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("Permutation importance failed: %s", exc)
            gain = _gain_importance(model, feat_cols)
            gain.to_csv(reports / "feature_importance.csv", index=False)

    joblib.dump(
        {
            "model": joblib.load(saved[selected["name"]])["model"],
            "name": selected["name"],
            "feature_columns": feat_cols,
            "snapshot": snapshot,
        },
        selected_path,
    )
    (models_dir / "selected.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"Selected model: {selected['name']}")
    print(
        f"Why: highest out-of-fold balanced accuracy "
        f"({selected['overall']['balanced_accuracy']:.3f}); "
        f"tie-break order is logistic_regression > lda > lightgbm."
    )
    return {"selected_path": selected_path, "results": results, "selected": selected["name"]}
