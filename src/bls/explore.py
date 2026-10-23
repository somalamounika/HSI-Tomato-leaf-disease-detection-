"""Exploratory plots and per-index tests (inoculated vs control)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bls.config import Config, ensure_dir, set_seed
from bls.features import INDEX_NAMES, run_features
from bls.leafmap import LeafMapError, UNCONFIRMED_WARNING, assert_ready_for_training, merge_leaf_map

_LOG = logging.getLogger(__name__)


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _r_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("R_") and c[2:].isdigit()]
    return sorted(cols, key=lambda c: int(c[2:]))


def _time_order_map(cfg: Config) -> dict[str, int]:
    allowed = list(cfg.require("filters", "allowed_time_points"))
    offset = int((cfg.get("time_order") or {}).get("day_offset", 1))
    out: dict[str, int] = {}
    for tp in allowed:
        if tp in (cfg.get("time_order") or {}):
            out[tp] = int(cfg.get("time_order")[tp])
        elif tp.startswith("d") and tp[1:].isdigit():
            out[tp] = int(tp[1:]) + offset
    return out


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: P(x>y) - P(x<y). x = inoculated, y = control."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    # Efficient pairwise via broadcasting for small n.
    diff = x[:, None] - y[None, :]
    n = diff.size
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / n)


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    out = np.ones(n, dtype=np.float64)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    order = np.argsort(p[finite])
    ranked = p[finite][order]
    q = ranked * n / (np.arange(len(ranked)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.empty(len(ranked), dtype=np.float64)
    tmp[order] = q
    out[finite] = tmp
    return out


def bootstrap_mean_diff_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap CI of mean(a) - mean(b) along axis 0. a, b: (n, bands)."""
    rng = np.random.default_rng(seed)
    n_a, n_b = a.shape[0], b.shape[0]
    n_bands = a.shape[1]
    diffs = np.empty((n_boot, n_bands), dtype=np.float64)
    for i in range(n_boot):
        aa = a[rng.integers(0, n_a, n_a)].mean(axis=0) if n_a else np.zeros(n_bands)
        bb = b[rng.integers(0, n_b, n_b)].mean(axis=0) if n_b else np.zeros(n_bands)
        diffs[i] = aa - bb
    lo = np.percentile(diffs, 100 * alpha / 2, axis=0)
    hi = np.percentile(diffs, 100 * (1 - alpha / 2), axis=0)
    return diffs.mean(axis=0), lo, hi


def _split_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = df["group"].astype(str).str.lower()
    return df[g == "inoculated"], df[g == "control"]


def plot_mean_spectra(df: pd.DataFrame, cfg: Config, fig_dir: Path) -> Path:
    plt = _mpl()
    rcols = _r_columns(df)
    nm = np.array([int(c[2:]) for c in rcols], dtype=float)
    tps = _ordered_time_points(df, cfg)
    fig, axes = plt.subplots(1, len(tps), figsize=(4.2 * len(tps), 4.0), sharey=True)
    if len(tps) == 1:
        axes = [axes]
    for ax, tp in zip(axes, tps):
        part = df[df["time_point"] == tp]
        n_total = len(part)
        for group, color in (("control", "#1f77b4"), ("inoculated", "#d62728")):
            sub = part[part["group"].astype(str).str.lower() == group]
            if sub.empty:
                continue
            mat = sub[rcols].to_numpy(dtype=float)
            mean = mat.mean(axis=0)
            sd = mat.std(axis=0)
            ax.plot(nm, mean, color=color, label=f"{group} n={len(sub)}")
            ax.fill_between(nm, mean - sd, mean + sd, color=color, alpha=0.2)
        ax.set_title(f"{tp} (n={n_total})")
        ax.set_xlabel("nm")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("reflectance (mean ± sd)")
    fig.suptitle("Mean spectrum ± sd per group")
    fig.tight_layout()
    path = fig_dir / "mean_spectra_by_time.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_diff_spectra(df: pd.DataFrame, cfg: Config, fig_dir: Path) -> Path:
    plt = _mpl()
    rcols = _r_columns(df)
    nm = np.array([int(c[2:]) for c in rcols], dtype=float)
    tps = _ordered_time_points(df, cfg)
    n_boot = int(cfg.get("models", "n_bootstrap", default=1000))
    seed = cfg.seed()
    fig, axes = plt.subplots(1, len(tps), figsize=(4.2 * len(tps), 4.0), sharey=True)
    if len(tps) == 1:
        axes = [axes]
    for ax, tp in zip(axes, tps):
        part = df[df["time_point"] == tp]
        inoc, ctrl = _split_groups(part)
        ax.axhline(0, color="grey", lw=0.8)
        if inoc.empty or ctrl.empty:
            ax.set_title(f"{tp} (missing group)")
            continue
        a = inoc[rcols].to_numpy(dtype=float)
        b = ctrl[rcols].to_numpy(dtype=float)
        mean, lo, hi = bootstrap_mean_diff_ci(a, b, n_boot, seed)
        ax.plot(nm, mean, color="#d62728")
        ax.fill_between(nm, lo, hi, color="#d62728", alpha=0.25)
        ax.set_title(f"{tp} n_inoc={len(inoc)} n_ctrl={len(ctrl)}")
        ax.set_xlabel("nm")
    axes[0].set_ylabel("inoculated − control")
    fig.suptitle("Difference spectrum with bootstrap 95% CI")
    fig.tight_layout()
    path = fig_dir / "diff_spectra_by_time.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_index_lines(df: pd.DataFrame, cfg: Config, fig_dir: Path) -> Path:
    plt = _mpl()
    tps = _ordered_time_points(df, cfg)
    x = np.arange(len(tps))
    names = [n for n in INDEX_NAMES if n in df.columns]
    ncols = 4
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows), squeeze=False)
    for i, name in enumerate(names):
        ax = axes[i // ncols][i % ncols]
        for group, color in (("control", "#1f77b4"), ("inoculated", "#d62728")):
            means, sds, ns = [], [], []
            for tp in tps:
                sub = df[(df["time_point"] == tp) & (df["group"].astype(str).str.lower() == group)]
                vals = sub[name].to_numpy(dtype=float) if not sub.empty else np.array([])
                vals = vals[np.isfinite(vals)]
                means.append(float(vals.mean()) if vals.size else np.nan)
                sds.append(float(vals.std()) if vals.size else 0.0)
                ns.append(int(vals.size))
            ax.errorbar(x, means, yerr=sds, marker="o", color=color, label=f"{group} (n={ns})")
        ax.set_xticks(x, tps)
        ax.set_title(name)
        ax.legend(fontsize=7)
    for j in range(len(names), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle("Index mean ± sd by group across time (n per point in legend)")
    fig.tight_layout()
    path = fig_dir / "indices_by_time.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_pca(df: pd.DataFrame, cfg: Config, fig_dir: Path) -> Path:
    plt = _mpl()
    feat_cols = [c for c in INDEX_NAMES if c in df.columns]
    feat_cols += [c for c in ["NDVI_std", "NDVI_p5", "NDVI_p95", "PRI_std", "PRI_p5", "PRI_p95", "NDVI_low_frac",
                               "RE_deriv_peak_nm", "RE_deriv_peak_amp"] if c in df.columns]
    tps = _ordered_time_points(df, cfg)
    fig, axes = plt.subplots(1, len(tps), figsize=(4.2 * len(tps), 4.2), squeeze=False)
    for ax, tp in zip(axes[0], tps):
        part = df[df["time_point"] == tp].dropna(subset=feat_cols)
        ax.set_title(f"{tp} n={len(part)}")
        if len(part) < 3:
            ax.text(0.5, 0.5, "too few leaves", ha="center", va="center", transform=ax.transAxes)
            continue
        X = StandardScaler().fit_transform(part[feat_cols].to_numpy(dtype=float))
        xy = PCA(n_components=2, random_state=cfg.seed()).fit_transform(X)
        for group, color in (("control", "#1f77b4"), ("inoculated", "#d62728")):
            m = part["group"].astype(str).str.lower() == group
            ax.scatter(xy[m, 0], xy[m, 1], c=color, label=f"{group} n={int(m.sum())}", s=40)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=8)
    fig.suptitle("PCA of index + pixel-stat features")
    fig.tight_layout()
    path = fig_dir / "pca_by_time.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _ordered_time_points(df: pd.DataFrame, cfg: Config) -> list[str]:
    order = _time_order_map(cfg)
    tps = [tp for tp in df["time_point"].dropna().unique()]
    return sorted(tps, key=lambda t: order.get(str(t), 99))


def index_tests(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    names = [n for n in INDEX_NAMES if n in df.columns]
    rows: list[dict[str, Any]] = []
    for tp, part in df.groupby("time_point"):
        inoc, ctrl = _split_groups(part)
        for name in names:
            x = inoc[name].to_numpy(dtype=float) if not inoc.empty else np.array([])
            y = ctrl[name].to_numpy(dtype=float) if not ctrl.empty else np.array([])
            x = x[np.isfinite(x)]
            y = y[np.isfinite(y)]
            if x.size == 0 or y.size == 0:
                p = float("nan")
            else:
                try:
                    p = float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)
                except ValueError:
                    p = float("nan")
            rows.append(
                {
                    "time_point": tp,
                    "index": name,
                    "n_inoculated": int(x.size),
                    "n_control": int(y.size),
                    "mean_inoculated": float(x.mean()) if x.size else float("nan"),
                    "mean_control": float(y.mean()) if y.size else float("nan"),
                    "p_raw": p,
                    "cliffs_delta": cliffs_delta(x, y),
                }
            )
    out = pd.DataFrame(rows)
    out["p_bh"] = bh_fdr(out["p_raw"].to_numpy()) if not out.empty else []
    return out


def check_bi_overlap(df: pd.DataFrame, tests: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    bi = tests[tests["time_point"] == "bi"]
    if bi.empty:
        return warnings
    strong = bi[bi["p_bh"] < 0.05]
    if not strong.empty:
        names = ", ".join(strong["index"].astype(str))
        msg = (
            f"WARNING: at bi (before inoculation) groups differ on {names} "
            f"(BH p<0.05). This suggests a batch or position effect, not infection."
        )
        warnings.append(msg)
        _LOG.warning(msg)
    return warnings


def run_explore(cfg: Config, force: bool = False) -> dict[str, Path]:
    set_seed(cfg)
    fig_dir = ensure_dir(cfg.path("paths", "figures_dir"))
    reports = ensure_dir(cfg.path("paths", "reports_dir"))
    tests_path = reports / "index_tests.csv"
    marker = fig_dir / "mean_spectra_by_time.png"
    if marker.exists() and tests_path.exists() and not force:
        _LOG.info("Explore outputs exist; skip (use --force to rebuild)")
        return {"figures": fig_dir, "tests": tests_path}

    df = run_features(cfg, force=False)
    df = merge_leaf_map(cfg, df)
    df = df[df.get("include", 1) == 1].copy()
    assert_ready_for_training(cfg, df)

    paths = {
        "mean_spectra": plot_mean_spectra(df, cfg, fig_dir),
        "diff_spectra": plot_diff_spectra(df, cfg, fig_dir),
        "indices": plot_index_lines(df, cfg, fig_dir),
        "pca": plot_pca(df, cfg, fig_dir),
    }
    tests = index_tests(df, cfg)
    tests.to_csv(tests_path, index=False)
    warnings = check_bi_overlap(df, tests)
    (reports / "explore_warnings.txt").write_text(
        "\n".join(warnings) + ("\n\n" if warnings else "") + UNCONFIRMED_WARNING + "\n",
        encoding="utf-8",
    )
    _LOG.info("Explore wrote %s and %s", fig_dir, tests_path)
    return {"figures": fig_dir, "tests": tests_path}
