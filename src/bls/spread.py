"""Early-change (days 0–3) summary: P(infected), atypical pixel %, median distance."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bls.anomaly import run_anomaly
from bls.config import Config, ensure_dir, set_seed
from bls.explore import _ordered_time_points
from bls.leafmap import UNCONFIRMED_WARNING, assert_ready_for_training, merge_leaf_map

_LOG = logging.getLogger(__name__)


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _mean_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(v.mean())
    rng = np.random.default_rng(seed)
    boots = np.array([v[rng.integers(0, v.size, v.size)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean, float(lo), float(hi)


def load_oof(cfg: Config) -> pd.DataFrame:
    path = cfg.path("paths", "reports_dir") / "oof_predictions.csv"
    if not path.exists():
        _LOG.warning("No oof_predictions.csv; P(infected) trends will be empty until training runs")
        return pd.DataFrame()
    return pd.read_csv(path)


def summarise(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    n_boot = int(cfg.get("spread", "n_bootstrap", default=1000))
    seed = cfg.seed()
    rows = []
    tps = _ordered_time_points(df, cfg)
    for tp in tps:
        for group in ("control", "inoculated"):
            part = df[(df["time_point"] == tp) & (df["group"].astype(str).str.lower() == group)]
            rec = {"time_point": tp, "group": group, "n": int(len(part))}
            for col, key in (
                ("P_infected", "p_infected"),
                ("atypical_pixel_pct", "atypical_pct"),
                ("median_distance", "median_distance"),
            ):
                if col not in part.columns:
                    rec[f"{key}_mean"] = float("nan")
                    rec[f"{key}_lo"] = float("nan")
                    rec[f"{key}_hi"] = float("nan")
                    continue
                mean, lo, hi = _mean_ci(part[col].to_numpy(), n_boot, seed)
                rec[f"{key}_mean"] = mean
                rec[f"{key}_lo"] = lo
                rec[f"{key}_hi"] = hi
            rows.append(rec)
    return pd.DataFrame(rows)


def auc_table(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for tp in _ordered_time_points(df, cfg):
        part = df[df["time_point"] == tp]
        y = (part["group"].astype(str).str.lower() == "inoculated").to_numpy(dtype=int)
        rec = {"time_point": tp, "n": int(len(part)), "n_inoculated": int(y.sum()), "n_control": int((1 - y).sum())}
        for col, key in (("P_infected", "p_infected"), ("atypical_pixel_pct", "atypical_pct")):
            if col not in part.columns or len(np.unique(y)) < 2:
                rec[f"auc_{key}"] = float("nan")
                continue
            scores = part[col].to_numpy(dtype=float)
            ok = np.isfinite(scores)
            try:
                rec[f"auc_{key}"] = float(roc_auc_score(y[ok], scores[ok])) if ok.sum() and len(np.unique(y[ok])) > 1 else float("nan")
            except ValueError:
                rec[f"auc_{key}"] = float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


def plot_lines(summary: pd.DataFrame, cfg: Config, fig_dir: Path) -> list[Path]:
    plt = _mpl()
    tps = list(dict.fromkeys(summary["time_point"].tolist()))
    x = np.arange(len(tps))
    paths = []
    for key, ylabel, fname in (
        ("p_infected", "out-of-fold P(infected)  mean ± 95% CI", "spread_p_infected.png"),
        ("atypical_pct", "atypical pixel %  mean ± 95% CI\n(exploratory: vs healthy reference)", "spread_atypical_pct.png"),
        ("median_distance", "median Mahalanobis distance  mean ± 95% CI\n(exploratory: vs healthy reference)", "spread_median_distance.png"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for group, color in (("control", "#1f77b4"), ("inoculated", "#d62728")):
            sub = summary[summary["group"] == group].set_index("time_point").reindex(tps)
            mean = sub[f"{key}_mean"].to_numpy(dtype=float)
            lo = sub[f"{key}_lo"].to_numpy(dtype=float)
            hi = sub[f"{key}_hi"].to_numpy(dtype=float)
            n = sub["n"].to_numpy()
            ax.plot(x, mean, marker="o", color=color, label=f"{group}")
            ax.fill_between(x, lo, hi, color=color, alpha=0.2)
            for xi, ni in zip(x, n):
                ax.text(xi, ax.get_ylim()[0] if False else 0, "", fontsize=7)
            ax.set_xticks(x, [f"{tp}\nn={int(n[i]) if np.isfinite(n[i]) else 0}" for i, tp in enumerate(tps)])
        ax.set_ylabel(ylabel)
        ax.set_title("Lines should overlap at bi; later divergence is the early signal")
        ax.legend()
        fig.tight_layout()
        path = fig_dir / fname
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_gallery(df: pd.DataFrame, cfg: Config, fig_dir: Path) -> Path | None:
    """One inoculated and one control leaf per time point: false colour + heatmap."""
    from PIL import Image

    plt = _mpl()
    heat_dir = cfg.path("paths", "figures_dir") / "heatmaps"
    overlay_dir = cfg.path("paths", "overlays_dir")
    tps = _ordered_time_points(df, cfg)
    chosen: list[tuple[str, pd.Series]] = []
    for tp in tps:
        for group in ("control", "inoculated"):
            part = df[(df["time_point"] == tp) & (df["group"].astype(str).str.lower() == group)]
            if part.empty:
                continue
            # Prefer a leaf with a heatmap file.
            if "heatmap" in part.columns and part["heatmap"].astype(str).str.len().gt(0).any():
                rec = part[part["heatmap"].astype(str).str.len() > 0].iloc[0]
            else:
                rec = part.iloc[0]
            chosen.append((f"{tp} {group}", rec))
    if not chosen:
        return None

    fig, axes = plt.subplots(len(chosen), 2, figsize=(8, 2.6 * len(chosen)))
    if len(chosen) == 1:
        axes = np.array([axes])
    for i, (title, rec) in enumerate(chosen):
        overlay = overlay_dir / f"{rec['stem']}.png"
        heat = Path(str(rec.get("heatmap", ""))) if "heatmap" in rec.index else heat_dir / f"{rec['stem']}__leaf{int(rec['leaf_no']):02d}_heatmap.png"
        for j, path in enumerate((overlay, heat)):
            ax = axes[i, j]
            ax.set_axis_off()
            if path.exists():
                ax.imshow(np.array(Image.open(path)))
            ax.set_title(f"{title}  leaf {int(rec['leaf_no'])}" + (" overlay" if j == 0 else " heatmap"), fontsize=9)
    fig.suptitle("Gallery: exploratory spectral difference from healthy reference (not disease area)")
    fig.tight_layout()
    path = fig_dir / "spread_gallery.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_markdown(summary: pd.DataFrame, aucs: pd.DataFrame, reports: Path) -> Path:
    lines = [
        "# Early-change analysis (days 0–3)",
        "",
        "All pixel maps are **exploratory: spectral difference from healthy reference**, never disease area.",
        "Healthy reference = bi leaves (both groups) + control leaves at all time points.",
        "Distances use a leave-one-unit-out reference when unit labels are present.",
        "",
        UNCONFIRMED_WARNING,
        "",
        "Lines should overlap at bi; any divergence later is the early signal.",
        "",
        "## Per time point × group (mean ± 95% bootstrap CI)",
        "",
        summary.to_string(index=False),
        "",
        "## AUC inoculated vs control",
        "",
        aucs.to_string(index=False),
        "",
    ]
    path = reports / "spread.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_spread(cfg: Config, force: bool = False) -> dict[str, Path]:
    set_seed(cfg)
    reports = ensure_dir(cfg.path("paths", "reports_dir"))
    fig_dir = ensure_dir(cfg.path("paths", "figures_dir"))
    md = reports / "spread.md"
    if md.exists() and not force:
        _LOG.info("spread.md exists; skip (use --force to rebuild)")
        return {"report": md}

    anomaly = run_anomaly(cfg, force=force)
    oof = load_oof(cfg)
    df = anomaly.copy()
    if not oof.empty:
        df = df.merge(
            oof[["stem", "leaf_no", "P_infected"]] if "P_infected" in oof.columns else oof[["stem", "leaf_no"]],
            on=["stem", "leaf_no"],
            how="left",
        )
    df = merge_leaf_map(cfg, df)
    try:
        assert_ready_for_training(cfg, df)
    except Exception as exc:
        _LOG.warning("Group labels incomplete; spread will still write exploratory tables: %s", exc)

    summary = summarise(df, cfg)
    aucs = auc_table(df, cfg)
    summary.to_csv(reports / "spread_summary.csv", index=False)
    aucs.to_csv(reports / "spread_auc.csv", index=False)
    plot_lines(summary, cfg, fig_dir)
    plot_gallery(df, cfg, fig_dir)
    path = write_markdown(summary, aucs, reports)
    _LOG.info("Wrote %s", path)
    return {"report": path}
