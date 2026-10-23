"""Per-leaf spectral features from eroded pixel caches."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from bls.config import Config, ensure_dir, set_seed
from bls.inventory import run_inventory
from bls.leafmap import merge_leaf_map

_LOG = logging.getLogger(__name__)


def safe_div(a: np.ndarray | float, b: np.ndarray | float, eps: float = 1e-8) -> np.ndarray | float:
    """Elementwise a/b with zeros (not inf/nan) where |b| <= eps."""
    a_arr, b_arr = np.broadcast_arrays(
        np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    )
    out = np.zeros_like(a_arr, dtype=np.float64)
    ok = np.abs(b_arr) > eps
    np.divide(a_arr, b_arr, out=out, where=ok)
    if out.ndim == 0:
        return float(out)
    return out


def snv(spectra: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Standard normal variate along the last axis."""
    mean = spectra.mean(axis=-1, keepdims=True)
    std = spectra.std(axis=-1, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (spectra - mean) / std


def _half_width(cfg: Config) -> int:
    return int(cfg.get("band", "half_width", default=2))


def _eps(cfg: Config) -> float:
    return float(cfg.get("features", "eps", default=1e-8))


def sample_at(spectrum: np.ndarray, wavelengths: np.ndarray, nm: float, half_width: int) -> np.ndarray:
    idx = int(np.argmin(np.abs(wavelengths - float(nm))))
    lo = max(0, idx - half_width)
    hi = min(spectrum.shape[-1], idx + half_width + 1)
    return np.mean(spectrum[..., lo:hi], axis=-1)


def vegetation_indices(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    cfg: Config,
) -> dict[str, np.ndarray]:
    """Compute all configured vegetation indices. *spectrum* is (..., bands)."""
    hw = _half_width(cfg)
    eps = _eps(cfg)
    wl = cfg.require("wavelengths")

    def R(nm_key: str, i: int = 0) -> np.ndarray:
        pair = wl[nm_key]
        nm = float(pair[i] if isinstance(pair, (list, tuple)) else pair)
        return sample_at(spectrum, wavelengths, nm, hw)

    ndvi = safe_div(R("ndvi", 0) - R("ndvi", 1), R("ndvi", 0) + R("ndvi", 1), eps)
    gndvi = safe_div(R("gndvi", 0) - R("gndvi", 1), R("gndvi", 0) + R("gndvi", 1), eps)
    ndre = safe_div(R("ndre", 0) - R("ndre", 1), R("ndre", 0) + R("ndre", 1), eps)
    pri = safe_div(R("pri", 0) - R("pri", 1), R("pri", 0) + R("pri", 1), eps)
    pssra = safe_div(R("pssra", 0), R("pssra", 1), eps)
    ctr2 = safe_div(R("ctr2", 0), R("ctr2", 1), eps)
    sipi = safe_div(R("sipi", 0) - R("sipi", 1), R("sipi", 0) - R("sipi", 2), eps)
    mnd705 = safe_div(
        R("mnd705", 0) - R("mnd705", 1),
        R("mnd705", 0) + R("mnd705", 1) - 2.0 * R("mnd705", 2),
        eps,
    )
    msr705 = safe_div(R("msr705", 0) - R("msr705", 1), R("msr705", 2) - R("msr705", 1), eps)
    ari = safe_div(1.0, R("ari", 0), eps) - safe_div(1.0, R("ari", 1), eps)
    # REP = 700 + 40 * ((R670 + R780)/2 - R700) / (R740 - R700)
    r670, r780, r700, r740 = R("rep", 0), R("rep", 1), R("rep", 2), R("rep", 3)
    rep = 700.0 + 40.0 * safe_div((r670 + r780) / 2.0 - r700, r740 - r700, eps)
    wi = safe_div(R("wi", 0), R("wi", 1), eps)
    return {
        "NDVI": ndvi,
        "GNDVI": gndvi,
        "NDRE": ndre,
        "PRI": pri,
        "PSSRa": pssra,
        "CTR2": ctr2,
        "SIPI": sipi,
        "mND705": mnd705,
        "mSR705": msr705,
        "ARI": ari,
        "REP": rep,
        "WI": wi,
    }


def red_edge_derivative(
    mean_spectrum: np.ndarray,
    wavelengths: np.ndarray,
    cfg: Config,
) -> dict[str, float]:
    """Savitzky–Golay first-derivative red-edge peak position (nm) and amplitude."""
    lo, hi = [float(x) for x in cfg.require("wavelengths", "red_edge_deriv")]
    window = int(cfg.require("features", "savgol_window"))
    poly = int(cfg.require("features", "savgol_polyorder"))
    n = mean_spectrum.size
    if window % 2 == 0:
        window += 1
    window = min(window, n if n % 2 == 1 else n - 1)
    window = max(window, poly + 2 if (poly + 2) % 2 == 1 else poly + 3)
    if n < window:
        return {"RE_deriv_peak_nm": float("nan"), "RE_deriv_peak_amp": float("nan")}
    d1 = savgol_filter(mean_spectrum, window_length=window, polyorder=poly, deriv=1, mode="interp")
    sel = (wavelengths >= lo) & (wavelengths <= hi)
    if not np.any(sel):
        return {"RE_deriv_peak_nm": float("nan"), "RE_deriv_peak_amp": float("nan")}
    i = int(np.argmax(d1[sel]))
    wl_sel = wavelengths[sel]
    amp_sel = d1[sel]
    return {
        "RE_deriv_peak_nm": float(wl_sel[i]),
        "RE_deriv_peak_amp": float(amp_sel[i]),
    }


def downsample_spectrum(
    mean_spectrum: np.ndarray,
    wavelengths: np.ndarray,
    step_nm: float,
) -> dict[str, float]:
    """Mean reflectance in consecutive *step_nm* bins; columns R_<nm>."""
    if wavelengths.size == 0:
        return {}
    start = float(np.floor(wavelengths.min() / step_nm) * step_nm)
    stop = float(np.ceil(wavelengths.max() / step_nm) * step_nm)
    edges = np.arange(start, stop + step_nm, step_nm)
    out: dict[str, float] = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (wavelengths >= lo) & (wavelengths < hi)
        if not np.any(sel):
            continue
        center = int(round((lo + hi) / 2.0))
        out[f"R_{center}"] = float(np.mean(mean_spectrum[sel]))
    return out


def _finite(value: Any, fill: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return fill
    return fill if not np.isfinite(v) else v


INDEX_NAMES = [
    "NDVI",
    "GNDVI",
    "NDRE",
    "PRI",
    "PSSRa",
    "CTR2",
    "SIPI",
    "mND705",
    "mSR705",
    "ARI",
    "REP",
    "WI",
]


def features_from_pixels(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    cfg: Config,
) -> dict[str, float]:
    """Features from one leaf's eroded pixel spectra, shape (n_pix, n_bands)."""
    eps = _eps(cfg)
    if spectra.size == 0:
        raise ValueError("Leaf has no eroded pixels")

    spec = np.asarray(spectra, dtype=np.float64)
    spec = np.where(np.isfinite(spec), spec, 0.0)
    norm = str(cfg.get("features", "spectrum_norm", default="none")).lower()
    if norm == "snv":
        spec = snv(spec, eps=eps)

    mean_spec = spec.mean(axis=0)
    idx_mean = vegetation_indices(mean_spec, wavelengths, cfg)
    idx_pix = vegetation_indices(spec, wavelengths, cfg)

    out: dict[str, float] = {}
    for name in INDEX_NAMES:
        out[name] = _finite(idx_mean[name])
        pix = np.asarray(idx_pix[name], dtype=np.float64)
        pix = pix[np.isfinite(pix)]
        out[f"{name}_median"] = _finite(np.median(pix) if pix.size else 0.0)

    out.update(red_edge_derivative(mean_spec, wavelengths, cfg))
    out["RE_deriv_peak_nm"] = _finite(out["RE_deriv_peak_nm"])
    out["RE_deriv_peak_amp"] = _finite(out["RE_deriv_peak_amp"])

    ndvi_pix = np.asarray(idx_pix["NDVI"], dtype=np.float64)
    pri_pix = np.asarray(idx_pix["PRI"], dtype=np.float64)
    ndvi_pix = ndvi_pix[np.isfinite(ndvi_pix)]
    pri_pix = pri_pix[np.isfinite(pri_pix)]
    low_thr = float(cfg.require("features", "ndvi_low_threshold"))

    def _pct(arr: np.ndarray, q: float) -> float:
        return _finite(np.percentile(arr, q) if arr.size else 0.0)

    out["NDVI_std"] = _finite(ndvi_pix.std() if ndvi_pix.size else 0.0)
    out["NDVI_p5"] = _pct(ndvi_pix, 5)
    out["NDVI_p95"] = _pct(ndvi_pix, 95)
    out["PRI_std"] = _finite(pri_pix.std() if pri_pix.size else 0.0)
    out["PRI_p5"] = _pct(pri_pix, 5)
    out["PRI_p95"] = _pct(pri_pix, 95)
    out["NDVI_low_frac"] = _finite((ndvi_pix < low_thr).mean() if ndvi_pix.size else 0.0)

    step = float(cfg.require("features", "spectrum_step_nm"))
    for k, v in downsample_spectrum(mean_spec, wavelengths, step).items():
        out[k] = _finite(v)
    return out


def load_leaf_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def run_features(cfg: Config, force: bool = False) -> pd.DataFrame:
    set_seed(cfg)
    out_path = cfg.path("paths", "features")
    ensure_dir(out_path.parent)
    if out_path.exists() and not force:
        _LOG.info("features.csv exists (%s); skip (use --force to rebuild)", out_path)
        return pd.read_csv(out_path)

    leaves_path = cfg.path("paths", "leaves")
    if not leaves_path.exists():
        raise FileNotFoundError(f"Run scripts/run_segment.py first; missing {leaves_path}")
    leaves = pd.read_csv(leaves_path)
    leaves = merge_leaf_map(cfg, leaves)

    cache_dir = cfg.path("paths", "cache_dir")
    rows: list[dict[str, Any]] = []
    for rec in leaves.itertuples(index=False):
        npz = cache_dir / f"{rec.stem}__leaf{int(rec.leaf_no):02d}.npz"
        if not npz.exists():
            raise FileNotFoundError(f"Missing leaf cache {npz}")
        blob = load_leaf_npz(npz)
        feats = features_from_pixels(blob["spectra"], blob["wavelengths"], cfg)
        meta = {c: getattr(rec, c) for c in leaves.columns}
        rows.append({**meta, **feats})
        _LOG.info("features %s leaf %02d NDVI=%.3f", rec.stem, int(rec.leaf_no), feats["NDVI"])

    df = pd.DataFrame(rows)
    # Guard: no inf
    num = df.select_dtypes(include=[np.number])
    df[num.columns] = num.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df.to_csv(out_path, index=False)
    _LOG.info("Wrote %s (%d leaves, %d columns)", out_path, len(df), df.shape[1])
    return df


def print_feature_summary(df: pd.DataFrame) -> None:
    cols = [c for c in INDEX_NAMES + [
        "RE_deriv_peak_nm",
        "RE_deriv_peak_amp",
        "NDVI_std",
        "NDVI_p5",
        "NDVI_p95",
        "PRI_std",
        "NDVI_low_frac",
    ] if c in df.columns]
    print("\n=== Feature summary (per leaf) ===")
    if not cols:
        print("  (no index columns)")
        return
    print(df[cols].describe().T.to_string(float_format=lambda x: f"{x:8.4f}"))
    if "NDVI" in df.columns:
        print(f"\nLeaf NDVI range: {df['NDVI'].min():.3f} – {df['NDVI'].max():.3f}  (expect ~0.6–0.9)")
    print()
