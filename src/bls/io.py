"""ENVI cube I/O, band lookup, and false-colour RGB."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from bls.config import Config

_LOG = logging.getLogger(__name__)

# ENVI data-type code → numpy dtype (fallback if config is incomplete).
_ENVI_DTYPE = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    12: np.uint16,
}


def parse_envi_header(hdr_path: Path) -> dict[str, Any]:
    """Parse an ENVI .hdr file into a dict of raw string values plus typed fields."""
    text = Path(hdr_path).read_text(encoding="utf-8", errors="replace")
    # Drop the leading "ENVI" token if present.
    lines = text.splitlines()
    if lines and lines[0].strip().upper() == "ENVI":
        lines = lines[1:]

    merged: list[str] = []
    buf = ""
    depth = 0
    for line in lines:
        depth += line.count("{") - line.count("}")
        buf = f"{buf}\n{line}" if buf else line
        if depth <= 0:
            merged.append(buf)
            buf = ""
            depth = 0
    if buf:
        merged.append(buf)

    raw: dict[str, str] = {}
    for item in merged:
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        raw[key.strip().lower()] = val.strip()

    def _int(name: str, default: int | None = None) -> int | None:
        v = raw.get(name)
        if v is None or v == "":
            return default
        try:
            return int(float(v))
        except ValueError:
            return default

    def _float(name: str, default: float | None = None) -> float | None:
        v = raw.get(name)
        if v is None or v == "":
            return default
        try:
            return float(v)
        except ValueError:
            return default

    wavelengths = _parse_wavelengths(raw.get("wavelength", ""))
    header: dict[str, Any] = {
        "raw": raw,
        "lines": _int("lines"),
        "samples": _int("samples"),
        "bands": _int("bands"),
        "data_type": _int("data type"),
        "header_offset": _int("header offset", 0) or 0,
        "byte_order": _int("byte order", 0) or 0,
        "interleave": raw.get("interleave", "bil").strip().lower().strip("{} ").strip(),
        "reflectance_scale_factor": _float("reflectance scale factor"),
        "wavelengths": wavelengths,
        "wavelength_min": float(wavelengths[0]) if len(wavelengths) else None,
        "wavelength_max": float(wavelengths[-1]) if len(wavelengths) else None,
        "label": raw.get("label", "").strip().strip("{}").strip(),
    }
    return header


def _parse_wavelengths(value: str) -> np.ndarray:
    inner = value
    if "{" in value and "}" in value:
        inner = value[value.find("{") + 1 : value.rfind("}")]
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", inner)
    if not nums:
        return np.array([], dtype=np.float64)
    return np.array([float(x) for x in nums], dtype=np.float64)


def load_cube(
    bil_path: Path | str,
    hdr_path: Path | str,
    cfg: Config | None = None,
    trim: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load an ENVI BIL cube as float32 reflectance [rows, cols, bands] plus wavelengths.

    Uses a memmap where possible. Divides by the header reflectance scale factor
    when present (config: io.use_header_reflectance_scale). Optionally trims
    bands to wavelengths.trim_min_nm–trim_max_nm.
    """
    bil_path = Path(bil_path)
    hdr_path = Path(hdr_path)
    if not hdr_path.exists():
        raise FileNotFoundError(f"Missing header: {hdr_path}")
    if not bil_path.exists():
        raise FileNotFoundError(f"Missing cube: {bil_path}")

    header = parse_envi_header(hdr_path)
    wavelengths = np.asarray(header["wavelengths"], dtype=np.float64)
    n_lines = header["lines"]
    n_samples = header["samples"]
    n_bands = header["bands"]
    if None in (n_lines, n_samples, n_bands):
        raise ValueError(f"Header {hdr_path} is missing lines/samples/bands")

    scale = _reflectance_scale(header, cfg)
    dtype = _cube_dtype(cfg)
    try:
        import spectral.io.envi as envi

        img = envi.open(str(hdr_path), image=str(bil_path))
        try:
            mmap = img.open_memmap(interleave="bip", writable=False)
            cube = _scaled_copy(mmap, scale, dtype)
        except Exception:
            _LOG.warning("Memmap failed for %s; loading via load()", bil_path.name)
            cube = _scaled_copy(np.asarray(img.load()), scale, dtype)
    except Exception as exc:
        _LOG.info("spectral.io.envi failed (%s); using raw memmap for %s", exc, bil_path.name)
        cube = _load_bil_raw(bil_path, header, scale=scale, dtype=dtype)

    if cube.ndim != 3:
        raise ValueError(f"Expected 3-D cube, got shape {cube.shape} from {bil_path}")

    # spectral may return (lines, samples, bands) already for BIP memmap.
    # If the last axis is not bands, try to fix common layouts.
    if cube.shape[2] != n_bands:
        if cube.shape[0] == n_bands:
            cube = np.transpose(cube, (1, 2, 0))
        elif cube.shape[1] == n_bands:
            cube = np.transpose(cube, (0, 2, 1))

    if wavelengths.size != cube.shape[2]:
        _LOG.warning(
            "Wavelength count %d != cube bands %d for %s",
            wavelengths.size,
            cube.shape[2],
            bil_path.name,
        )

    if trim and cfg is not None:
        cube, wavelengths = trim_bands(cube, wavelengths, cfg)

    _LOG.info(
        "Loaded %s shape=%s wl=[%.1f, %.1f] nm",
        bil_path.name,
        tuple(cube.shape),
        float(wavelengths[0]) if wavelengths.size else float("nan"),
        float(wavelengths[-1]) if wavelengths.size else float("nan"),
    )
    return cube, wavelengths


def _cube_dtype(cfg: Config | None) -> np.dtype:
    name = "float16"
    if cfg is not None:
        name = str(cfg.get("io", "cube_dtype", default="float16"))
    return np.dtype(name)


def _scaled_copy(src: np.ndarray, scale: float, dtype: np.dtype, row_chunk: int = 32) -> np.ndarray:
    """Copy a cube to *dtype* reflectance without holding a full float32 duplicate."""
    scale_f = np.float32(scale if scale else 1.0)
    out = np.empty(src.shape, dtype=dtype)
    for i in range(0, src.shape[0], row_chunk):
        sl = np.asarray(src[i : i + row_chunk], dtype=np.float32)
        if scale_f != 1.0:
            sl /= scale_f
        out[i : i + row_chunk] = sl
    return out


def _load_bil_raw(
    bil_path: Path,
    header: dict[str, Any],
    scale: float = 1.0,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    dtype_code = int(header["data_type"] or 12)
    raw_dtype = _ENVI_DTYPE.get(dtype_code, np.uint16)
    n_lines = int(header["lines"])
    n_samples = int(header["samples"])
    n_bands = int(header["bands"])
    offset = int(header.get("header_offset") or 0)
    mmap = np.memmap(bil_path, dtype=raw_dtype, mode="r", offset=offset)
    expected = n_lines * n_samples * n_bands
    if mmap.size < expected:
        raise ValueError(
            f"{bil_path} has {mmap.size} elements, expected at least {expected}"
        )
    # BIL: (lines, bands, samples)
    shaped = mmap[:expected].reshape(n_lines, n_bands, n_samples)
    arr = _scaled_copy(shaped, scale, dtype or np.dtype("float16"))
    return np.transpose(arr, (0, 2, 1))  # (lines, samples, bands)


def _reflectance_scale(header: dict[str, Any], cfg: Config | None) -> float:
    use_header = True
    fallback = 10000.0
    if cfg is not None:
        use_header = bool(cfg.get("io", "use_header_reflectance_scale", default=True))
        fallback = float(cfg.get("io", "fallback_reflectance_scale", default=fallback))
    if use_header and header.get("reflectance_scale_factor"):
        return float(header["reflectance_scale_factor"])
    return fallback


def trim_bands(
    cube: np.ndarray, wavelengths: np.ndarray, cfg: Config
) -> tuple[np.ndarray, np.ndarray]:
    lo = float(cfg.require("wavelengths", "trim_min_nm"))
    hi = float(cfg.require("wavelengths", "trim_max_nm"))
    keep = (wavelengths >= lo) & (wavelengths <= hi)
    if not np.any(keep):
        raise ValueError(f"No bands in trim range {lo}–{hi} nm")
    return cube[:, :, keep], wavelengths[keep]


def nearest_band_index(wavelengths: np.ndarray, nm: float) -> int:
    return int(np.argmin(np.abs(wavelengths - float(nm))))


def band(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    nm: float,
    half_width: int = 2,
) -> np.ndarray:
    """Mean of bands within ±half_width indices of the nearest wavelength (noise reduction)."""
    idx = nearest_band_index(wavelengths, nm)
    lo = max(0, idx - int(half_width))
    hi = min(cube.shape[2], idx + int(half_width) + 1)
    return np.mean(cube[:, :, lo:hi], axis=2)


def band_value(spectrum: np.ndarray, wavelengths: np.ndarray, nm: float, half_width: int = 2) -> np.ndarray:
    """Like band() but for a 1-D spectrum or (n_pix, n_bands) array on the last axis."""
    idx = nearest_band_index(wavelengths, nm)
    lo = max(0, idx - int(half_width))
    hi = min(spectrum.shape[-1], idx + int(half_width) + 1)
    return np.mean(spectrum[..., lo:hi], axis=-1)


def false_rgb(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    cfg: Config | None = None,
) -> np.ndarray:
    """False-colour RGB at 650/550/470 nm, scaled by the 99th percentile, clipped to 0–1."""
    rgb_nm = [650.0, 550.0, 470.0]
    half = 2
    pct = 99.0
    if cfg is not None:
        rgb_nm = list(cfg.require("wavelengths", "rgb_nm"))
        half = int(cfg.get("band", "half_width", default=2))
        pct = float(cfg.get("io", "rgb_percentile", default=99.0))
    channels = [band(cube, wavelengths, float(nm), half_width=half) for nm in rgb_nm]
    rgb = np.stack(channels, axis=-1).astype(np.float32)
    finite = rgb[np.isfinite(rgb)]
    scale = float(np.percentile(finite, pct)) if finite.size else 1.0
    if scale <= 0:
        scale = 1.0
    rgb = np.clip(rgb / scale, 0.0, 1.0)
    return rgb


def ndvi_map(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    cfg: Config | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """NDVI = (R800 − R670) / (R800 + R670) as a 2-D map."""
    nir_nm, red_nm = 800.0, 670.0
    half = 2
    if cfg is not None:
        pair = cfg.get("wavelengths", "ndvi", default=[800.0, 670.0])
        nir_nm, red_nm = float(pair[0]), float(pair[1])
        half = int(cfg.get("band", "half_width", default=2))
        eps = float(cfg.get("features", "eps", default=eps))
    nir = band(cube, wavelengths, nir_nm, half_width=half)
    red = band(cube, wavelengths, red_nm, half_width=half)
    denom = nir + red
    out = np.zeros_like(nir, dtype=np.float32)
    ok = np.abs(denom) > eps
    out[ok] = (nir[ok] - red[ok]) / denom[ok]
    return out


def central_crop(arr: np.ndarray, frac: float) -> np.ndarray:
    """Return the central *frac* (0–1) of a 2-D or 3-D array along rows and cols."""
    frac = min(max(float(frac), 0.05), 1.0)
    rows, cols = arr.shape[0], arr.shape[1]
    h = max(1, int(round(rows * frac)))
    w = max(1, int(round(cols * frac)))
    r0 = (rows - h) // 2
    c0 = (cols - w) // 2
    return arr[r0 : r0 + h, c0 : c0 + w]


def save_false_rgb_png(rgb: np.ndarray, path: Path) -> Path:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return path


def plot_top_ndvi_spectrum(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    ndvi: np.ndarray,
    path: Path,
    n_pixels: int,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flat = ndvi.reshape(-1)
    finite = np.isfinite(flat)
    n = min(int(n_pixels), int(finite.sum()))
    order = np.argsort(np.where(finite, flat, -np.inf))[-n:]
    rows = order // ndvi.shape[1]
    cols = order % ndvi.shape[1]
    spectra = cube[rows, cols, :]
    mean = spectra.mean(axis=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(wavelengths, mean, color="green")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("reflectance")
    ax.set_title(f"Mean spectrum of the {n} highest-NDVI pixels")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {
        "n_pixels": n,
        "ndvi_min": float(np.min(flat[order])),
        "ndvi_max": float(np.max(flat[order])),
        "spectrum_min": float(np.min(mean)),
        "spectrum_median": float(np.median(mean)),
        "spectrum_max": float(np.max(mean)),
        "path": str(path),
    }


def run_loader_qc(cfg: Config, stem: str | None = None) -> dict[str, Any]:
    """Loader check for one in-scope scan: shape, wl range, central-crop stats, RGB, NDVI spectrum."""
    from bls.inventory import run_inventory

    inv = run_inventory(cfg, force=False)
    in_scope = inv[inv["in_scope"] & inv["pair_ok"]]
    if in_scope.empty:
        raise RuntimeError("No in-scope scans to preview")
    if stem is None:
        row = in_scope.iloc[0]
    else:
        hit = in_scope[in_scope["stem"] == stem]
        if hit.empty:
            raise FileNotFoundError(f"{stem} is not an in-scope scan")
        row = hit.iloc[0]
    cube, wl = load_cube(row["bil_path"], row["hdr_path"], cfg=cfg, trim=True)
    stats = preview_stats(cube, wl, cfg)
    stats["stem"] = row["stem"]
    fig_dir = Path(cfg.path("paths", "figures_dir"))
    fig_dir.mkdir(parents=True, exist_ok=True)
    rgb = false_rgb(cube, wl, cfg)
    rgb_path = save_false_rgb_png(rgb, fig_dir / f"{row['stem']}_false_rgb.png")
    stats["false_rgb"] = str(rgb_path)
    ndvi = ndvi_map(cube, wl, cfg)
    n_top = int(cfg.get("io", "preview_top_ndvi_pixels", default=1000))
    spec_stats = plot_top_ndvi_spectrum(
        cube, wl, ndvi, fig_dir / f"{row['stem']}_top_ndvi_spectrum.png", n_top
    )
    stats["top_ndvi"] = spec_stats
    return stats


def preview_stats(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    cfg: Config,
) -> dict[str, Any]:
    """Shape, wavelength range, and min/median/max of a central crop (loader QC)."""
    frac = float(cfg.get("io", "preview_central_frac", default=0.25))
    crop = central_crop(cube, frac)
    finite = crop[np.isfinite(crop)]
    stats = {
        "shape": tuple(int(x) for x in cube.shape),
        "wavelength_min": float(wavelengths[0]),
        "wavelength_max": float(wavelengths[-1]),
        "n_bands": int(wavelengths.size),
        "crop_frac": frac,
        "crop_shape": tuple(int(x) for x in crop.shape),
        "reflectance_min": float(np.min(finite)) if finite.size else float("nan"),
        "reflectance_median": float(np.median(finite)) if finite.size else float("nan"),
        "reflectance_max": float(np.max(finite)) if finite.size else float("nan"),
    }
    return stats
