"""NDVI + Otsu leaf segmentation, stable numbering, overlays, and per-leaf cache."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import disk, erosion, opening, remove_small_objects

from bls.config import Config, ensure_dir, set_seed
from bls.filenames import parse_name
from bls.inventory import run_inventory
from bls.io import band, false_rgb, load_cube, ndvi_map

_LOG = logging.getLogger(__name__)


def _margins(cfg: Config) -> dict[str, int]:
    m = cfg.require("segmentation", "crop_margins")
    return {k: int(m[k]) for k in ("top", "bottom", "left", "right")}


def apply_crop_margins(mask: np.ndarray, cfg: Config) -> np.ndarray:
    """Zero out configured border strips (reflectance bars / non-cloth)."""
    top, bottom, left, right = _margins(cfg).values()
    out = mask.copy()
    if top:
        out[:top, :] = False
    if bottom:
        out[-bottom:, :] = False
    if left:
        out[:, :left] = False
    if right:
        out[:, -right:] = False
    return out


def _drop_bright_rows(mask: np.ndarray, cube: np.ndarray, wavelengths: np.ndarray, cfg: Config) -> np.ndarray:
    spec = cfg.get("segmentation", "drop_bright_rows", default=None) or {}
    if not spec or not bool(spec.get("enabled", False)):
        return mask
    nm = float(spec.get("band_nm", 650.0))
    thr = float(spec.get("min_mean_reflectance", 0.25))
    half = int(cfg.get("band", "half_width", default=2))
    plane = band(cube, wavelengths, nm, half_width=half)
    row_mean = plane.mean(axis=1)
    drop = row_mean >= thr
    out = mask.copy()
    out[drop, :] = False
    return out


def _green_gate(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    cfg: Config,
    restrict: np.ndarray | None = None,
) -> np.ndarray:
    """Pixels whose green-band reflectance exceeds Otsu (cloth is dark at 550 nm)."""
    spec = cfg.get("segmentation", "green_gate", default=None) or {}
    if not spec or not bool(spec.get("enabled", True)):
        return np.ones(cube.shape[:2], dtype=bool)
    nm = float(spec.get("band_nm", 550.0))
    floor = float(spec.get("otsu_floor", 0.05))
    half = int(cfg.get("band", "half_width", default=2))
    green = band(cube, wavelengths, nm, half_width=half)
    sample_mask = np.isfinite(green)
    if restrict is not None:
        sample_mask &= restrict
    finite = green[sample_mask]
    if finite.size == 0:
        return np.zeros(green.shape, dtype=bool)
    finite = finite[finite < 0.9]
    if finite.size == 0:
        return np.zeros(green.shape, dtype=bool)
    try:
        otsu = float(threshold_otsu(finite))
    except ValueError:
        otsu = floor
    thresh = max(otsu, floor)
    _LOG.info("green-gate band=%.0f nm otsu=%.3f thresh=%.3f", nm, otsu, thresh)
    return np.isfinite(green) & (green > thresh)


def _drop_huge_components(mask: np.ndarray, max_frac: float) -> np.ndarray:
    if max_frac <= 0:
        return mask
    labeled = label(mask, connectivity=2)
    if labeled.max() == 0:
        return mask
    limit = max_frac * mask.size
    keep = np.zeros(mask.shape, dtype=bool)
    for prop in regionprops(labeled):
        if prop.area <= limit:
            keep[labeled == prop.label] = True
        else:
            _LOG.warning("dropped component label=%d area=%d (%.1f%% of image)", prop.label, prop.area, 100 * prop.area / mask.size)
    return keep


def ndvi_leaf_mask(
    ndvi: np.ndarray,
    cfg: Config,
    cube: np.ndarray | None = None,
    wavelengths: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Leaf mask = green gate (cloth) then NDVI > max(Otsu, floor); crop / open / fill / size."""
    floor = float(cfg.require("segmentation", "ndvi_otsu_floor"))
    cap = cfg.get("segmentation", "ndvi_otsu_cap", default=None)

    valid = np.isfinite(ndvi)
    valid = apply_crop_margins(valid, cfg)
    if cube is not None and wavelengths is not None:
        valid = _drop_bright_rows(valid, cube, wavelengths, cfg)
        green_ok = _green_gate(cube, wavelengths, cfg, restrict=valid)
        candidate = valid & green_ok
    else:
        candidate = valid

    sample = ndvi[candidate]
    if sample.size == 0:
        return np.zeros(ndvi.shape, dtype=bool), floor
    try:
        otsu = float(threshold_otsu(sample))
    except ValueError:
        otsu = floor
    thresh = max(otsu, floor)
    if cap is not None:
        thresh = min(thresh, float(cap))
    mask = candidate & (ndvi > thresh)

    radius = int(cfg.require("segmentation", "opening_radius_px"))
    if radius > 0:
        mask = opening(mask, disk(radius))
    mask = ndi.binary_fill_holes(mask)

    min_area = int(cfg.require("segmentation", "min_leaf_area"))
    mask = remove_small_objects(mask.astype(bool), min_size=max(1, min_area))
    max_frac = float(cfg.get("segmentation", "max_component_frac", default=0.2))
    mask = _drop_huge_components(mask, max_frac)

    if bool(cfg.get("segmentation", "discard_edge_touching", default=True)):
        keep_ndvi = float(cfg.get("segmentation", "keep_edge_leaf_min_ndvi", default=0.4))
        mask = _drop_edge_touching(mask, ndvi=ndvi, min_area=min_area, min_ndvi=keep_ndvi)
        mask = remove_small_objects(mask.astype(bool), min_size=max(1, min_area))

    return mask.astype(bool), thresh


def _drop_edge_touching(
    mask: np.ndarray,
    ndvi: np.ndarray | None = None,
    min_area: int = 0,
    min_ndvi: float = 0.4,
) -> np.ndarray:
    """Drop edge-touching components unless they look like leaves (large + high NDVI)."""
    labeled = label(mask, connectivity=2)
    if labeled.max() == 0:
        return mask
    keep = np.zeros(labeled.shape, dtype=bool)
    rows, cols = mask.shape
    for prop in regionprops(labeled):
        minr, minc, maxr, maxc = prop.bbox
        touches = minr <= 0 or minc <= 0 or maxr >= rows or maxc >= cols
        if touches:
            mean_ndvi = float(ndvi[labeled == prop.label].mean()) if ndvi is not None else 0.0
            if not (prop.area >= min_area and mean_ndvi >= min_ndvi):
                continue
        keep[labeled == prop.label] = True
    return keep


def number_leaves(
    mask: np.ndarray, row_tolerance_px: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Stable numbering: top-to-bottom by centroid row; same-row leaves left-to-right.

    Leaves whose centroid rows differ by less than *row_tolerance_px* share a row.
    """
    labeled = label(mask, connectivity=2)
    if labeled.max() == 0:
        return np.zeros(mask.shape, dtype=np.int32), []

    items: list[dict[str, Any]] = []
    for prop in regionprops(labeled):
        cy, cx = prop.centroid
        items.append(
            {
                "label": int(prop.label),
                "centroid_row": float(cy),
                "centroid_col": float(cx),
                "bbox": tuple(int(x) for x in prop.bbox),
                "n_pixels_full": int(prop.area),
            }
        )
    items.sort(key=lambda d: (d["centroid_row"], d["centroid_col"]))

    rows: list[list[dict[str, Any]]] = []
    for item in items:
        if not rows or abs(item["centroid_row"] - rows[-1][0]["centroid_row"]) > row_tolerance_px:
            rows.append([item])
        else:
            rows[-1].append(item)
    for group in rows:
        group.sort(key=lambda d: d["centroid_col"])

    numbered = np.zeros(mask.shape, dtype=np.int32)
    ordered: list[dict[str, Any]] = []
    leaf_no = 1
    for group in rows:
        for item in group:
            item = dict(item)
            item["leaf_no"] = leaf_no
            numbered[labeled == item["label"]] = leaf_no
            ordered.append(item)
            leaf_no += 1
    return numbered, ordered


def erode_leaf_masks(numbered: np.ndarray, erosion_px: int) -> np.ndarray:
    """Erode each labelled leaf independently; keep labels."""
    if erosion_px <= 0:
        return numbered.copy()
    selem = disk(int(erosion_px))
    out = np.zeros_like(numbered)
    for leaf_no in np.unique(numbered):
        if leaf_no == 0:
            continue
        eroded = erosion(numbered == leaf_no, selem)
        if not np.any(eroded):
            # Fall back to the full mask if erosion would wipe the leaf.
            eroded = numbered == leaf_no
        out[eroded] = leaf_no
    return out


def flag_touching(numbered: np.ndarray, dilate_px: int) -> set[int]:
    """Return leaf numbers whose dilated masks overlap another leaf."""
    touching: set[int] = set()
    if dilate_px <= 0:
        return touching
    ids = [int(i) for i in np.unique(numbered) if i != 0]
    dilated = {}
    struct = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), dtype=bool)
    for i in ids:
        dilated[i] = ndi.binary_dilation(numbered == i, structure=struct)
    for a_i, a in enumerate(ids):
        for b in ids[a_i + 1 :]:
            if np.any(dilated[a] & dilated[b]):
                touching.add(a)
                touching.add(b)
    return touching


def maybe_split_sam2(
    rgb: np.ndarray,
    numbered: np.ndarray,
    cfg: Config,
    stem: str,
    expected: int | None,
) -> np.ndarray:
    """Optional SAM2 split of merged leaves. Off by default; never required."""
    if not bool(cfg.get("segmentation", "use_sam2", default=False)):
        return numbered
    n_found = int(numbered.max())
    if expected is None:
        _LOG.warning(
            "%s: use_sam2 is true but no expected_leaves count is available; skipping SAM2",
            stem,
        )
        return numbered
    if n_found == expected:
        return numbered
    try:
        return _run_sam2(rgb, numbered, cfg)
    except ImportError:
        _LOG.warning("use_sam2 is true but sam2 is not installed; continuing with NDVI masks")
        return numbered
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("SAM2 failed on %s (%s); continuing with NDVI masks", stem, exc)
        return numbered


def _run_sam2(rgb: np.ndarray, numbered: np.ndarray, cfg: Config) -> np.ndarray:
    """Point-prompt SAM2 at each NDVI component centroid. Imported only when called."""
    from sam2.build_sam import build_sam2  # type: ignore
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

    _LOG.info("SAM2 is enabled; using NDVI centroids as point prompts")
    # Checkpoint / model-id must come from config — we do not guess a weight path.
    ckpt = cfg.get("segmentation", "sam2_checkpoint", default=None)
    model_cfg = cfg.get("segmentation", "sam2_model_cfg", default=None)
    if not ckpt or not model_cfg:
        raise RuntimeError(
            "segmentation.sam2_checkpoint and segmentation.sam2_model_cfg must be "
            "set in config.yaml to run SAM2 (left unset because use_sam2 is false by default)."
        )
    sam = build_sam2(model_cfg, ckpt)
    predictor = SAM2ImagePredictor(sam)
    image = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    predictor.set_image(image)

    out = np.zeros(numbered.shape, dtype=np.int32)
    next_id = 1
    for leaf_no in np.unique(numbered):
        if leaf_no == 0:
            continue
        ys, xs = np.where(numbered == leaf_no)
        cy, cx = float(ys.mean()), float(xs.mean())
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[cx, cy]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        best = masks[int(np.argmax(scores))].astype(bool)
        # Restrict to the original component's bounding box to avoid swallowing neighbours.
        minr, maxr = int(ys.min()), int(ys.max()) + 1
        minc, maxc = int(xs.min()), int(xs.max()) + 1
        clip = np.zeros_like(best)
        clip[minr:maxr, minc:maxc] = True
        best = best & clip
        if not np.any(best):
            best = numbered == leaf_no
        labeled = label(best, connectivity=2)
        for prop in regionprops(labeled):
            out[labeled == prop.label] = next_id
            next_id += 1
    return out


def load_expected_leaves(cfg: Config) -> dict[str, int]:
    path = cfg.path("paths", "expected_leaves_file")
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "stem" not in df.columns or "expected_n" not in df.columns:
        raise ValueError(f"{path} must have columns stem, expected_n")
    return {str(r.stem): int(r.expected_n) for r in df.itertuples(index=False)}


def save_leaf_cache(
    cfg: Config,
    stem: str,
    leaf_no: int,
    cube: np.ndarray,
    wavelengths: np.ndarray,
    numbered_full: np.ndarray,
    numbered_eroded: np.ndarray,
    meta: dict[str, Any],
) -> Path:
    cache_dir = ensure_dir(cfg.path("paths", "cache_dir"))
    path = cache_dir / f"{stem}__leaf{leaf_no:02d}.npz"
    eroded = numbered_eroded == leaf_no
    full = numbered_full == leaf_no
    rows, cols = np.where(eroded)
    dtype_name = str(cfg.get("segmentation", "cache_dtype", default="float16"))
    dtype = np.dtype(dtype_name)
    max_pix = int(cfg.get("segmentation", "max_cache_pixels", default=150000))
    if rows.size > max_pix:
        _LOG.error(
            "Refusing to cache %s leaf %02d: %d pixels > max_cache_pixels=%d",
            stem, leaf_no, rows.size, max_pix,
        )
        raise RuntimeError(
            f"{stem} leaf {leaf_no} has {rows.size} pixels; segmentation likely merged the frame. "
            f"Check overlays and green_gate / crop_margins in config.yaml."
        )
    spectra = cube[rows, cols, :] if rows.size else np.zeros((0, cube.shape[2]), dtype=dtype)
    np.savez_compressed(
        path,
        spectra=np.asarray(spectra, dtype=dtype),
        rows=rows.astype(np.int32),
        cols=cols.astype(np.int32),
        wavelengths=wavelengths.astype(np.float64),
        bbox=np.array(meta["bbox"], dtype=np.int32),
        n_pixels=np.int32(rows.size),
        n_pixels_full=np.int32(int(full.sum())),
        centroid_row=np.float64(meta["centroid_row"]),
        centroid_col=np.float64(meta["centroid_col"]),
        stem=np.asarray(stem),
        leaf_no=np.int32(leaf_no),
    )
    return path


def save_overlay(
    cfg: Config,
    stem: str,
    rgb: np.ndarray,
    numbered: np.ndarray,
    leaves: list[dict[str, Any]],
    touching: set[int],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patheffects as pe

    out_dir = ensure_dir(cfg.path("paths", "overlays_dir"))
    path = out_dir / f"{stem}.png"
    dpi = int(cfg.get("segmentation", "overlay_dpi", default=120))
    fontsize = int(cfg.get("segmentation", "overlay_fontsize", default=18))
    h, w = rgb.shape[:2]
    fig_w = max(4.0, w / dpi)
    fig_h = max(4.0, h / dpi)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(rgb)
    for leaf in leaves:
        n = int(leaf["leaf_no"])
        color = "yellow" if n not in touching else "red"
        ax.contour(numbered == n, colors=[color], linewidths=1.2, levels=[0.5])
        ax.text(
            leaf["centroid_col"],
            leaf["centroid_row"],
            str(n),
            color="white",
            fontsize=fontsize,
            fontweight="bold",
            ha="center",
            va="center",
            path_effects=[pe.withStroke(linewidth=3, foreground="black")],
        )
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def segment_scan(
    cfg: Config,
    stem: str,
    bil_path: Path,
    hdr_path: Path,
    expected: int | None = None,
) -> tuple[list[dict[str, Any]], Path]:
    """Segment one scan. Returns per-leaf rows and the overlay path."""
    cube, wl = load_cube(bil_path, hdr_path, cfg=cfg, trim=True)
    ndvi = ndvi_map(cube, wl, cfg)
    mask, thresh = ndvi_leaf_mask(ndvi, cfg, cube=cube, wavelengths=wl)
    numbered, leaves = number_leaves(mask, int(cfg.require("segmentation", "row_tolerance_px")))
    rgb = false_rgb(cube, wl, cfg)
    numbered = maybe_split_sam2(rgb, numbered, cfg, stem, expected)
    if bool(cfg.get("segmentation", "use_sam2", default=False)) and expected is not None:
        # Re-number after a possible SAM2 split so numbering stays stable.
        numbered, leaves = number_leaves(numbered > 0, int(cfg.require("segmentation", "row_tolerance_px")))

    eroded = erode_leaf_masks(numbered, int(cfg.require("segmentation", "edge_erosion_px")))
    touching = flag_touching(numbered, int(cfg.get("segmentation", "touch_dilate_px", default=3)))

    parsed = parse_name(
        stem,
        camera_map={str(k): str(v) for k, v in (cfg.get("camera_map") or {}).items()},
        time_order_cfg=dict(cfg.get("time_order") or {}),
    )
    rows: list[dict[str, Any]] = []
    for leaf in leaves:
        n = int(leaf["leaf_no"])
        n_eroded = int((eroded == n).sum())
        save_leaf_cache(cfg, stem, n, cube, wl, numbered, eroded, leaf)
        minr, minc, maxr, maxc = leaf["bbox"]
        rec = {
            **parsed,
            "leaf_no": n,
            "n_pixels": n_eroded,
            "n_pixels_full": int(leaf["n_pixels_full"]),
            "centroid_row": float(leaf["centroid_row"]),
            "centroid_col": float(leaf["centroid_col"]),
            "bbox_min_row": int(minr),
            "bbox_min_col": int(minc),
            "bbox_max_row": int(maxr),
            "bbox_max_col": int(maxc),
            "touching": n in touching,
            "ndvi_threshold": float(thresh),
            "expected_n": expected,
            "count_mismatch": (expected is not None and len(leaves) != expected),
        }
        rows.append(rec)

    overlay = save_overlay(cfg, stem, rgb, numbered, leaves, touching)
    _LOG.info(
        "segmented %s: n_leaves=%d thresh=%.3f touching=%s overlay=%s",
        stem,
        len(rows),
        thresh,
        sorted(touching) or "-",
        overlay.name,
    )
    return rows, overlay


def run_segment(cfg: Config, force: bool = False) -> pd.DataFrame:
    set_seed(cfg)
    inv = run_inventory(cfg, force=False)
    in_scope = inv[inv["in_scope"] & inv["pair_ok"]].copy()
    leaves_path = cfg.path("paths", "leaves")
    ensure_dir(leaves_path.parent)
    expected_map = load_expected_leaves(cfg)

    if leaves_path.exists() and not force:
        _LOG.info("leaves.csv exists (%s); skip (use --force to rebuild)", leaves_path)
        return pd.read_csv(leaves_path)

    all_rows: list[dict[str, Any]] = []
    for rec in in_scope.itertuples(index=False):
        stem = str(rec.stem)
        expected = expected_map.get(stem)
        try:
            rows, _ = segment_scan(
                cfg,
                stem,
                Path(str(rec.bil_path)),
                Path(str(rec.hdr_path)),
                expected=expected,
            )
            all_rows.extend(rows)
        except Exception:
            _LOG.exception("Segmentation failed for %s", stem)
            raise

    df = pd.DataFrame(all_rows)
    df.to_csv(leaves_path, index=False)
    _LOG.info("Wrote %s (%d leaves)", leaves_path, len(df))
    return df


def print_leaf_counts(df: pd.DataFrame) -> None:
    print("\n=== Leaf count per scan ===")
    if df.empty:
        print("  (none)")
        return
    counts = (
        df.groupby(["stem", "time_point", "run"], dropna=False)
        .agg(n_leaves=("leaf_no", "size"), n_touching=("touching", "sum"))
        .reset_index()
    )
    print(counts.to_string(index=False))
    bad = df[df.get("count_mismatch", False) == True] if "count_mismatch" in df.columns else df.iloc[0:0]  # noqa: E712
    touching = df[df["touching"] == True] if "touching" in df.columns else df.iloc[0:0]  # noqa: E712
    print("\nFlagged (count mismatch vs expected_leaves file):")
    if bad.empty:
        print("  (none — expected_leaves.csv is optional and currently unused unless present)")
    else:
        print(bad[["stem", "leaf_no", "expected_n"]].drop_duplicates("stem").to_string(index=False))
    print("\nFlagged (touching neighbours):")
    if touching.empty:
        print("  (none)")
    else:
        print(touching.groupby("stem")["leaf_no"].apply(lambda s: list(s)).to_string())
    print()
