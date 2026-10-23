"""Walk the data folder, pair .bil / .bil.hdr files, write inventory.csv."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from bls.config import Config, ensure_dir
from bls.filenames import FilenameError, parse_name
from bls.io import parse_envi_header

_LOG = logging.getLogger(__name__)


def _camera_map(cfg: Config) -> dict[str, str]:
    raw = cfg.get("camera_map", default=None) or {}
    return {str(k).lower(): str(v) for k, v in raw.items()}


def _time_order_cfg(cfg: Config) -> dict[str, Any]:
    return dict(cfg.get("time_order", default=None) or {})


def iter_scan_stems(data_dir: Path) -> list[dict[str, Any]]:
    """Find .bil files and their .bil.hdr partners (recursive)."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder does not exist: {data_dir}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for bil in sorted(data_dir.rglob("*.bil")):
        stem = bil.name[: -len(".bil")]
        if stem in seen:
            continue
        seen.add(stem)
        hdr = bil.with_name(bil.name + ".hdr")  # foo.bil → foo.bil.hdr
        rows.append(
            {
                "stem": stem,
                "bil_path": str(bil),
                "hdr_path": str(hdr),
                "bil_exists": bil.exists(),
                "hdr_exists": hdr.exists(),
                "pair_ok": bil.exists() and hdr.exists(),
            }
        )
    # Orphan headers with no cube.
    for hdr in sorted(data_dir.rglob("*.bil.hdr")):
        stem = hdr.name[: -len(".bil.hdr")]
        if stem in seen:
            continue
        seen.add(stem)
        bil = hdr.with_name(stem + ".bil")
        rows.append(
            {
                "stem": stem,
                "bil_path": str(bil),
                "hdr_path": str(hdr),
                "bil_exists": bil.exists(),
                "hdr_exists": hdr.exists(),
                "pair_ok": bil.exists() and hdr.exists(),
            }
        )
    return rows


def classify_scan(parsed: dict[str, Any] | None, cfg: Config, extra_flags: list[str]) -> tuple[bool, str, str]:
    """Return (in_scope, skip_reason, flags)."""
    flags = list(extra_flags)
    if parsed is None:
        return False, "unparseable_name", ";".join(flags) if flags else "unparseable_name"

    camera_ok = parsed["camera"] == cfg.require("filters", "camera")
    bg_ok = parsed["background"] == cfg.require("filters", "background")
    allowed_tp = set(cfg.require("filters", "allowed_time_points"))
    max_order = int(cfg.require("filters", "max_time_order"))
    tp_ok = parsed["time_point"] in allowed_tp and int(parsed["time_order"]) <= max_order

    reasons: list[str] = []
    if not camera_ok:
        reasons.append(f"camera={parsed['camera']}")
    if not bg_ok:
        reasons.append(f"background={parsed['background']}")
    if not tp_ok:
        reasons.append(f"time_point={parsed['time_point']}")

    in_scope = camera_ok and bg_ok and tp_ok
    skip_reason = "" if in_scope else "out_of_scope:" + ",".join(reasons)
    return in_scope, skip_reason, ";".join(flags)


def build_inventory(cfg: Config) -> pd.DataFrame:
    data_dir = cfg.path("paths", "data_dir")
    cam_map = _camera_map(cfg)
    to_cfg = _time_order_cfg(cfg)
    expected_bands = int(cfg.require("filters", "expected_vnir_bands"))
    vnir_name = cfg.require("filters", "camera")

    records: list[dict[str, Any]] = []
    for row in iter_scan_stems(data_dir):
        flags: list[str] = []
        parsed: dict[str, Any] | None = None
        parse_error = ""
        try:
            parsed = parse_name(row["stem"], camera_map=cam_map, time_order_cfg=to_cfg)
        except FilenameError as exc:
            parse_error = str(exc)
            flags.append("unparseable_name")

        header: dict[str, Any] = {}
        if row["hdr_exists"]:
            try:
                header = parse_envi_header(Path(row["hdr_path"]))
            except Exception as exc:  # noqa: BLE001
                flags.append("header_read_error")
                parse_error = (parse_error + f" | header: {exc}").strip(" |")
        else:
            flags.append("missing_hdr")
        if not row["bil_exists"]:
            flags.append("missing_bil")
        if not row["pair_ok"]:
            flags.append("missing_pair")

        bands = header.get("bands")
        camera = parsed["camera"] if parsed else None
        if camera == vnir_name and bands is not None and int(bands) != expected_bands:
            flags.append(f"bands_ne_{expected_bands}")

        in_scope, skip_reason, flag_str = classify_scan(parsed, cfg, flags)

        rec: dict[str, Any] = {
            "stem": row["stem"],
            "bil_path": row["bil_path"],
            "hdr_path": row["hdr_path"],
            "pair_ok": row["pair_ok"],
            "in_scope": in_scope,
            "skip_reason": skip_reason,
            "flags": flag_str,
            "parse_error": parse_error,
            "day": parsed["day"] if parsed else None,
            "time_point": parsed["time_point"] if parsed else None,
            "time_order": parsed["time_order"] if parsed else None,
            "camera": parsed["camera"] if parsed else None,
            "camera_token": parsed["camera_token"] if parsed else None,
            "background": parsed["background"] if parsed else None,
            "run": parsed["run"] if parsed else None,
            "scan": parsed["scan"] if parsed else None,
            "lines": header.get("lines"),
            "samples": header.get("samples"),
            "bands": bands,
            "wavelength_min": header.get("wavelength_min"),
            "wavelength_max": header.get("wavelength_max"),
            "data_type": header.get("data_type"),
            "reflectance_scale_factor": header.get("reflectance_scale_factor"),
            "envi_label": header.get("label"),
        }
        records.append(rec)
        _LOG.info(
            "scan %s in_scope=%s skip=%s flags=%s",
            row["stem"],
            in_scope,
            skip_reason or "-",
            flag_str or "-",
        )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "stem",
                "bil_path",
                "hdr_path",
                "pair_ok",
                "in_scope",
                "skip_reason",
                "flags",
                "time_point",
                "run",
            ]
        )
    else:
        df = df.sort_values(["time_order", "run", "scan", "stem"], na_position="last").reset_index(drop=True)
    return df


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Number of scans per time point × run, and which pass the filters."""
    if df.empty:
        return pd.DataFrame(columns=["time_point", "run", "n_total", "n_in_scope", "n_skipped"])
    work = df.copy()
    work["time_point"] = work["time_point"].fillna("unparsed")
    work["run"] = work["run"].fillna(-1)
    grouped = (
        work.groupby(["time_point", "run"], dropna=False)
        .agg(
            n_total=("stem", "size"),
            n_in_scope=("in_scope", "sum"),
        )
        .reset_index()
    )
    grouped["n_in_scope"] = grouped["n_in_scope"].astype(int)
    grouped["n_skipped"] = grouped["n_total"] - grouped["n_in_scope"]
    return grouped


def run_inventory(cfg: Config, force: bool = False) -> pd.DataFrame:
    out_path = cfg.path("paths", "inventory")
    ensure_dir(out_path.parent)
    if out_path.exists() and not force:
        _LOG.info("Inventory exists (%s); skip (use --force to rebuild)", out_path)
        return pd.read_csv(out_path)

    df = build_inventory(cfg)
    df.to_csv(out_path, index=False)
    _LOG.info("Wrote %s (%d scans)", out_path, len(df))
    return df


def print_summary(df: pd.DataFrame) -> None:
    summary = summary_table(df)
    print("\n=== Inventory summary: scans per time_point × run ===")
    print(summary.to_string(index=False))
    in_scope = df[df["in_scope"]] if "in_scope" in df.columns else df.iloc[0:0]
    skipped = df[~df["in_scope"]] if "in_scope" in df.columns else df
    print(f"\nIn-scope scans ({len(in_scope)}):")
    if in_scope.empty:
        print("  (none)")
    else:
        cols = [c for c in ["stem", "time_point", "run", "scan", "lines", "samples", "bands", "flags"] if c in in_scope.columns]
        print(in_scope[cols].to_string(index=False))
    print(f"\nSkipped scans ({len(skipped)}):")
    if skipped.empty:
        print("  (none)")
    else:
        cols = [c for c in ["stem", "skip_reason", "flags"] if c in skipped.columns]
        print(skipped[cols].to_string(index=False))
    print()
