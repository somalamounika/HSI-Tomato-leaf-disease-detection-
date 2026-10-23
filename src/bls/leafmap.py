"""Load, validate, and merge the manual leaf_map.csv into leaves.csv."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from bls.config import Config

_LOG = logging.getLogger(__name__)

LEAF_MAP_COLUMNS = ["stem", "leaf_no", "experiment", "unit", "group", "include", "notes"]

UNCONFIRMED_WARNING = (
    "WARNING: leaf order within each scan is UNCONFIRMED. The leaf_map must be "
    "checked against the dataset's own metadata or confirmed with the dataset "
    "authors (Song Li, songli@vt.edu). Do not treat assumed unit/group labels "
    "as ground truth."
)


class LeafMapError(ValueError):
    """Invalid leaf map or missing groups that block training/analysis."""


def experiment_unit_table(cfg: Config) -> dict[int, dict[str, Any]]:
    raw = cfg.require("experiments")
    table: dict[int, dict[str, Any]] = {}
    for key, spec in raw.items():
        exp = int(key)
        control = {int(x) for x in spec["control"]}
        inoculated = {int(x) for x in spec["inoculated"]}
        table[exp] = {
            "n_units": int(spec["n_units"]),
            "control": control,
            "inoculated": inoculated,
            "unit_to_group": {u: "control" for u in control} | {u: "inoculated" for u in inoculated},
        }
    return table


def run_to_experiment(cfg: Config, run: int | None) -> int | None:
    mapping = cfg.require("run_to_experiment")
    if run is None or (isinstance(run, float) and pd.isna(run)):
        return None
    key = int(run)
    if key not in mapping and str(key) not in mapping:
        _LOG.warning("run %s is not in run_to_experiment; experiment left blank", run)
        return None
    return int(mapping[key] if key in mapping else mapping[str(key)])


def group_for_unit(cfg: Config, experiment: int, unit: int) -> str:
    table = experiment_unit_table(cfg)
    if experiment not in table:
        raise LeafMapError(f"Unknown experiment {experiment}; not in config experiments table")
    mapping = table[experiment]["unit_to_group"]
    if unit not in mapping:
        raise LeafMapError(
            f"Unit {unit} is not in experiment {experiment}'s unit table "
            f"(control={sorted(table[experiment]['control'])}, "
            f"inoculated={sorted(table[experiment]['inoculated'])})"
        )
    return str(mapping[unit])


def empty_leaf_map() -> pd.DataFrame:
    return pd.DataFrame(columns=LEAF_MAP_COLUMNS)


def make_template(
    cfg: Config,
    leaves: pd.DataFrame,
    assume_order: bool = False,
    path: Path | None = None,
) -> pd.DataFrame:
    """Write data/leaf_map.csv. Never overwrite filled rows; only add new ones.

    --assume-order fills unit = leaf_no and group from the unit table only when
    the leaf count equals the number of units for that experiment. Every such
    row is tagged notes='assumed order, unconfirmed'.
    """
    path = path or cfg.path("paths", "leaf_map")
    existing = pd.read_csv(path) if path.exists() else empty_leaf_map()
    for col in LEAF_MAP_COLUMNS:
        if col not in existing.columns:
            existing[col] = pd.NA

    existing["_key"] = existing["stem"].astype(str) + "::" + existing["leaf_no"].astype(int).astype(str)
    existing_keys = set(existing["_key"])

    table = experiment_unit_table(cfg)
    new_rows: list[dict[str, Any]] = []
    assumed_stems: list[str] = []

    leaves = leaves.copy()
    leaves["leaf_no"] = leaves["leaf_no"].astype(int)
    counts = leaves.groupby("stem")["leaf_no"].size().to_dict()

    for rec in leaves.itertuples(index=False):
        key = f"{rec.stem}::{int(rec.leaf_no)}"
        if key in existing_keys:
            continue
        run = None if pd.isna(getattr(rec, "run", None)) else int(rec.run)
        experiment = run_to_experiment(cfg, run)
        unit: int | None = None
        group: str | None = None
        notes = ""
        if assume_order and experiment is not None and experiment in table:
            n_units = table[experiment]["n_units"]
            if counts.get(str(rec.stem), 0) == n_units:
                unit = int(rec.leaf_no)
                if unit in table[experiment]["unit_to_group"]:
                    group = table[experiment]["unit_to_group"][unit]
                    notes = "assumed order, unconfirmed"
                    assumed_stems.append(str(rec.stem))
        new_rows.append(
            {
                "stem": rec.stem,
                "leaf_no": int(rec.leaf_no),
                "experiment": experiment,
                "unit": unit,
                "group": group,
                "include": 1,
                "notes": notes,
            }
        )

    added = pd.DataFrame(new_rows, columns=LEAF_MAP_COLUMNS)
    out = pd.concat([existing.drop(columns=["_key"], errors="ignore"), added], ignore_index=True)
    out = out[LEAF_MAP_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    _LOG.info("Wrote %s (%d rows, %d new)", path, len(out), len(added))

    if assume_order and assumed_stems:
        unique = sorted(set(assumed_stems))
        print("\n" + "!" * 72)
        print("LOUD WARNING: --assume-order filled unit = leaf_no and group from the")
        print("experiment unit table for scans whose leaf count equals n_units:")
        for s in unique:
            print(f"  {s}")
        print("This mapping is UNCONFIRMED. Check it against dataset metadata or the authors.")
        print("!" * 72 + "\n")
    return out


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def validate_leaf_map(cfg: Config, leaf_map: pd.DataFrame) -> None:
    """Raise LeafMapError on inconsistent unit/group vs the experiment table."""
    allowed = {str(g).lower() for g in cfg.require("allowed_groups")}
    table = experiment_unit_table(cfg)
    errors: list[str] = []
    for rec in leaf_map.itertuples(index=False):
        loc = f"{rec.stem} leaf {rec.leaf_no}"
        if _is_blank(rec.group) and _is_blank(rec.unit):
            continue
        group = None if _is_blank(rec.group) else str(rec.group).strip().lower()
        if group is not None and group not in allowed:
            errors.append(f"{loc}: group={rec.group!r} not in {sorted(allowed)}")
            continue
        if _is_blank(rec.experiment) or _is_blank(rec.unit):
            if group is not None:
                # group filled without unit is allowed only if we cannot check the table
                continue
            continue
        experiment = int(rec.experiment)
        unit = int(rec.unit)
        if experiment not in table:
            errors.append(f"{loc}: experiment {experiment} is not in config")
            continue
        expected_group = table[experiment]["unit_to_group"].get(unit)
        if expected_group is None:
            errors.append(
                f"{loc}: unit {unit} is not in experiment {experiment}'s unit table"
            )
            continue
        if group is not None and group != expected_group:
            errors.append(
                f"{loc}: group={group} does not match unit table ({expected_group}) "
                f"for experiment {experiment} unit {unit}"
            )
    if errors:
        raise LeafMapError("leaf_map.csv failed validation:\n  " + "\n  ".join(errors))


def infected_flag(group: Any, time_point: Any) -> int:
    """infected = 1 iff group is inoculated AND time_point is not bi.

    Control leaves at every time point, and every leaf at bi, are 0.
    """
    if _is_blank(group) or _is_blank(time_point):
        return 0
    if str(time_point).strip().lower() == "bi":
        return 0
    return int(str(group).strip().lower() == "inoculated")


def merge_leaf_map(cfg: Config, leaves: pd.DataFrame, leaf_map: pd.DataFrame | None = None) -> pd.DataFrame:
    """Left-join leaf_map onto leaves and add experiment, unit, group, include, infected."""
    if leaf_map is None:
        path = cfg.path("paths", "leaf_map")
        if not path.exists():
            _LOG.warning("No leaf_map.csv at %s; groups will be blank", path)
            leaf_map = empty_leaf_map()
        else:
            leaf_map = pd.read_csv(path)
    validate_leaf_map(cfg, leaf_map)

    left = leaves.copy()
    left["leaf_no"] = left["leaf_no"].astype(int)
    right = leaf_map.copy()
    if right.empty:
        for col in ("experiment", "unit", "group", "include", "notes"):
            if col not in left.columns:
                left[col] = pd.NA
        left["include"] = left["include"].fillna(1).astype(int)
        left["infected"] = 0
        return left

    right["leaf_no"] = right["leaf_no"].astype(int)
    keep = ["stem", "leaf_no", "experiment", "unit", "group", "include", "notes"]
    right = right[[c for c in keep if c in right.columns]]
    merged = left.merge(right, on=["stem", "leaf_no"], how="left", suffixes=("", "_map"))
    # Prefer map columns when both exist.
    for col in ("experiment", "unit", "group", "include", "notes"):
        map_col = f"{col}_map"
        if map_col in merged.columns:
            merged[col] = merged[map_col].combine_first(merged[col]) if col in merged.columns else merged[map_col]
            merged = merged.drop(columns=[map_col])
    merged["include"] = merged["include"].fillna(1).astype(int)
    merged["infected"] = [
        infected_flag(g, tp) for g, tp in zip(merged.get("group", []), merged.get("time_point", []))
    ]
    return merged


def assert_ready_for_training(cfg: Config, leaves: pd.DataFrame) -> None:
    """Stop with a clear message if included leaves lack group, or a training scan is one-group."""
    if not bool(cfg.get("training", "require_group_on_included", default=True)):
        return
    work = leaves[leaves.get("include", 1) == 1].copy()
    missing = work[work["group"].apply(_is_blank)]
    if not missing.empty:
        stems = ", ".join(sorted(missing["stem"].astype(str).unique()))
        raise LeafMapError(
            f"Training/analysis stopped: {len(missing)} included leaf/leaves have no group. "
            f"Fill unit and group in data/leaf_map.csv (stems: {stems}). "
            + UNCONFIRMED_WARNING
        )

    if not bool(cfg.get("training", "require_both_groups_per_scan", default=True)):
        return
    train_tp = set(cfg.require("training", "time_points"))
    train = work[work["time_point"].isin(train_tp)]
    for stem, part in train.groupby("stem"):
        groups = sorted({str(g).lower() for g in part["group"].dropna().unique()})
        if len(groups) == 1:
            raise LeafMapError(
                f"Scan {stem} contains only {groups[0]} leaves; "
                f"time point and group would be confounded."
            )
        if len(groups) == 0:
            raise LeafMapError(
                f"Scan {stem} has no group labels; time point and group would be confounded."
            )


def print_group_table(leaves: pd.DataFrame) -> None:
    print("\n=== Leaves per time_point × group ===")
    work = leaves.copy()
    work["group"] = work["group"].fillna("(unlabelled)").replace("", "(unlabelled)")
    tab = (
        work.groupby(["time_point", "group"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
        .pivot(index="time_point", columns="group", values="n")
        .fillna(0)
        .astype(int)
    )
    print(tab.to_string())
    print("\n" + UNCONFIRMED_WARNING + "\n")
