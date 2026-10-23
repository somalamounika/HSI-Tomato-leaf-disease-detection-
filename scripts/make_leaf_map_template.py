"""Write data/leaf_map.csv from leaves.csv. Never overwrite filled rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bls.config import load_config, setup_logging  # noqa: E402
from bls.leafmap import UNCONFIRMED_WARNING, make_template, merge_leaf_map, print_group_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument(
        "--assume-order",
        action="store_true",
        help="Fill unit=leaf_no and group from the unit table when n_leaves == n_units. UNCONFIRMED.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)
    leaves_path = cfg.path("paths", "leaves")
    if not leaves_path.exists():
        raise SystemExit(f"Run scripts/run_segment.py first; missing {leaves_path}")
    leaves = pd.read_csv(leaves_path)
    make_template(cfg, leaves, assume_order=args.assume_order)
    merged = merge_leaf_map(cfg, leaves)
    print_group_table(merged)
    print(UNCONFIRMED_WARNING)


if __name__ == "__main__":
    main()
