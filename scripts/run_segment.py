"""Segment in-scope scans, write overlays, leaf caches, and leaves.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bls.config import load_config, setup_logging  # noqa: E402
from bls.segment import print_leaf_counts, run_segment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)
    df = run_segment(cfg, force=args.force)
    print_leaf_counts(df)
    overlays = cfg.path("paths", "overlays_dir")
    print(f"Overlays: {overlays}")
    for p in sorted(overlays.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
