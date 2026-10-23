"""Walk data/raw, pair .bil/.bil.hdr, write outputs/inventory.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bls.config import load_config, setup_logging  # noqa: E402
from bls.inventory import print_summary, run_inventory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--force", action="store_true", help="rebuild even if inventory.csv exists")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)
    df = run_inventory(cfg, force=args.force)
    print_summary(df)


if __name__ == "__main__":
    main()
