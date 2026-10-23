"""Compute per-leaf features from eroded-pixel caches → outputs/features.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bls.config import load_config, setup_logging  # noqa: E402
from bls.features import print_feature_summary, run_features  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)
    df = run_features(cfg, force=args.force)
    print_feature_summary(df)


if __name__ == "__main__":
    main()
