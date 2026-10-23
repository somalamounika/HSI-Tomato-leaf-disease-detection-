"""Add 6dai, train the 3-class stage model, write outputs/stage3 and models/stage3_model.joblib."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bls.config import load_config, setup_logging  # noqa: E402
from bls.stage3 import CLASS_DISPLAY, run_stage3  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)
    result = run_stage3(cfg, force=args.force)
    table = result["table"]
    bundle = result["bundle"]
    print(f"Model: {bundle['name']}")
    print(f"Leaves: {len(table)}")
    print(table.groupby("stage").size().rename("n").to_string())
    print("Display names:", CLASS_DISPLAY)


if __name__ == "__main__":
    main()
