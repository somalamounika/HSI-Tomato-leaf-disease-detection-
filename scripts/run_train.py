"""Train LightGBM + logistic + LDA; write reports and the selected model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bls.config import load_config, setup_logging  # noqa: E402
from bls.leafmap import LeafMapError  # noqa: E402
from bls.train_leaf import run_train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)
    try:
        run_train(cfg, force=args.force)
    except LeafMapError as exc:
        raise SystemExit(str(exc)) from exc
    report = cfg.path("paths", "reports_dir") / "leaf_model.md"
    if report.exists():
        print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
