"""Early-change analysis for days 0–3 (not a disease-area map)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bls.config import load_config, setup_logging  # noqa: E402
from bls.spread import run_spread  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)
    paths = run_spread(cfg, force=args.force)
    report = Path(paths["report"])
    if report.exists():
        print(report.read_text(encoding="utf-8"))
    fig_dir = cfg.path("paths", "figures_dir")
    print(f"Figures: {fig_dir}")
    for p in sorted(fig_dir.glob("spread_*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
