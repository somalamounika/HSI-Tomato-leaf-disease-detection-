"""Exploratory spectra, indices, PCA, and Mann–Whitney tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bls.config import load_config, setup_logging  # noqa: E402
from bls.explore import run_explore  # noqa: E402
from bls.leafmap import UNCONFIRMED_WARNING, LeafMapError  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)
    try:
        paths = run_explore(cfg, force=args.force)
    except LeafMapError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Figures: {paths['figures']}")
    for p in sorted(Path(paths["figures"]).glob("*.png")):
        print(f"  {p.name}")
    print(f"Index tests: {paths['tests']}")
    print(UNCONFIRMED_WARNING)


if __name__ == "__main__":
    main()
