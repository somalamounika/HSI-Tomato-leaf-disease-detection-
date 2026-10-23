"""Load config.yaml and resolve paths relative to the project root."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_LOG = logging.getLogger(__name__)


class Config:
    """Nested-dict config with path resolution against the project root."""

    def __init__(self, data: dict[str, Any], root: Path, source: Path) -> None:
        self.data = data
        self.root = root
        self.source = source

    def get(self, *keys: str, default: Any = None) -> Any:
        cur: Any = self.data
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    def require(self, *keys: str) -> Any:
        cur: Any = self.data
        trail: list[str] = []
        for key in keys:
            trail.append(key)
            if not isinstance(cur, dict) or key not in cur:
                raise KeyError(f"Missing config key: {'.'.join(trail)}")
            cur = cur[key]
        return cur

    def path(self, *keys: str) -> Path:
        value = Path(str(self.require(*keys)))
        if not value.is_absolute():
            value = self.root / value
        return value

    def seed(self) -> int:
        return int(self.require("random_seed"))


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from *start* (or this file) until config.yaml is found."""
    candidates: list[Path] = []
    if start is not None:
        candidates.append(start if start.is_dir() else start.parent)
    candidates.append(Path(__file__).resolve().parent)
    candidates.append(Path.cwd())
    seen: set[Path] = set()
    for base in candidates:
        for p in [base, *base.parents]:
            if p in seen:
                continue
            seen.add(p)
            if (p / "config.yaml").exists():
                return p
    raise FileNotFoundError("Could not find config.yaml from this file or the cwd.")


def load_config(path: Path | str | None = None) -> Config:
    """Load YAML config. *path* may be a file or a directory containing config.yaml."""
    if path is None:
        root = find_project_root()
        cfg_path = root / "config.yaml"
    else:
        cfg_path = Path(path)
        if cfg_path.is_dir():
            root = cfg_path
            cfg_path = cfg_path / "config.yaml"
        else:
            root = cfg_path.parent
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {cfg_path} must be a mapping")
    return Config(data, root=root, source=cfg_path)


def setup_logging(cfg: Config | None = None, level: str | None = None) -> None:
    """Configure the root logger once."""
    if level is None:
        level = (cfg.get("log_level") if cfg is not None else None) or "INFO"
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=False,
    )


def set_seed(cfg: Config) -> int:
    """Seed Python, NumPy (and callers pass the same value to sklearn/LightGBM)."""
    seed = cfg.seed()
    random.seed(seed)
    np.random.seed(seed)
    return seed


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
