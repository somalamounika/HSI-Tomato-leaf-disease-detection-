"""Parse tomato-leaf hyperspectral scan stems into structured fields."""

from __future__ import annotations

import re
from typing import Any

# <day>dai[_bi|_2hr]_<camera>_<background>_run<N>[_|-]<scan>
# Separators after run are inconsistent: run2_1 and run2-1 are both used.
STEM_RE = re.compile(
    r"^(?P<day>\d+)dai"
    r"(?:_(?P<tp>bi|2hr))?"
    r"_(?P<camera>[A-Za-z0-9]+)"
    r"_(?P<background>[A-Za-z0-9]+)"
    r"_run(?P<run>\d+)"
    r"(?:[_-](?P<scan>\d+))?$",
    re.IGNORECASE,
)

DEFAULT_CAMERA_MAP = {"visnir": "vnir", "swir": "swir"}
DEFAULT_TIME_ORDER = {"bi": 0, "2hr": 1, "day_offset": 1}


class FilenameError(ValueError):
    """Raised when a scan stem does not match the expected pattern."""


def time_order_of(time_point: str, time_order_cfg: dict[str, Any] | None = None) -> int:
    """bi=0, 2hr=1, dN = N + day_offset (default 1 so d1=2, d2=3, d3=4)."""
    cfg = time_order_cfg or DEFAULT_TIME_ORDER
    if time_point in cfg and time_point not in {"day_offset"}:
        return int(cfg[time_point])
    if time_point.startswith("d") and time_point[1:].isdigit():
        return int(time_point[1:]) + int(cfg.get("day_offset", 1))
    raise FilenameError(f"Cannot assign time_order for time_point={time_point!r}")


def parse_name(
    stem: str,
    camera_map: dict[str, str] | None = None,
    time_order_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse a scan stem into day, time_point, time_order, camera, background, run, scan.

    Raises FilenameError with a clear message when the stem does not match.
    """
    raw = stem.strip()
    match = STEM_RE.fullmatch(raw)
    if match is None:
        raise FilenameError(
            f"Filename stem {raw!r} does not match "
            f"'<day>dai[_bi|_2hr]_<camera>_<background>_run<N>[_|-]<scan>'. "
            f"Examples: '0dai_bi_visnir_bc_run2_1', '0dai_2hr_visnir_bc_run2-1', "
            f"'1dai_visnir_bc_run1'."
        )

    day = int(match.group("day"))
    tp_token = match.group("tp")
    time_point = tp_token.lower() if tp_token else f"d{day}"

    camera_raw = match.group("camera").lower()
    mapping = camera_map or DEFAULT_CAMERA_MAP
    camera = mapping.get(camera_raw, camera_raw)

    background = match.group("background")  # keep the raw token (e.g. 'bc')
    run = int(match.group("run"))
    scan_g = match.group("scan")
    scan = int(scan_g) if scan_g is not None else None

    time_order = time_order_of(time_point, time_order_cfg)

    return {
        "stem": raw,
        "day": day,
        "time_point": time_point,
        "time_order": time_order,
        "camera": camera,
        "camera_token": camera_raw,
        "background": background,
        "run": run,
        "scan": scan,
    }
