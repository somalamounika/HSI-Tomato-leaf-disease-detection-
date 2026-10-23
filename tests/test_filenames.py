"""Parser tests for every stem listed in the project brief (section 2)."""

from __future__ import annotations

import pytest

from bls.filenames import FilenameError, parse_name

# Section 2 stems — these must parse exactly as specified.
CASES = [
    {
        "stem": "0dai_bi_visnir_bc_run2_1",
        "day": 0,
        "time_point": "bi",
        "time_order": 0,
        "camera": "vnir",
        "background": "bc",
        "run": 2,
        "scan": 1,
    },
    {
        "stem": "0dai_2hr_visnir_bc_run2-1",
        "day": 0,
        "time_point": "2hr",
        "time_order": 1,
        "camera": "vnir",
        "background": "bc",
        "run": 2,
        "scan": 1,
    },
    {
        "stem": "1dai_visnir_bc_run1",
        "day": 1,
        "time_point": "d1",
        "time_order": 2,
        "camera": "vnir",
        "background": "bc",
        "run": 1,
        "scan": None,
    },
    {
        "stem": "2dai_visnir_bc_run1",
        "day": 2,
        "time_point": "d2",
        "time_order": 3,
        "camera": "vnir",
        "background": "bc",
        "run": 1,
        "scan": None,
    },
    {
        "stem": "3dai_visnir_bc_run1",
        "day": 3,
        "time_point": "d3",
        "time_order": 4,
        "camera": "vnir",
        "background": "bc",
        "run": 1,
        "scan": None,
    },
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["stem"])
def test_parse_section2_stems(case: dict) -> None:
    got = parse_name(case["stem"])
    assert got["stem"] == case["stem"]
    assert got["day"] == case["day"]
    assert got["time_point"] == case["time_point"]
    assert got["time_order"] == case["time_order"]
    assert got["camera"] == case["camera"]
    assert got["background"] == case["background"]
    assert got["run"] == case["run"]
    assert got["scan"] == case["scan"]


def test_run2_underscore_and_hyphen_are_equivalent() -> None:
    a = parse_name("0dai_bi_visnir_bc_run2_1")
    b = parse_name("0dai_2hr_visnir_bc_run2-1")
    assert a["run"] == b["run"] == 2
    assert a["scan"] == b["scan"] == 1


def test_swir_is_parsed_but_not_vnir() -> None:
    got = parse_name("3dai_swir_bc_run1")
    assert got["camera"] == "swir"
    assert got["time_point"] == "d3"


def test_later_day_has_higher_time_order() -> None:
    got = parse_name("7dai_visnir_bc_run1")
    assert got["time_point"] == "d7"
    assert got["time_order"] == 8  # N + 1


def test_rejects_unknown_stem() -> None:
    with pytest.raises(FilenameError, match="does not match"):
        parse_name("not_a_scan")


def test_rejects_extra_token_before_camera() -> None:
    with pytest.raises(FilenameError, match="does not match"):
        parse_name("7dai_1_visnir_bc_run1")
