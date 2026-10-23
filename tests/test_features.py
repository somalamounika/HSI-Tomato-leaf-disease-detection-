"""Index values on a synthetic vegetation-like spectrum with known band values."""

from __future__ import annotations

import numpy as np
import pytest

from bls.config import load_config
from bls.features import features_from_pixels, safe_div, vegetation_indices


def _cfg():
    return load_config()


def _spectrum() -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear vegetation spectrum with exact values at index wavelengths."""
    known = {
        445.0: 0.040,
        531.0: 0.090,
        550.0: 0.100,
        570.0: 0.080,
        670.0: 0.050,
        680.0: 0.048,
        695.0: 0.080,
        700.0: 0.120,
        705.0: 0.180,
        720.0: 0.300,
        740.0: 0.420,
        750.0: 0.450,
        760.0: 0.470,
        780.0: 0.490,
        790.0: 0.495,
        800.0: 0.500,
        900.0: 0.480,
        970.0: 0.430,
    }
    wl = np.arange(400.0, 1000.0 + 2.0, 2.0)
    spec = np.interp(wl, list(known.keys()), list(known.values()))
    return spec.astype(np.float64), wl.astype(np.float64), known


def test_safe_div_guards_zero() -> None:
    assert safe_div(1.0, 0.0) == 0.0
    assert np.isfinite(safe_div(np.array([1.0, 1.0]), np.array([0.0, 2.0]))).all()


def test_known_index_values() -> None:
    cfg = _cfg()
    spec, wl, _k = _spectrum()
    idx = vegetation_indices(spec, wl, cfg)
    from bls.features import sample_at

    hw = int(cfg.get("band", "half_width", default=2))

    def R(nm: float) -> float:
        return float(sample_at(spec, wl, nm, hw))

    r800, r670 = R(800), R(670)
    assert idx["NDVI"] == pytest.approx((r800 - r670) / (r800 + r670), rel=1e-6)
    r550 = R(550)
    assert idx["GNDVI"] == pytest.approx((r800 - r550) / (r800 + r550), rel=1e-6)
    r790, r720 = R(790), R(720)
    assert idx["NDRE"] == pytest.approx((r790 - r720) / (r790 + r720), rel=1e-6)
    r531, r570 = R(531), R(570)
    assert idx["PRI"] == pytest.approx((r531 - r570) / (r531 + r570), rel=1e-6)
    r680 = R(680)
    assert idx["PSSRa"] == pytest.approx(r800 / r680, rel=1e-6)
    r695, r760 = R(695), R(760)
    assert idx["CTR2"] == pytest.approx(r695 / r760, rel=1e-6)
    r445 = R(445)
    assert idx["SIPI"] == pytest.approx((r800 - r445) / (r800 - r680), rel=1e-6)
    r750, r705 = R(750), R(705)
    assert idx["mND705"] == pytest.approx((r750 - r705) / (r750 + r705 - 2 * r445), rel=1e-6)
    assert idx["mSR705"] == pytest.approx((r750 - r445) / (r705 - r445), rel=1e-6)
    r700 = R(700)
    assert idx["ARI"] == pytest.approx(1 / r550 - 1 / r700, rel=1e-6)
    r780, r740 = R(780), R(740)
    expected_rep = 700 + 40 * ((r670 + r780) / 2 - r700) / (r740 - r700)
    assert idx["REP"] == pytest.approx(expected_rep, rel=1e-6)
    r900, r970 = R(900), R(970)
    assert idx["WI"] == pytest.approx(r900 / r970, rel=1e-6)
    assert 0.6 < float(idx["NDVI"]) < 0.9


def test_features_from_pixels_no_nan() -> None:
    cfg = _cfg()
    spec, wl, _ = _spectrum()
    pix = np.stack([spec, spec * 1.01, spec * 0.99], axis=0)
    feats = features_from_pixels(pix, wl, cfg)
    assert feats["NDVI"] > 0.6
    for key, val in feats.items():
        assert np.isfinite(val), f"{key} is not finite: {val}"
