"""Invariant tests for kino.galaxy.rotation_curve (README.md Phase 1)."""

import numpy as np
import pytest

from kino.galaxy.rotation_curve import ROTATION_CURVES, arctan, flat_turnover, tanh

MODELS = [arctan, tanh, flat_turnover]


@pytest.mark.parametrize("model", MODELS)
def test_zero_at_center(model):
    assert model(0.0, v_flat=180.0, r_t=3.0) == pytest.approx(0.0)


@pytest.mark.parametrize("model", MODELS)
def test_monotonically_non_decreasing(model):
    r = np.linspace(0, 50, 500)
    v = model(r, v_flat=180.0, r_t=3.0)
    assert np.all(np.diff(v) >= -1e-12)


@pytest.mark.parametrize("model", [arctan, tanh])
def test_asymptotes_to_v_flat(model):
    v = model(1e5, v_flat=180.0, r_t=3.0)
    assert v == pytest.approx(180.0, rel=1e-3)


def test_flat_turnover_reaches_v_flat_exactly_at_r_t():
    assert flat_turnover(3.0, v_flat=180.0, r_t=3.0) == pytest.approx(180.0)
    assert flat_turnover(10.0, v_flat=180.0, r_t=3.0) == pytest.approx(180.0)


def test_registry_matches_functions():
    expected = {"arctan": arctan, "tanh": tanh, "flat_turnover": flat_turnover}
    assert expected == ROTATION_CURVES
