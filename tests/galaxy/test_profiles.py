"""Invariant tests for kino.galaxy.profiles (README.md Phase 1)."""

import numpy as np
import pytest
from scipy.integrate import quad

from kino.galaxy.profiles import sersic_bn, sersic_effective_intensity, sersic_profile


def test_sersic_bn_exponential_matches_known_value():
    # Standard literature value for n=1 (e.g. Ciotti & Bertin 1999): b_1 ~= 1.678347.
    assert sersic_bn(1.0) == pytest.approx(1.678347, abs=1e-3)


def test_profile_equals_amplitude_at_effective_radius():
    # exp(-b_n * ((r_e/r_e)**(1/n) - 1)) = exp(0) = 1 exactly, for any n, b_n.
    for n in (0.5, 1.0, 2.0, 4.0):
        assert sersic_profile(3.0, r_e=3.0, n=n, amplitude=5.0) == pytest.approx(5.0)


def test_profile_monotonically_decreasing():
    r = np.linspace(0, 10, 200)
    profile = sersic_profile(r, r_e=2.0, n=1.0, amplitude=1.0)
    assert np.all(np.diff(profile) <= 0)


def test_profile_positive():
    r = np.linspace(0, 50, 500)
    profile = sersic_profile(r, r_e=1.5, n=1.0, amplitude=2.0)
    assert np.all(profile > 0)


def test_effective_intensity_recovers_total_flux():
    r_e, n, total_flux = 2.0, 1.0, 7.0
    i_e = sersic_effective_intensity(total_flux, r_e, n=n, axis_ratio=1.0)

    def integrand(r):
        return sersic_profile(r, r_e, n=n, amplitude=i_e) * 2 * np.pi * r

    integral, _ = quad(integrand, 0, 60 * r_e)

    assert integral == pytest.approx(total_flux, rel=1e-3)


def test_effective_intensity_scales_inversely_with_axis_ratio():
    r_e, n, total_flux = 2.0, 1.0, 7.0
    i_e_faceon = sersic_effective_intensity(total_flux, r_e, n=n, axis_ratio=1.0)
    i_e_inclined = sersic_effective_intensity(total_flux, r_e, n=n, axis_ratio=0.5)
    assert i_e_inclined == pytest.approx(2.0 * i_e_faceon)
