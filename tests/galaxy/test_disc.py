"""Invariant tests for kino.galaxy.disc.DiscGalaxy (README.md Phase 1).

Covers the roadmap's required checks -- face-on zero projected rotation,
edge-on limits, symmetry -- plus the position-angle/redshifted-side
convention that kino/io/kinms.py's module docstring flags as needing a
cross-check once this module exists.
"""

import numpy as np
import pytest

from kino.galaxy.disc import DiscGalaxy


def _grid(n=15, extent=8.0):
    coords = np.linspace(-extent, extent, n)
    x, y = np.meshgrid(coords, coords)
    return x, y


def test_face_on_has_zero_projected_rotation():
    gal = DiscGalaxy(r_e=1.0, inc=0.0, pa=40.0, v_flat=200.0, v_sys=50.0)
    x, y = _grid()
    assert np.allclose(gal.vlos(x, y), 50.0)


def test_edge_on_major_axis_matches_rotation_curve_exactly():
    gal = DiscGalaxy(r_e=1.0, inc=90.0, pa=0.0, v_flat=200.0, r_t=2.0, v_sys=10.0)
    pa_rad = np.radians(gal.pa)
    d = np.array([0.5, 1.0, 2.0, 5.0])
    x = gal.x0 + d * np.sin(pa_rad)
    y = gal.y0 + d * np.cos(pa_rad)

    from kino.galaxy.rotation_curve import arctan

    expected = gal.v_sys + arctan(d, gal.v_flat, gal.r_t)
    assert np.allclose(gal.vlos(x, y), expected)
    # the opposite side of the major axis is exactly antisymmetric about v_sys
    assert np.allclose(
        gal.vlos(gal.x0 - d * np.sin(pa_rad), gal.y0 - d * np.cos(pa_rad)), 2 * gal.v_sys - expected
    )


def test_edge_on_off_axis_has_no_flux_or_rotation():
    gal = DiscGalaxy(r_e=1.0, inc=90.0, pa=0.0, v_flat=200.0, v_sys=10.0)
    # a point off the major axis (x != 0 at pa=0 means off the minor... here pa=0
    # so the major axis is the y-axis; (2, 0) is off it)
    assert gal.surface_brightness(2.0, 0.0) == pytest.approx(0.0)
    assert gal.vlos(2.0, 0.0) == pytest.approx(gal.v_sys)


def test_vlos_point_antisymmetric_about_centroid():
    gal = DiscGalaxy(r_e=1.0, inc=55.0, pa=25.0, v_flat=180.0, v_sys=30.0, x0=1.0, y0=-2.0)
    rng = np.random.default_rng(0)
    dx, dy = rng.uniform(-5, 5, size=(2, 50))
    v_plus = gal.vlos(gal.x0 + dx, gal.y0 + dy)
    v_minus = gal.vlos(gal.x0 - dx, gal.y0 - dy)
    assert np.allclose(v_plus - gal.v_sys, -(v_minus - gal.v_sys))


def test_surface_brightness_point_symmetric_about_centroid():
    gal = DiscGalaxy(r_e=1.0, inc=55.0, pa=25.0, v_flat=180.0, x0=1.0, y0=-2.0)
    rng = np.random.default_rng(1)
    dx, dy = rng.uniform(-5, 5, size=(2, 50))
    assert np.allclose(
        gal.surface_brightness(gal.x0 + dx, gal.y0 + dy),
        gal.surface_brightness(gal.x0 - dx, gal.y0 - dy),
    )


def test_ellipticity_limits():
    face_on = DiscGalaxy(r_e=1.0, inc=0.0, pa=0.0, v_flat=100.0)
    edge_on = DiscGalaxy(r_e=1.0, inc=90.0, pa=0.0, v_flat=100.0)
    assert face_on.ellipticity == pytest.approx(0.0, abs=1e-12)
    assert edge_on.ellipticity == pytest.approx(1.0, abs=1e-8)


def test_velocity_dispersion_floor_and_gradient():
    gal = DiscGalaxy(r_e=1.0, inc=45.0, pa=0.0, v_flat=100.0, sigma0=20.0, sigma_grad=-5.0)
    # far out, a negative gradient must not drive sigma below zero
    assert gal.velocity_dispersion(100.0, 0.0) == pytest.approx(0.0)
    assert gal.velocity_dispersion(gal.x0, gal.y0) == pytest.approx(20.0)


def test_pa_zero_redshifted_side_matches_kinms_convention():
    """kino/io/kinms.py's module docstring flags this as unverified until
    kino.galaxy lands: at pa=0, the +y (North) end of the major axis should
    be redshifted -- which is exactly what KinMS's wrapper independently
    produces (see tests/io/test_kinms.py::test_position_angle_convention)."""
    gal = DiscGalaxy(r_e=1.0, inc=85.0, pa=0.0, v_flat=150.0, r_t=1.0)
    assert gal.vlos(gal.x0, gal.y0 + 5.0) - gal.v_sys > 0  # +y = North = redshifted
    assert gal.vlos(gal.x0, gal.y0 - 5.0) - gal.v_sys < 0  # -y = South = blueshifted


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"inc": 91.0}, "inc"),
        ({"inc": -1.0}, "inc"),
        ({"r_e": 0.0}, "r_e"),
        ({"sersic_n": -1.0}, "sersic_n"),
        ({"rotation_curve": "bogus"}, "rotation_curve"),
        ({"r_t": -1.0}, "r_t"),
    ],
)
def test_invalid_parameters_raise(kwargs, message):
    base = {"r_e": 1.0, "inc": 45.0, "pa": 0.0, "v_flat": 100.0}
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        DiscGalaxy(**base)
