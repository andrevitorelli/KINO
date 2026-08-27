"""Invariant tests for kino.lensing.scene.LensedScene (README.md Phase 2).

Includes the "misalignment" physics this module exists to produce: the
photometric position angle of a sheared disc drifts away from the disc's
true (kinematic) position angle -- except when the shear happens to be
aligned with the disc's own axes, where the two exactly coincide. This is
the effect behind the demo notebook and the plots the user asked for.
"""

import numpy as np
import pytest

from kino.galaxy import DiscGalaxy
from kino.lensing import apply_shear
from kino.lensing.moments import measure_ellipticity
from kino.lensing.shear import shear_ellipticity


def _grid(n=25, extent=10.0):
    coords = np.linspace(-extent, extent, n)
    return np.meshgrid(coords, coords)


def test_zero_shear_is_identity():
    gal = DiscGalaxy(r_e=2.0, inc=55.0, pa=25.0, v_flat=180.0, x0=1.0, y0=-0.5)
    scene = apply_shear(gal, 0.0, 0.0)
    x, y = _grid()
    assert np.allclose(scene.surface_brightness(x, y), gal.surface_brightness(x, y))
    assert np.allclose(scene.vlos(x, y), gal.vlos(x, y))
    assert np.allclose(scene.velocity_dispersion(x, y), gal.velocity_dispersion(x, y))


def test_vlos_at_centroid_is_shear_invariant():
    # the shear centre (the galaxy's own centroid) is a fixed point of the
    # delensing map for any g, so Vlos there is always exactly v_sys
    gal = DiscGalaxy(r_e=2.0, inc=55.0, pa=25.0, v_flat=180.0, v_sys=42.0, x0=3.0, y0=-1.0)
    for g1, g2 in [(0.0, 0.0), (0.2, -0.1), (-0.15, 0.25)]:
        scene = apply_shear(gal, g1, g2)
        assert scene.vlos(gal.x0, gal.y0) == pytest.approx(42.0)


def test_photometric_ellipticity_matches_direct_shear_formula():
    gal = DiscGalaxy(r_e=2.0, inc=60.0, pa=25.0, v_flat=180.0)
    g1, g2 = 0.1, -0.05
    scene = apply_shear(gal, g1, g2)

    phi_rad = np.pi / 2.0 - np.radians(gal.pa)
    eps_int = gal.ellipticity * np.exp(2j * phi_rad)
    eps_obs = shear_ellipticity(eps_int, g1, g2)
    expected_mag = abs(eps_obs)
    expected_pa = (np.degrees(np.pi / 2.0 - np.angle(eps_obs) / 2.0)) % 180.0

    mag, pa = scene.photometric_ellipticity
    assert mag == pytest.approx(expected_mag)
    assert pa == pytest.approx(expected_pa)


def test_photometric_ellipticity_matches_rendered_image_moments():
    gal = DiscGalaxy(r_e=2.0, inc=60.0, pa=25.0, v_flat=180.0, sersic_n=1.0)
    scene = apply_shear(gal, 0.15, -0.08)

    extent = 8 * gal.r_e
    coords = np.linspace(-extent, extent, 300)
    x, y = np.meshgrid(coords, coords)
    image = scene.surface_brightness(x, y)
    _, _, eps1, eps2, _, _ = measure_ellipticity(image, x, y)
    measured_mag = np.hypot(eps1, eps2)
    measured_phi = 0.5 * np.arctan2(eps2, eps1)
    measured_pa = np.degrees(np.pi / 2.0 - measured_phi) % 180.0

    analytic_mag, analytic_pa = scene.photometric_ellipticity
    assert measured_mag == pytest.approx(analytic_mag, rel=0.1)
    assert min(abs(measured_pa - analytic_pa), 180 - abs(measured_pa - analytic_pa)) < 2.0


def test_misalignment_vanishes_when_shear_aligned_with_disc_axes():
    gal = DiscGalaxy(r_e=2.0, inc=60.0, pa=25.0, v_flat=180.0)
    phi_rad = np.pi / 2.0 - np.radians(gal.pa)
    g = 0.2 * np.exp(2j * phi_rad)  # shear phase-aligned with the disc's own ellipse

    scene = apply_shear(gal, g.real, g.imag)
    _, pa_obs = scene.photometric_ellipticity
    assert pa_obs == pytest.approx(gal.pa, abs=1e-8)


def test_misalignment_is_nonzero_for_generic_shear():
    gal = DiscGalaxy(r_e=2.0, inc=60.0, pa=25.0, v_flat=180.0)
    scene = apply_shear(gal, 0.0, 0.2)  # not aligned with pa=25
    _, pa_obs = scene.photometric_ellipticity
    misalignment = min(abs(pa_obs - gal.pa), 180 - abs(pa_obs - gal.pa))
    assert misalignment > 1.0


def test_rejects_supercritical_shear():
    gal = DiscGalaxy(r_e=1.0, inc=45.0, pa=0.0, v_flat=100.0)
    with pytest.raises(ValueError, match=r"\|g\|"):
        apply_shear(gal, 0.8, 0.8)
