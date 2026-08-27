"""Invariant tests for kino.lensing.moments, against a closed-form Gaussian."""

import numpy as np
import pytest

from kino.lensing.moments import measure_ellipticity, measure_moments


def _gaussian_image(qxx, qyy, qxy, n=401, x0=0.3, y0=-0.2):
    cov = np.array([[qxx, qxy], [qxy, qyy]])
    max_std = np.sqrt(np.linalg.eigvalsh(cov).max())
    extent = 10.0 * max_std  # cover well beyond the major axis for accurate moments
    coords = np.linspace(-extent, extent, n)
    x, y = np.meshgrid(coords + x0, coords + y0)
    inv = np.linalg.inv(cov)
    dx, dy = x - x0, y - y0
    quad = inv[0, 0] * dx**2 + 2 * inv[0, 1] * dx * dy + inv[1, 1] * dy**2
    image = np.exp(-0.5 * quad)
    return image, x, y


def test_moments_recover_known_gaussian_covariance():
    qxx, qyy, qxy = 4.0, 1.0, 0.8
    image, x, y = _gaussian_image(qxx, qyy, qxy)
    xbar, ybar, qxx_meas, qyy_meas, qxy_meas = measure_moments(image, x, y)

    assert xbar == pytest.approx(0.3, abs=5e-3)
    assert ybar == pytest.approx(-0.2, abs=5e-3)
    assert qxx_meas == pytest.approx(qxx, rel=2e-2)
    assert qyy_meas == pytest.approx(qyy, rel=2e-2)
    assert qxy_meas == pytest.approx(qxy, rel=2e-2)


def test_ellipticity_of_circular_gaussian_is_zero():
    image, x, y = _gaussian_image(2.0, 2.0, 0.0)
    _, _, eps1, eps2, a, b = measure_ellipticity(image, x, y)
    assert eps1 == pytest.approx(0.0, abs=1e-6)
    assert eps2 == pytest.approx(0.0, abs=1e-6)
    assert a == pytest.approx(b, rel=1e-6)


def test_ellipticity_orientation_along_x():
    # elongated along x (qxx > qyy, qxy=0) => eps1 > 0, eps2 == 0 (see
    # kino.lensing.shear's module docstring: phi=0 means elongation along +x)
    image, x, y = _gaussian_image(9.0, 1.0, 0.0)
    _, _, eps1, eps2, a, b = measure_ellipticity(image, x, y)
    assert eps1 > 0
    assert eps2 == pytest.approx(0.0, abs=1e-6)
    expected_eps = (3.0 - 1.0) / (3.0 + 1.0)  # a=sqrt(9)=3, b=sqrt(1)=1
    assert eps1 == pytest.approx(expected_eps, rel=2e-2)


def test_ellipticity_orientation_along_diagonal():
    # a 45-degree diagonal ellipse (equal qxx, qyy, nonzero qxy) => pure eps2
    image, x, y = _gaussian_image(3.0, 3.0, 1.5)
    _, _, eps1, eps2, a, b = measure_ellipticity(image, x, y)
    assert eps2 > 0
    assert eps1 == pytest.approx(0.0, abs=1e-6)
