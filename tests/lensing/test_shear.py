"""Invariant tests for kino.lensing.shear (README.md Phase 2)."""

import numpy as np
import pytest

from kino.lensing.shear import magnification, shear_ellipticity, shear_matrix, unshear_ellipticity


def test_zero_shear_is_identity():
    assert np.allclose(shear_matrix(0.0, 0.0), np.eye(2))


def test_matrix_determinant_matches_1_minus_g_squared():
    # standard reduced-shear Jacobian result (kappa=0): det(A) = 1 - |g|^2
    g1, g2 = 0.15, -0.08
    det = np.linalg.det(shear_matrix(g1, g2))
    assert det == pytest.approx(1.0 - (g1**2 + g2**2))


def test_rejects_supercritical_shear():
    with pytest.raises(ValueError, match=r"\|g\|"):
        shear_matrix(0.8, 0.8)


def test_circular_source_ellipticity_equals_g():
    # the defining property of the (a-b)/(a+b) convention under reduced shear
    for g1, g2 in [(0.1, 0.0), (0.0, -0.2), (0.05, 0.07)]:
        eps_obs = shear_ellipticity(0j, g1, g2)
        assert eps_obs == pytest.approx(complex(g1, g2))


def test_zero_shear_ellipticity_is_identity():
    eps_int = complex(0.12, -0.05)
    assert shear_ellipticity(eps_int, 0.0, 0.0) == pytest.approx(eps_int)


def test_shear_unshear_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        eps_int = complex(*rng.uniform(-0.3, 0.3, size=2))
        g1, g2 = rng.uniform(-0.3, 0.3, size=2)
        eps_obs = shear_ellipticity(eps_int, g1, g2)
        recovered = unshear_ellipticity(eps_obs, g1, g2)
        assert recovered == pytest.approx(eps_int)


def test_small_shear_linearisation():
    """The exact Mobius formula's Taylor expansion around g=0 is
    ``eps_int + g - eps_int**2 * conj(g)`` -- *not* simply ``eps_int + g``
    (that naive sum is only zeroth-order accurate whenever eps_int != 0; the
    ``-eps_int**2 * conj(g)`` term is the standard weak-lensing shear
    responsivity, linear in g, and dominates the naive sum's error at small
    g). Check the exact transform converges to the *correct* first-order
    expansion quadratically as g -> 0.
    """
    eps_int = complex(0.1, 0.05)

    def first_order_expansion(g1, g2):
        g = complex(g1, g2)
        return eps_int + g - eps_int**2 * np.conj(g)

    errors = []
    for scale in (1e-2, 1e-3, 1e-4):
        g1, g2 = scale, -2 * scale
        eps_obs = shear_ellipticity(eps_int, g1, g2)
        errors.append(abs(eps_obs - first_order_expansion(g1, g2)))

    assert errors[1] < errors[0] / 50  # ~quadratic convergence per decade in g
    assert errors[2] < errors[1] / 50


def test_zero_shear_has_no_magnification():
    assert magnification(0.0, 0.0) == pytest.approx(1.0)


def test_magnification_matches_matrix_determinant():
    g1, g2 = 0.2, -0.15
    assert magnification(g1, g2) == pytest.approx(1.0 / np.linalg.det(shear_matrix(g1, g2)))


def test_magnification_exceeds_one_for_any_nonzero_shear():
    rng = np.random.default_rng(1)
    for _ in range(20):
        g1, g2 = rng.uniform(-0.5, 0.5, size=2)
        if g1 == 0.0 and g2 == 0.0:
            continue
        assert magnification(g1, g2) > 1.0


def test_magnification_depends_only_on_shear_magnitude():
    mag = 0.3
    values = [magnification(mag * np.cos(t), mag * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 8)]
    assert np.allclose(values, values[0])
