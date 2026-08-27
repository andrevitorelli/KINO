"""Reduced-shear coordinate transform and its effect on ellipticity.

Weak-lensing convention (Bartelmann & Schneider 2001; kappa=0, pure shear --
KINO does not model *convergence* as a free parameter). Pure shear alone
still magnifies (see :func:`magnification`): with kappa=0, reduced shear
equals shear exactly, but ``det(A) = 1 - |g|**2 != 1``, so even an
unconverged lens changes the observed solid angle -- and therefore total
flux -- of the source. The lens Jacobian ``A = d(beta)/d(theta)`` maps an
**observed** sky position ``theta`` to the **source-plane** position
``beta`` it came from::

    A = [[1 - g1,   -g2  ],
         [  -g2,   1 + g1]]
    beta = A @ theta

This is the standard ray-tracing direction: to render what an intrinsic
(source-plane) model looks like once lensed, evaluate the source model at
``beta = A @ theta`` for each observed pixel ``theta`` (see
``kino.lensing.scene.LensedScene``).

Ellipticity convention
-----------------------
KINO's complex ellipticity is ``eps = eps1 + i*eps2``, with magnitude
``eps = (a-b)/(a+b)`` (``CLAUDE.md``) and phase ``2*phi``, ``phi`` the
position angle measured **the standard math way, counter-clockwise from the
+x (East) axis** -- *not* directly KINO's own ``pa`` (East of North), which
runs the opposite sense from North: ``phi = 90 deg - pa``. This module works
in ``(eps1, eps2)`` / ``phi`` throughout; convert to/from ``kino.galaxy``'s
``pa`` explicitly (see ``kino.lensing.scene.photometric_ellipticity`` for a
worked example) -- do not assume ``eps1 = eps*cos(2*pa)`` directly, that
drops the 90-degree offset and comes out with the wrong sign.

For a **circular** source (``eps=0``), reduced shear ``g = g1 + i*g2``
produces ``eps_obs = g`` exactly -- this is what fixes the sign/orientation
of ``A`` above (elongation from positive ``g1`` alone is along the ``+x``
axis, i.e. ``phi=0``), and is pinned by
``tests/lensing/test_shear.py::test_circular_source_ellipticity_equals_g``.

For a general intrinsic ellipticity, the composition follows the standard
Mobius law (Seitz & Schneider 1997): ``eps_obs = (eps_int + g) / (1 +
conj(g) * eps_int)``. Note this is *not* well-approximated by the naive sum
``eps_int + g`` except when ``eps_int`` is small: expanding in ``g``,
``eps_obs = eps_int + g - eps_int**2 * conj(g) + O(g**2)`` -- the
responsivity term ``-eps_int**2 * conj(g)`` is first order in ``g`` too and
dominates the naive sum's error at small ``g`` whenever ``eps_int != 0``
(see ``tests/lensing/test_shear.py::test_small_shear_linearisation``).
"""

from __future__ import annotations

import numpy as np


def shear_matrix(g1: float, g2: float) -> np.ndarray:
    """The lens Jacobian ``A`` such that ``beta = A @ theta`` (see module docstring)."""
    if g1**2 + g2**2 >= 1.0:
        raise ValueError(f"|g| must be < 1 (reduced shear), got g1={g1}, g2={g2}")
    return np.array([[1.0 - g1, -g2], [-g2, 1.0 + g1]])


def shear_ellipticity(eps: complex, g1: float, g2: float) -> complex:
    """Apply reduced shear ``(g1, g2)`` to intrinsic complex ellipticity ``eps``."""
    g = complex(g1, g2)
    return (eps + g) / (1.0 + np.conj(g) * eps)


def unshear_ellipticity(eps_obs: complex, g1: float, g2: float) -> complex:
    """Inverse of :func:`shear_ellipticity`: recover the intrinsic ``eps``."""
    g = complex(g1, g2)
    return (eps_obs - g) / (1.0 - np.conj(g) * eps_obs)


def magnification(g1: float, g2: float) -> float:
    """Flux magnification ``mu = 1 / det(A) = 1 / (1 - g1**2 - g2**2)``.

    KINO's forward model conserves *surface brightness* under lensing
    (``LensedScene`` just evaluates the source at the delensed coordinate,
    with no extra factor) -- magnification of the *total* flux integrated
    over a fixed observed area emerges automatically from that coordinate
    change of variables, exactly as it should (Liouville's theorem). Since
    ``det(A) = 1 - |g|**2 < 1`` for any nonzero reduced shear, ``mu > 1``
    always -- pure shear alone magnifies, with no convergence needed. This
    matters for anything that uses observed flux as if it were the true
    (unlensed) flux, e.g. an uncorrected Tully-Fisher luminosity.
    """
    return 1.0 / (1.0 - g1**2 - g2**2)
