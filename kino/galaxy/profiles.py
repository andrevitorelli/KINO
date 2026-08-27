"""Sersic surface-brightness profile.

Radii in arcsec (see ``CLAUDE.md``). ``n=1`` is the standard exponential
disc profile.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma


def sersic_bn(n: float | np.ndarray) -> float | np.ndarray:
    """Ciotti & Bertin (1999) asymptotic approximation to ``b_n``.

    ``b_n`` is defined implicitly by the condition that ``r_e`` encloses
    half the total light; this series approximation avoids an iterative
    numerical inversion. Valid for ``n >~ 0.36``.
    """
    n = np.asarray(n, dtype=float)
    return (
        2.0 * n
        - 1.0 / 3.0
        + 4.0 / (405.0 * n)
        + 46.0 / (25515.0 * n**2)
        + 131.0 / (1148175.0 * n**3)
        - 2194697.0 / (30690717750.0 * n**4)
    )


def sersic_profile(
    r: float | np.ndarray, r_e: float, n: float = 1.0, amplitude: float = 1.0
) -> np.ndarray:
    """Sersic profile ``I(r) = amplitude * exp(-b_n * ((r/r_e)**(1/n) - 1))``.

    ``amplitude`` is the intensity at ``r = r_e`` (``I_e``); pass the output
    of :func:`sersic_effective_intensity` to normalise to a given total flux.
    """
    b_n = sersic_bn(n)
    r = np.asarray(r, dtype=float)
    return amplitude * np.exp(-b_n * ((r / r_e) ** (1.0 / n) - 1.0))


def sersic_effective_intensity(
    total_flux: float, r_e: float, n: float = 1.0, axis_ratio: float = 1.0
) -> float:
    """Solve for ``I_e`` so the projected profile integrates to ``total_flux``.

    ``axis_ratio`` is the projected minor/major axis ratio ``q = cos(inc)``
    (``<= 1``), accounting for the compression of the sky-plane area under
    inclination when ``r`` is the deprojected disc-plane radius (see
    ``kino.galaxy.disc``); ``q=1`` is the face-on/circular case.
    """
    b_n = sersic_bn(n)
    norm = 2.0 * np.pi * n * np.exp(b_n) * b_n ** (-2.0 * n) * gamma(2.0 * n)
    return total_flux / (norm * r_e**2 * axis_ratio)
