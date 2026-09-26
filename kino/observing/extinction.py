"""Atmospheric extinction at Cerro Tololo (used for Cerro Pachon).

The curve is the CTIO extinction table distributed with IRAF, which follows
Stone & Baldwin (1983, MNRAS 204, 347) out to ~6400 A and Baldwin & Stone
(1984, MNRAS 206, 241) redward, with averages in the overlap (see the
``specreduce`` documentation, "Atmospheric Extinction", note 1). The numbers
below are read from the ``ctioextinct.dat`` file in the ``astropy/specreduce-data``
repository, which mirrors that IRAF table.

Caveats, all of which make this an *average* rather than a per-night value:

* it was measured at Cerro Tololo, ~10 km from Cerro Pachon, and is assumed to
  hold there (Gemini makes the same assumption of similar conditions);
* it dates from the early 1980s and does not track aerosol/dust variations,
  which change extinction by tens of percent from night to night;
* it does not include cloud, which is a separate (grey) attenuation.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np

#: Wavelengths of the tabulated CTIO extinction curve, Angstrom.
CTIO_WAVELENGTH_ANGSTROM = np.array(
    [
        3200.0, 3250.0, 3300.0, 3350.0, 3400.0, 3450.0, 3500.0, 3571.0, 3636.0, 3704.0,
        3862.0, 4036.0, 4167.0, 4255.0, 4464.0, 4566.0, 4785.0, 5000.0, 5263.0, 5556.0,
        5840.0, 6056.0, 6436.0, 6790.0, 7100.0, 7550.0, 7780.0, 8090.0, 8370.0, 8708.0,
        9832.0, 10256.0, 10400.0, 10610.0, 10796.0, 10870.0,
    ]
)  # fmt: skip

#: Extinction coefficient at those wavelengths, mag per airmass.
CTIO_EXTINCTION_MAG_PER_AIRMASS = np.array(
    [
        1.05, 0.91, 0.82, 0.75, 0.70, 0.66, 0.62, 0.57, 0.55, 0.50,
        0.42, 0.38, 0.34, 0.31, 0.27, 0.25, 0.22, 0.19, 0.17, 0.14,
        0.16, 0.15, 0.11, 0.071, 0.067, 0.053, 0.049, 0.053, 0.050, 0.030,
        0.053, 0.023, 0.024, 0.026, 0.029, 0.037,
    ]
)  # fmt: skip


def extinction_coefficient(wavelength: u.Quantity) -> np.ndarray:
    """CTIO extinction coefficient ``k`` (mag per airmass) at ``wavelength``.

    Linear interpolation in wavelength. Raises ``ValueError`` outside
    3200-10870 A rather than extrapolating.
    """
    wave = np.atleast_1d(u.Quantity(wavelength).to_value(u.angstrom)).astype(float)
    lo, hi = CTIO_WAVELENGTH_ANGSTROM[[0, -1]]
    tol = 1e-6  # Angstrom: unit-conversion round-off at the table edges
    if np.any(wave < lo - tol) or np.any(wave > hi + tol):
        raise ValueError(f"wavelength outside the tabulated CTIO range {lo:.0f}-{hi:.0f} A")
    return np.interp(wave, CTIO_WAVELENGTH_ANGSTROM, CTIO_EXTINCTION_MAG_PER_AIRMASS)


def atmospheric_transmission(wavelength: u.Quantity, airmass: float | np.ndarray) -> np.ndarray:
    """Fraction of light surviving the atmosphere, ``10^(-0.4 k X)``.

    ``wavelength`` and ``airmass`` broadcast against each other
    (``wavelength[:, None]`` and ``airmass[None, :]`` give a grid). Returns 0
    for infinite airmass (target below the horizon).
    """
    x = np.asarray(airmass, dtype=float)
    if np.any(x < 1.0):
        raise ValueError("airmass must be >= 1")
    k = extinction_coefficient(wavelength)
    with np.errstate(over="ignore"):
        return 10.0 ** (-0.4 * k * x)
