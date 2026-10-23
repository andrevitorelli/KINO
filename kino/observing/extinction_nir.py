"""Broadband near-infrared (J/H/Ks) atmospheric extinction, CTIO (assumed for Cerro Pachon).

``kino.observing.extinction``'s CTIO curve (Stone & Baldwin 1983; Baldwin & Stone
1984) is optical only and stops at 1.087 um. This module adds the near-infrared
regime from two independent, real, measured **broadband** (not continuous) mean
extinction coefficients at CTIO:

* Nikolaev, Weinberg, Skrutskie, Cutri, Wheelock, Gizis & Howard (2000, AJ,
  120, 3340; arXiv:astro-ph/0008002), Table 2 -- a long-term "global
  photometric calibration" average from the Southern (Cerro Tololo) 2MASS
  1.3 m telescope: ``A_J = 0.092 +/- 0.001``, ``A_H = 0.031 +/- 0.001``,
  ``A_Ks = 0.065 +/- 0.001`` mag/airmass. Used below as the primary value
  (smaller formal uncertainty, longer baseline, and the same Cerro Tololo
  site ``kino.observing.extinction``'s own optical curve is sourced from).
* Frogel (1998, PASP, 110, 200; arXiv:astro-ph/9712188), ~200 nights of JHKL
  photometry at CTIO (1978-1992), quoted in Nikolaev et al.'s own Table 2 for
  comparison: ``A_J = 0.100 +/- 0.024``, ``A_H = 0.055 +/- 0.021``,
  ``A_Ks = 0.085 +/- 0.018`` mag/airmass -- notably higher than the 2MASS
  value in H and Ks, a real, unresolved disagreement between the two papers
  (both separately note seasonal variation of a comparable size, tracking
  water vapour). Kept here as :data:`NIR_EXTINCTION_FROGEL` for comparison
  only -- :func:`nir_extinction_coefficient` uses the 2MASS value.

Unlike ``kino.observing.extinction``'s fine-wavelength CTIO optical table,
this is only **three single numbers**, each a long-term mean over its own
filter bandpass -- there is no measurement between J, H and Ks.
Interpolating across them would cross the deep telluric water-vapour
absorption bands near 1.4 and 1.9 um, where real transmission falls close to
zero: exactly the kind of "plausible, wrong answer" ``CLAUDE.md`` warns
against. :func:`nir_extinction_coefficient` therefore never interpolates
between bands: a wavelength must fall inside one band's own isophotal
bandwidth (Cohen, Wheaton & Megeath 2003, AJ, 126, 1090, Table 2 -- J:
1.235 +/- 0.081 um; H: 1.662 +/- 0.1255 um; Ks: 2.159 +/- 0.131 um, using the
table's own bandwidth as a +/- half-width around the isophotal wavelength)
or ``ValueError`` is raised, mirroring
``kino.observing.extinction.extinction_coefficient``'s own "raise, don't
extrapolate" rule. This is deliberately conservative: real filter
transmission extends further into the wings than the isophotal bandwidth, so
a wavelength just outside one of these ranges is not necessarily
unobservable -- it is simply not covered by this verified curve.

**This does not unblock `forecast_ifu`/`forecast_slit` for NIR targets by
itself.** Their own sky-spectrum library (``kino.observing.skymodel``) is the
binding constraint (it stops at 1 micron, built from ESO SkyCalc, see
``CLAUDE.md``) -- extending it needs new SkyCalc queries
(``scripts/build_skycalc_library.py --wmax <nm>``), which requires live
network access to ESO's SkyCalc API host
(``etimecalret-002.eso.org``); that host was unreachable (TCP connect
timeout, not a DNS or code issue) from the sandbox this module was written
in. **2026-10-23 update**: the user independently confirmed the same
TCP-level unreachability from a separate, unrelated network (VPN off, no
proxy, IPv4 forced, DNS identical across three resolvers, traceroute unable
to route there at all) -- the whole ``134.171.46.0/24`` ESO exposure-time-
calculator subnet (``etimecalret-002``/``etimecal`` both on it) appears
unreachable from outside, while ``www.eso.org`` (a different subnet) is
fine, pointing to a problem on ESO's side (or just upstream of it), not a
sandbox artefact or a client/request-volume issue. Reported to ESO; retry
``scripts/build_skycalc_library.py --wmax <nm>`` once that's resolved. This
module is usable standalone today (e.g. to extinction-correct a
J/H/Ks-band line flux by hand, as ``notebooks/13_gmtifs_forecast.ipynb``'s
own `shear_fisher`-direct workaround could), but ``gmtifs.yaml``'s own
1.10-1.14 um demo window falls just short of the J range above
(1.154-1.316 um) -- it is **not** covered by this module either; see that
notebook's own workaround for that specific case.
"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np


@dataclass(frozen=True)
class _Band:
    name: str
    center_um: float
    halfwidth_um: float
    mag_per_airmass: float
    mag_per_airmass_err: float


#: Primary value: Nikolaev et al. (2000) CTIO/2MASS "South" long-term average,
#: windowed by Cohen, Wheaton & Megeath (2003)'s isophotal bandwidth (see the
#: module docstring for both citations).
NIR_EXTINCTION_BANDS: tuple[_Band, ...] = (
    _Band("J", 1.235, 0.081, 0.092, 0.001),
    _Band("H", 1.662, 0.1255, 0.031, 0.001),
    _Band("Ks", 2.159, 0.131, 0.065, 0.001),
)

#: Frogel (1998) CTIO values, kept for comparison only -- see the module
#: docstring's "real, unresolved disagreement" note. (mag/airmass, uncertainty)
NIR_EXTINCTION_FROGEL: dict[str, tuple[float, float]] = {
    "J": (0.100, 0.024),
    "H": (0.055, 0.021),
    "Ks": (0.085, 0.018),
}


def nir_extinction_coefficient(wavelength: u.Quantity) -> np.ndarray:
    """Broadband CTIO J/H/Ks extinction coefficient (mag per airmass) at ``wavelength``.

    No interpolation between bands (see the module docstring): ``wavelength``
    must fall inside one band's own isophotal-bandwidth window. Raises
    ``ValueError`` otherwise, naming the covered windows.
    """
    wave_um = np.atleast_1d(u.Quantity(wavelength).to_value(u.um)).astype(float)
    out = np.full(wave_um.shape, np.nan)
    covered = np.zeros(wave_um.shape, dtype=bool)
    for band in NIR_EXTINCTION_BANDS:
        lo, hi = band.center_um - band.halfwidth_um, band.center_um + band.halfwidth_um
        inside = (wave_um >= lo) & (wave_um <= hi)
        out[inside] = band.mag_per_airmass
        covered |= inside
    if not covered.all():
        bad = np.round(wave_um[~covered], 4)
        windows = ", ".join(
            f"{b.name} {b.center_um - b.halfwidth_um:.3f}-{b.center_um + b.halfwidth_um:.3f} um"
            for b in NIR_EXTINCTION_BANDS
        )
        raise ValueError(
            f"wavelength(s) {bad} um fall outside every covered NIR band ({windows}) -- "
            "no measured broadband extinction there (see the module docstring)"
        )
    return out


def nir_atmospheric_transmission(wavelength: u.Quantity, airmass: float | np.ndarray) -> np.ndarray:
    """Fraction of NIR light surviving the atmosphere, ``10^(-0.4 k X)``.

    Same convention as ``kino.observing.extinction.atmospheric_transmission``
    (infinite airmass gives zero transmission; raises on ``airmass < 1``), but
    ``k`` comes from :func:`nir_extinction_coefficient`.
    """
    x = np.asarray(airmass, dtype=float)
    if np.any(x < 1.0):
        raise ValueError("airmass must be >= 1")
    k = nir_extinction_coefficient(wavelength)
    with np.errstate(over="ignore"):
        return 10.0 ** (-0.4 * k * x)
