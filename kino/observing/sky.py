"""V-band night-sky brightness: dark sky plus scattered moonlight.

Implements the moonlight model of Krisciunas & Schaefer (1991, PASP 103, 1033;
equations 1-3 and 15-22 as summarised in their Section 5) and the CTIO
dark-sky levels of Krisciunas et al. (2007, PASP 119, 687; arXiv:0705.1153,
Table 6). :func:`la_serena_worst_case_factor` adds a conservative, V-band-only
upper bound on La Serena/Coquimbo light pollution at high zenith angle, from
the same 2007 paper and its 2010 follow-up (Krisciunas, Bogglio, Sanhueza &
Smith, PASP 122, 373; arXiv:1001.2246) -- see that function's own docstring.

What this is *not*: it is V band only. The moon term has no colour model yet,
so nothing here gives a sky brightness at an arbitrary wavelength or a sky
*spectrum*, and it has no airglow line structure (the OH forest that dominates
red and near-IR sky). K&S91 quote 8-23% rms accuracy for the moonlight term
(worse with the Moon or the target near the horizon); the dark-sky level
varies by ~0.2 mag rms (CTIO Table 6) and by 10-50% within a night.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np

#: Mean zenith dark-sky brightness at CTIO, mag/arcsec^2, and the standard
#: deviation *of the distribution* (not of the mean). 1992-2006 CCD data, Moon
#: down, airmass < 1.6, away from the Galactic plane. Krisciunas et al. (2007),
#: Table 6. The V and B values carry a solar-cycle dependence (brighter near
#: solar maximum); R and I show none.
CTIO_DARK_SKY = {
    "U": (22.12, 0.19),
    "B": (22.82, 0.19),
    "V": (21.79, 0.22),
    "R": (21.19, 0.19),
    "I": (19.85, 0.25),
}

# Krisciunas & Schaefer (1991) eq. (1) and the S10 form quoted in
# Krisciunas et al. (2007): B[nL] = 0.263 * a**(Q - V), a = 100**0.2.
_A = 100.0**0.2
_Q = 27.78151
_S10_NL = 0.263


def nanolamberts_from_mag(mag_arcsec2: float | np.ndarray) -> np.ndarray:
    """Sky surface brightness, mag/arcsec^2 (V) -> nanoLamberts."""
    return _S10_NL * _A ** (_Q - np.asarray(mag_arcsec2, dtype=float))


def mag_from_nanolamberts(nl: float | np.ndarray) -> np.ndarray:
    """Sky surface brightness, nanoLamberts -> mag/arcsec^2 (V)."""
    return _Q - np.log(np.asarray(nl, dtype=float) / _S10_NL) / np.log(_A)


def _night_glow_airmass(zenith_distance: u.Quantity) -> np.ndarray:
    """K&S91 eq. (3): ``X = (1 - 0.96 sin^2 Z)^(-1/2)``.

    The van Rhijn form for an emitting layer, finite at the horizon (X = 5);
    K&S91 use it for both the Moon and the sky path.
    """
    z = np.asarray(u.Quantity(zenith_distance).to_value(u.rad), dtype=float)
    return (1.0 - 0.96 * np.sin(z) ** 2) ** -0.5


def dark_sky_brightness(
    zenith_brightness_mag: float,
    zenith_distance: u.Quantity,
    k: float,
) -> np.ndarray:
    """Dark sky (nanoLamberts) at zenith distance ``Z`` (K&S91 eq. 2-3).

    ``B0(Z) = B_zen * 10^(-0.4 k (X - 1)) * X``: the sky brightens toward the
    horizon (longer path through the emitting layer) but is dimmed by
    extinction. ``k`` is the V-band extinction in mag per airmass.
    """
    b_zen = nanolamberts_from_mag(zenith_brightness_mag)
    x = _night_glow_airmass(zenith_distance)
    return b_zen * 10.0 ** (-0.4 * k * (x - 1.0)) * x


def moonlight_brightness(
    phase_angle: u.Quantity,
    separation: u.Quantity,
    moon_zenith_distance: u.Quantity,
    zenith_distance: u.Quantity,
    k: float,
) -> np.ndarray:
    """Scattered moonlight in the V band, nanoLamberts (K&S91 Section 5).

    ::

        I*   = 10^(-0.4 (3.84 + 0.026|a| + 4e-9 a^4))     lunar illuminance
        f(r) = 10^5.36 (1.06 + cos^2 r) + 10^(6.15 - r/40)    r >= 10 deg
        f(r) = 10^5.36 (1.06 + cos^2 r) + 6.2e7 / r^2         r <  10 deg
        B    = f(r) I* 10^(-0.4 k X(Zm)) [1 - 10^(-0.4 k X(Z))]

    with ``a`` the lunar phase angle (0 = full), ``r`` the Moon-target
    separation, ``Zm``/``Z`` the zenith distances of the Moon and the target.
    The ``r < 10 deg`` branch is the aureole term (K&S91 eq. 19, Schaefer 1991)
    replacing the Mie term; the paper's summary formula is stated for
    r > 10 deg only. Returns 0 when the Moon is below the horizon.

    ``phase_angle``, ``separation`` and the zenith distances broadcast
    together; ``k`` is the V-band extinction (mag per airmass).
    """
    alpha = np.abs(np.asarray(u.Quantity(phase_angle).to_value(u.deg), dtype=float))
    rho = np.asarray(u.Quantity(separation).to_value(u.deg), dtype=float)
    zm = u.Quantity(moon_zenith_distance)
    z = u.Quantity(zenith_distance)

    i_star = 10.0 ** (-0.4 * (3.84 + 0.026 * alpha + 4e-9 * alpha**4))
    cos2 = np.cos(np.radians(rho)) ** 2
    rayleigh = 10.0**5.36 * (1.06 + cos2)
    with np.errstate(divide="ignore"):
        mie = np.where(rho >= 10.0, 10.0 ** (6.15 - rho / 40.0), 6.2e7 / np.maximum(rho, 1e-6) ** 2)
    f = rayleigh + mie

    x_moon = _night_glow_airmass(zm)
    x_sky = _night_glow_airmass(z)
    b = f * i_star * 10.0 ** (-0.4 * k * x_moon) * (1.0 - 10.0 ** (-0.4 * k * x_sky))
    return np.where(np.asarray(zm.to_value(u.deg)) < 90.0, b, 0.0)


def sky_brightness_v(
    dark_zenith_mag: float,
    phase_angle: u.Quantity,
    separation: u.Quantity,
    moon_zenith_distance: u.Quantity,
    zenith_distance: u.Quantity,
    k: float,
) -> dict[str, np.ndarray]:
    """Total V-band sky brightness with the Moon.

    Returns ``{"dark": nL, "moon": nL, "total": nL, "total_mag": mag/arcsec^2,
    "delta_v": mag}`` where ``delta_v = -2.5 log10(total / dark)`` is the
    brightening from moonlight (K&S91 eq. 22, negative = brighter).
    """
    dark = dark_sky_brightness(dark_zenith_mag, zenith_distance, k)
    moon = moonlight_brightness(phase_angle, separation, moon_zenith_distance, zenith_distance, k)
    total = dark + moon
    return {
        "dark": dark,
        "moon": moon,
        "total": total,
        "total_mag": mag_from_nanolamberts(total),
        "delta_v": -2.5 * np.log10(total / dark),
    }


def la_serena_worst_case_factor(zenith_distance: u.Quantity) -> np.ndarray:
    """Worst-case V-band light-pollution enhancement from La Serena/Coquimbo at
    Cerro Tololo (assumed for Cerro Pachon, ~10 km away), vs. zenith distance.

    Krisciunas et al. (2007, PASP, 119, 687; arXiv:0705.1153) found **no**
    detectable effect of the towns' artificial light for elevation > 45 deg
    (zenith distance < 45 deg, airmass < 1.41) -- only below 45 deg elevation
    does La Serena's glow become measurable (visible from ~10 deg elevation).
    Krisciunas, Bogglio, Sanhueza & Smith (2010, PASP, 122, 373;
    arXiv:1001.2246) confirmed this with new data and quantified the worst
    case: on a clear night (no marine cloud layer over La Serena/Coquimbo),
    facing directly toward the town (azimuth roughly 260-310 deg in their
    data), V-band flux reached up to a factor of ~2 (44-72% excess) above the
    Garstang (1989) model near zenith distance 80 deg (airmass ~5.8); B-band
    showed **no** corresponding enhancement at all, and a similarly high
    zenith angle toward Andacollo (a smaller, less light-polluted town, a
    different azimuth) also showed none.

    This returns a **conservative, azimuth-independent upper bound**, not a
    measured correction: 1.0 (no effect) below 45 deg zenith distance,
    linearly ramping to 2.0 by 80 deg (a simple interpolation of the two
    papers' own anchor points, not a physical radiative-transfer model --
    ``kino.observing`` has no notion of "direction toward La Serena" to do
    better), held flat at 2.0 beyond 80 deg (unverified extrapolation past
    the papers' own data). V-band only; nothing here applies to B or other
    bands, where the effect was not detected. ``forecast_ifu``/
    ``forecast_slit``'s own default ``max_airmass=2.0`` corresponds to
    zenith distance 60 deg -- comfortably inside the well-measured
    no-detectable-effect regime (<~70 deg in the 2010 data) for any forecast
    run at that default or lower; this function exists for the unusual case
    of deliberately forecasting at much higher airmass.
    """
    z = np.asarray(u.Quantity(zenith_distance).to_value(u.deg), dtype=float)
    lo, hi = 45.0, 80.0
    frac = np.clip((z - lo) / (hi - lo), 0.0, 1.0)
    return 1.0 + frac
