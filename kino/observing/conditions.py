"""Target geometry, moon state and seeing scaling.

Sky *brightness* (moonlight, airglow) and atmospheric extinction are not
here yet: they need published models and site data that have to be verified
first. This module has only the geometry and the standard scalings.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import (
    TETE,
    AltAz,
    GeocentricTrueEcliptic,
    SkyCoord,
    get_body,
    get_sun,
)
from astropy.coordinates.baseframe import NonRotationTransformationWarning
from astropy.time import Time

from kino.observing.site import Site

#: Wavelength at which quoted (site) seeing is conventionally specified.
REFERENCE_WAVELENGTH = 500.0 * u.nm


def airmass_from_altitude(altitude: u.Quantity) -> np.ndarray:
    """Relative airmass from apparent altitude (Kasten & Young 1989).

    ``X = 1 / [sin h + 0.50572 (h + 6.07995 deg)^-1.6364]`` (``h`` in
    degrees; Kasten & Young 1989, Appl. Opt. 28, 4735). Finite all the way to
    the horizon (X ~ 38), unlike ``sec z``. Returns ``inf`` for targets below
    the horizon (``h <= 0``).
    """
    h = np.atleast_1d(u.Quantity(altitude).to_value(u.deg)).astype(float)
    x = np.full(h.shape, np.inf)
    up = h > 0.0
    x[up] = 1.0 / (np.sin(np.radians(h[up])) + 0.50572 * (h[up] + 6.07995) ** -1.6364)
    return x


@dataclass(frozen=True)
class TargetTrack:
    """A target's position over a set of times.

    Attributes
    ----------
    altitude, azimuth : Quantity (deg)
        Apparent horizontal coordinates (azimuth East of North).
    airmass : numpy.ndarray
        Kasten-Young airmass (``inf`` below the horizon).
    parallactic_angle : Quantity (deg)
        Angle at the target between the direction to the zenith and to the
        North celestial pole, positive West of the meridian. A slit aligned
        to this angle sees no differential-refraction losses.
    """

    altitude: u.Quantity
    azimuth: u.Quantity
    airmass: np.ndarray
    parallactic_angle: u.Quantity


def target_track(site: Site, coord: SkyCoord, times: Time) -> TargetTrack:
    """Altitude, airmass and parallactic angle of ``coord`` at ``times``."""
    altaz = coord.transform_to(AltAz(obstime=times, location=site.location))
    # hour angle needs the target's RA *of date* (apparent sidereal time is
    # referred to the true equinox of date); J2000 RA would be off by the
    # accumulated precession (~0.3 deg after 25 years).
    of_date = coord.transform_to(TETE(obstime=times))
    lst = times.sidereal_time("apparent", longitude=site.location.lon)
    hour_angle = (lst - of_date.ra).wrap_at(180 * u.deg)
    lat, dec = site.location.lat, of_date.dec
    q = np.arctan2(
        np.sin(hour_angle),
        np.tan(lat) * np.cos(dec) - np.sin(dec) * np.cos(hour_angle),
    )
    return TargetTrack(
        altitude=altaz.alt.to(u.deg),
        azimuth=altaz.az.to(u.deg),
        airmass=airmass_from_altitude(altaz.alt),
        parallactic_angle=q.to(u.deg),
    )


@dataclass(frozen=True)
class MoonState:
    """The Moon as seen from the site.

    Attributes
    ----------
    illumination : numpy.ndarray
        Illuminated fraction of the disc, 0 (new) to 1 (full).
    phase_angle : Quantity (deg)
        Sun-Moon-Earth angle: 0 at full moon, 180 at new moon. (The "lunar
        phase angle" ``alpha`` used by sky-brightness models.)
    altitude : Quantity (deg)
    separation : Quantity (deg) or None
        Angular distance from the target, if a target was given.
    """

    illumination: np.ndarray
    phase_angle: u.Quantity
    altitude: u.Quantity
    separation: u.Quantity | None


def moon_state(site: Site, times: Time, coord: SkyCoord | None = None) -> MoonState:
    """Moon illumination, phase angle, altitude and (optionally) separation.

    The phase angle uses the Sun-Moon elongation and the true distances
    (``i = atan2(R_sun sin e, R_moon - R_sun cos e)``); accurate to well under
    a degree, which is far finer than sky-brightness models need.
    """
    # geocentric Sun and Moon for the phase (both in the same frame); the
    # topocentric Moon for altitude and separation from the target
    moon_geo = get_body("moon", times)
    moon = get_body("moon", times, location=site.location)
    sun = get_sun(times)
    elong = sun.separation(moon_geo)
    r_sun, r_moon = sun.distance.to(u.au), moon_geo.distance.to(u.au)
    phase = np.arctan2(r_sun * np.sin(elong), r_moon - r_sun * np.cos(elong))
    altaz = moon.transform_to(AltAz(obstime=times, location=site.location))
    sep = None
    if coord is not None:
        # ``moon`` is in the site's topocentric GCRS frame and ``coord`` (a distant
        # target) is transformed into it, so this is the topocentric separation --
        # the one that matters for scattered moonlight (lunar parallax is ~1 deg).
        # astropy warns that the result can depend on the transform direction; for a
        # target at infinite distance it does not.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NonRotationTransformationWarning)
            sep = moon.separation(coord).to(u.deg)
    return MoonState(
        illumination=(1.0 + np.cos(phase)) / 2.0,
        phase_angle=phase.to(u.deg),
        altitude=altaz.alt.to(u.deg),
        separation=sep,
    )


def seeing_at(
    seeing_ref: u.Quantity,
    airmass: float | np.ndarray,
    wavelength: u.Quantity,
    *,
    reference_wavelength: u.Quantity = REFERENCE_WAVELENGTH,
) -> u.Quantity:
    """Effective seeing FWHM at an airmass and wavelength.

    Standard Kolmogorov scaling from a zenith seeing quoted at
    ``reference_wavelength`` (conventionally 500 nm): the Fried parameter
    goes as ``r0 ~ lambda^(6/5) X^(-3/5)`` and the FWHM as ``lambda / r0``, so

        FWHM = FWHM_ref * X^(3/5) * (lambda / lambda_ref)^(-1/5).

    An approximation (it ignores the outer scale, which lowers the FWHM at
    long wavelengths, and dome/instrument seeing); use it to *shape* a
    seeing curve, not to predict a single image to better than ~10%.
    """
    x = np.asarray(airmass, dtype=float)
    if np.any(x < 1.0):
        raise ValueError("airmass must be >= 1")
    if u.Quantity(seeing_ref).value <= 0:
        raise ValueError("seeing_ref must be positive")
    scale = x**0.6 * (u.Quantity(wavelength) / reference_wavelength).decompose().value ** -0.2
    return u.Quantity(seeing_ref).to(u.arcsec) * scale


def ecliptic_from_sun(coord: SkyCoord, times: Time) -> tuple[u.Quantity, u.Quantity]:
    """Target ecliptic longitude *relative to the Sun* (wrapped to +/-180 deg)
    and ecliptic latitude, as the zodiacal-light model needs.

    Geocentric true-of-date ecliptic. (SkyCalc's own almanac differed from this
    by ~4 deg in longitude for one tested target; see ``kino.observing.skymodel``.)
    """
    frame = GeocentricTrueEcliptic(obstime=times)
    target = coord.transform_to(frame)
    sun = get_sun(times).transform_to(frame)
    lon = ((target.lon - sun.lon + 180 * u.deg) % (360 * u.deg)) - 180 * u.deg
    return lon.to(u.deg), target.lat.to(u.deg)
