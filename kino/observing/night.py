"""The usable dark window of a night."""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, get_sun
from astropy.time import Time

from kino.observing.site import Site


@dataclass(frozen=True)
class Night:
    """A night's dark window on a regular time grid.

    Attributes
    ----------
    times : astropy.time.Time
        Grid points with the Sun below the requested altitude, in order.
    sun_altitude : Quantity (deg)
        Sun altitude at ``times``.
    start, end : astropy.time.Time
        First and last grid point.
    """

    times: Time
    sun_altitude: u.Quantity
    start: Time
    end: Time

    @property
    def duration(self) -> u.Quantity:
        return (self.end - self.start).to(u.hour)


def night_window(
    site: Site,
    date: str | Time,
    *,
    sun_altitude_max: u.Quantity = -18.0 * u.deg,
    step: u.Quantity = 5.0 * u.min,
) -> Night:
    """The dark window of the night that *begins* on local evening ``date``.

    ``date`` is a calendar date (``"2026-10-03"``): the evening of that date
    at the site, running into the next morning. The search starts at the
    site's mean solar noon on that date (12h UT minus the longitude in hours)
    and takes the first contiguous stretch with the Sun below
    ``sun_altitude_max`` -- so no timezone or daylight-saving handling is
    needed, and the result is unambiguous across the UT date boundary.

    Parameters
    ----------
    sun_altitude_max : Quantity (angle), optional
        Default -18 deg (astronomical twilight); use -12 deg for nautical.
    step : Quantity (time), optional
        Grid spacing.
    """
    day = Time(date, scale="utc")
    noon = Time(int(day.mjd) + 0.5, format="mjd", scale="utc") - site.longitude.hour * u.hour
    n = int(round((24 * u.hour / step).decompose().value))
    times = noon + np.arange(n) * step

    sun_alt = get_sun(times).transform_to(AltAz(obstime=times, location=site.location)).alt
    dark = sun_alt < sun_altitude_max
    if not dark.any():
        raise ValueError(
            f"the Sun never drops below {sun_altitude_max} at {site.name} on {day.iso[:10]}"
        )

    first = int(np.argmax(dark))
    stretch = dark[first:]
    length = int(np.argmin(stretch)) if not stretch.all() else len(stretch)
    sel = slice(first, first + length)
    return Night(
        times=times[sel],
        sun_altitude=sun_alt[sel].to(u.deg),
        start=times[sel][0],
        end=times[sel][-1],
    )
