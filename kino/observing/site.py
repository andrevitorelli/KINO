"""Observing sites.

Only sites whose coordinates have been checked against a primary source are
predefined; build a :class:`Site` yourself from an ``EarthLocation`` for
anything else.
"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
from astropy.coordinates import EarthLocation


@dataclass(frozen=True)
class Site:
    """A named observing site.

    Parameters
    ----------
    name : str
    location : astropy.coordinates.EarthLocation
        Geodetic (WGS84) position; ``height`` is the *ellipsoidal* height.
    """

    name: str
    location: EarthLocation

    @property
    def latitude(self) -> u.Quantity:
        return self.location.lat

    @property
    def longitude(self) -> u.Quantity:
        return self.location.lon


def _dms(sign: int, d: float, m: float, s: float) -> u.Quantity:
    return sign * (d + m / 60.0 + s / 3600.0) * u.deg


#: SOAR 4.1-m telescope, Cerro Pachon. Geodetic latitude -30 14 16.41,
#: longitude -70 44 01.11, ellipsoidal height 2738 m (orthometric elevation
#: 2703 m): Mamajek (2013), arXiv:1210.1616v3, Table 6 (positions from GPS;
#: the height was adopted from the SOAR website and agrees with the author's
#: GPS value of 2740 m).
SOAR = Site(
    name="SOAR",
    location=EarthLocation.from_geodetic(
        lon=_dms(-1, 70, 44, 1.11),
        lat=_dms(-1, 30, 14, 16.41),
        height=2738.0 * u.m,
    ),
)
