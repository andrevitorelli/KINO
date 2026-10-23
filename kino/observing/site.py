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

#: Giant Magellan Telescope, Las Campanas Peak (Las Campanas Observatory's
#: highest point, ~2 km south of the main LCO site Mamajek (2013) covers --
#: not the same mountain as SOAR/Cerro Pachon, so this needed its own
#: source). Geodetic 29 02 54.0 S, 70 41 01.0 W, 2514 m: GMTO's own site
#: page (https://giantmagellan.org/location/, retrieved 2026-10-23), quoted
#: as "altitude" -- presumed orthometric (MSL), not explicitly stated as
#: GPS ellipsoidal height the way SOAR's Mamajek (2013) figure is. The
#: geoid-ellipsoid offset in this region is tens of metres at most,
#: negligible for airmass/parallactic-angle/moon-geometry purposes at
#: kino.observing's own stated precision (see this module's siblings'
#: caveats on RA-of-date precession, ~0.3 deg after 25 years).
GMT = Site(
    name="GMT",
    location=EarthLocation.from_geodetic(
        lon=_dms(-1, 70, 41, 1.0),
        lat=_dms(-1, 29, 2, 54.0),
        height=2514.0 * u.m,
    ),
)
