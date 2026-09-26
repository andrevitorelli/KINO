"""Observing conditions: where the target is, when it is observable, and what
the atmosphere and moon are doing.

This layer sits *between* a target and an instrument backend: it turns
"RA/Dec on this night" into airmass, moon state and effective seeing, which
later feed the instrument's throughput, sky background and PSF. It knows
nothing about instruments. See ``README.md`` (observing-conditions bullets).
"""

from kino.observing.conditions import (
    MoonState,
    TargetTrack,
    airmass_from_altitude,
    moon_state,
    seeing_at,
    target_track,
)
from kino.observing.extinction import atmospheric_transmission, extinction_coefficient
from kino.observing.night import Night, night_window
from kino.observing.site import SOAR, Site
from kino.observing.sky import (
    CTIO_DARK_SKY,
    dark_sky_brightness,
    mag_from_nanolamberts,
    moonlight_brightness,
    nanolamberts_from_mag,
    sky_brightness_v,
)
from kino.observing.skymodel import (
    SkyConditions,
    SkyLibrary,
    SkySpectrum,
    ifu_sky_electron_rate,
    sky_electron_rate,
)

__all__ = [
    "CTIO_DARK_SKY",
    "SOAR",
    "MoonState",
    "Night",
    "Site",
    "SkyConditions",
    "SkyLibrary",
    "SkySpectrum",
    "TargetTrack",
    "airmass_from_altitude",
    "atmospheric_transmission",
    "dark_sky_brightness",
    "extinction_coefficient",
    "ifu_sky_electron_rate",
    "mag_from_nanolamberts",
    "moonlight_brightness",
    "moon_state",
    "nanolamberts_from_mag",
    "night_window",
    "seeing_at",
    "sky_brightness_v",
    "sky_electron_rate",
    "target_track",
]
