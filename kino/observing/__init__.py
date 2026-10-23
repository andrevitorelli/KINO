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
    ecliptic_from_sun,
    moon_state,
    seeing_at,
    target_track,
)
from kino.observing.extinction import atmospheric_transmission, extinction_coefficient
from kino.observing.extinction_nir import (
    NIR_EXTINCTION_BANDS,
    NIR_EXTINCTION_FROGEL,
    nir_atmospheric_transmission,
    nir_extinction_coefficient,
)
from kino.observing.night import Night, night_window
from kino.observing.site import GMT, SOAR, Site
from kino.observing.sky import (
    CTIO_DARK_SKY,
    dark_sky_brightness,
    la_serena_worst_case_factor,
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
    slit_sky_electron_rate,
)
from kino.observing.solar import NOAA_SWPC_F107_PREDICTION, solar_flux

__all__ = [
    "NOAA_SWPC_F107_PREDICTION",
    "CTIO_DARK_SKY",
    "GMT",
    "NIR_EXTINCTION_BANDS",
    "NIR_EXTINCTION_FROGEL",
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
    "ecliptic_from_sun",
    "extinction_coefficient",
    "ifu_sky_electron_rate",
    "la_serena_worst_case_factor",
    "mag_from_nanolamberts",
    "moonlight_brightness",
    "moon_state",
    "nanolamberts_from_mag",
    "night_window",
    "nir_atmospheric_transmission",
    "nir_extinction_coefficient",
    "seeing_at",
    "solar_flux",
    "sky_brightness_v",
    "sky_electron_rate",
    "slit_sky_electron_rate",
    "target_track",
]
