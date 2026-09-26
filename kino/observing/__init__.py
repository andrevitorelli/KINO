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
from kino.observing.night import Night, night_window
from kino.observing.site import SOAR, Site

__all__ = [
    "SOAR",
    "MoonState",
    "Night",
    "Site",
    "TargetTrack",
    "airmass_from_altitude",
    "moon_state",
    "night_window",
    "seeing_at",
    "target_track",
]
