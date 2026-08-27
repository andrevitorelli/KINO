"""Reader for MaNGA / MaNGIA / iMaNGA FITS cubes -> kino.io.base.Datacube.

Covers TEST_DATA.md Tiers B (MaNGIA, iMaNGA — TNG50 mock MaNGA cubes) and C
(real MaNGA/SDSS DR17 cubes): both ship in the MaNGA LOGCUBE/DAP MAPS FITS
layout, so one reader serves all three.

Not implemented yet. Writing this against assumed HDU names/layout without a
real fixture file on hand risks exactly the kind of silently-wrong reader
CLAUDE.md warns about (a convention bug that "produces plausible, wrong
answers"). Per TEST_DATA.md Section 4/6: first fetch one or two real
MaNGA/MaNGIA/iMaNGA cubes into the curated fixture set (kino.io.download),
inspect their actual HDU layout, WCS, spectral axis (wavelength vs. velocity
DAP maps), mask, and variance conventions, *then* implement this reader
against real files and pin the unit/coordinate conversions with tests
(TEST_DATA.md Section 5).
"""

from __future__ import annotations

from pathlib import Path

from kino.io.base import Datacube


def read_manga_cube(path: Path) -> Datacube:
    """Read a MaNGA/MaNGIA/iMaNGA LOGCUBE (+ optional DAP MAPS) FITS file.

    Converts to KINO's conventions on read (arcsec, km/s, PA East-of-North)
    per TEST_DATA.md Section 5 — the conversion belongs here, not downstream.
    """
    raise NotImplementedError(
        "kino.io.manga is not implemented yet -- see TEST_DATA.md Sections 2, 4, and 6 "
        "(fetch a real fixture cube first, then implement against its actual layout)."
    )
