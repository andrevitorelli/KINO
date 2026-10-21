"""Shared JSON (de)serialization for ``Datacube.meta``.

``meta`` is documented as free-form (``kino.io.base.Datacube``), and
``IFU.observe``/``SlitSpectrograph.observe`` already populate it with
``astropy.units.Quantity`` values (``line_wavelength_rest``,
``slit_offset``, ...), not just plain scalars. ``kino.io.fitsio`` and
``kino.io.hdf5`` both need to encode this into one JSON string, so the
encoding lives here once rather than twice.

Anything not natively JSON-serializable (``Quantity``, numpy scalars, ...)
is stringified via ``str()``. This is a display/provenance round-trip, not
a type-preserving one: reading a written file back gives you the same
*text* (e.g. ``"0.65628 um"``), not a reconstructed ``Quantity`` object.
"""

from __future__ import annotations

import json
from typing import Any


def _default(obj: Any) -> str:
    return str(obj)


def dumps_meta(meta: dict) -> str:
    """JSON-encode a ``Datacube.meta`` dict, stringifying anything that
    isn't natively JSON-serializable (e.g. ``astropy.units.Quantity``)."""
    return json.dumps(meta, default=_default)


def loads_meta(blob: str) -> dict:
    """Inverse of :func:`dumps_meta` (text round-trip, see module docstring)."""
    return json.loads(blob)
