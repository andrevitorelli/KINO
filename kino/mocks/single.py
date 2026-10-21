"""Assemble one galaxy + shear + instrument into a mock data product.

Implements README.md Phase 4's "mocks layer" bullet: the galaxy -> shear ->
instrument assembly that every forecast/notebook/example already does
inline (``apply_shear`` then ``instrument.observe()``), now a single,
tested entry point so :mod:`kino.mocks.catalogue` and
``examples/simulate_one_galaxy.py`` share it rather than duplicating it.

Its only value-add over calling ``apply_shear``/``observe`` directly is
attaching **provenance** into the result's ``Datacube.meta`` (a field that
already existed on :class:`~kino.io.base.Datacube` but nothing previously
populated): every :class:`~kino.galaxy.DiscGalaxy` field, the shear, the
line flux/wavelength, and any caller-supplied ``meta``.
"""

from __future__ import annotations

from dataclasses import asdict

import astropy.units as u
import numpy as np

from kino.galaxy import DiscGalaxy
from kino.instruments.base import Instrument
from kino.instruments.slit import SlitObservation
from kino.io.base import Datacube
from kino.lensing import apply_shear


def simulate_observation(
    galaxy: DiscGalaxy,
    g1: float,
    g2: float,
    instrument: Instrument,
    *,
    line_flux: u.Quantity,
    line_wavelength: u.Quantity,
    rng: np.random.Generator | None = None,
    meta: dict | None = None,
) -> Datacube | SlitObservation:
    """Shear ``galaxy`` and render it through ``instrument``.

    Parameters
    ----------
    galaxy : DiscGalaxy
        The intrinsic (unlensed) disc.
    g1, g2 : float
        Reduced shear applied via :func:`kino.lensing.apply_shear`.
    instrument : Instrument
        An :class:`~kino.instruments.ifu.IFU` or
        :class:`~kino.instruments.slit.SlitSpectrograph`.
    line_flux : Quantity
        Intrinsic (unlensed) line flux, e.g. ``erg / s / cm**2``.
    line_wavelength : Quantity
        Rest-frame line wavelength.
    rng : numpy.random.Generator or None, optional
        ``None`` (default) returns the noiseless expected-count result;
        pass a seeded generator for one noisy realisation.
    meta : dict or None, optional
        Extra provenance merged into the result's ``Datacube.meta`` (e.g.
        an instrument config path or a batch seed) -- galaxy/shear/line
        fields are always included and take precedence on key collision.

    Returns
    -------
    Datacube or SlitObservation
        Whatever ``instrument.observe()`` returns. For a
        ``SlitObservation``, provenance is written to
        ``result.spectrum.meta`` (the only ``Datacube`` it carries).
    """
    scene = apply_shear(galaxy, g1, g2)
    result = instrument.observe(
        scene, line_flux=line_flux, line_wavelength=line_wavelength, rng=rng
    )

    provenance = dict(meta or {})
    provenance.update(asdict(galaxy))
    provenance["g1"] = g1
    provenance["g2"] = g2
    provenance["line_flux"] = str(line_flux)
    provenance["line_wavelength"] = str(line_wavelength)

    cube = result.spectrum if isinstance(result, SlitObservation) else result
    cube.meta.update(provenance)
    return result
