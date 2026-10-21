"""FITS read/write for KINO's own mock data products (README.md Phase 4).

Generalizes the ad hoc FITS round-trip that previously lived only inside
``examples/simulate_one_galaxy.py``/``run_inference.py`` into a real,
tested ``kino.io`` API, used by both those scripts and
``kino.mocks.catalogue``. ``kino.io.hdf5`` mirrors the same four function
names for an HDF5-backed alternative.

Datacube layout
----------------
``write_datacube``/``read_datacube`` store a :class:`~kino.io.base.Datacube`
as one ``extname`` image extension (default ``"FLUX"``) plus, if present, a
``VARIANCE`` and a ``MASK`` extension (mask as ``uint8``, ``1`` = masked).
Everything needed to reconstruct the spectral axis, pixel scale, redshift,
and flux unit is round-tripped through header cards on the data extension
itself (``BUNIT``, ``SPECKIND``, ``SPECUNIT``, ``SPECVAL0``, ``SPECSTEP``,
``PIXSCALE``, ``REDSHIFT``); ``Datacube.meta`` (free-form provenance) is
JSON-encoded into one ``METAJSON`` header card -- astropy's ``fits.Header``
folds a long string value across ``CONTINUE`` cards automatically, so no
manual chunking is needed (verified to round-trip exactly for a
representative galaxy-provenance blob).

SlitObservation layout
------------------------
``write_slit_observation``/``read_slit_observation`` store the same
``SPECTRUM``/``VARIANCE``/``MASK`` extensions for ``obs.spectrum``, plus an
``IMAGE`` extension for the companion broadband image, with the slit's
geometry (``obs.slit_width``, ``slit_length``, ``slit_pa``, ``slit_offset``,
``image_pixel_scale``) on the ``IMAGE`` header (``SLITW``, ``SLITLEN``,
``SLITPA``, ``SLITOFFX``, ``SLITOFFY``, ``IMGSCALE``; ``slit_length`` is
derived from ``n_spatial * pixel_scale`` on read via
:class:`~kino.instruments.slit.SlitObservation`'s own field, so it is
written but not required to reconstruct the object).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from astropy.io import fits

from kino.io._meta import dumps_meta, loads_meta
from kino.io.base import Datacube

if TYPE_CHECKING:
    from kino.instruments.slit import SlitObservation

_VARIANCE_NAME = "VARIANCE"
_MASK_NAME = "MASK"


def _cube_to_hdus(cube: Datacube, extname: str) -> list[fits.hdu.base._BaseHDU]:
    hdu = fits.ImageHDU(data=np.asarray(cube.flux, dtype=np.float64), name=extname)
    h = hdu.header
    h["BUNIT"] = str(cube.flux_unit)
    h["SPECKIND"] = cube.spectral_kind
    h["SPECUNIT"] = str(cube.spectral_axis.unit)
    h["SPECVAL0"] = float(cube.spectral_axis[0].value)
    if len(cube.spectral_axis) > 1:
        h["SPECSTEP"] = float((cube.spectral_axis[1] - cube.spectral_axis[0]).value)
    h["PIXSCALE"] = cube.pixel_scale.to_value(u.arcsec)
    h["REDSHIFT"] = cube.redshift
    h["METAJSON"] = dumps_meta(cube.meta)
    hdus: list[fits.hdu.base._BaseHDU] = [hdu]
    if cube.variance is not None:
        variance = np.asarray(cube.variance, dtype=np.float64)
        hdus.append(fits.ImageHDU(data=variance, name=_VARIANCE_NAME))
    if cube.mask is not None:
        hdus.append(fits.ImageHDU(data=np.asarray(cube.mask, dtype=np.uint8), name=_MASK_NAME))
    return hdus


def _hdus_to_cube(hdul: fits.HDUList, extname: str) -> Datacube:
    hdu = hdul[extname]
    h = hdu.header
    n_spec = hdu.data.shape[0]
    spec_step = h.get("SPECSTEP", 0.0)
    spectral_axis = (h["SPECVAL0"] + spec_step * np.arange(n_spec)) * u.Unit(h["SPECUNIT"])
    variance = None
    if _VARIANCE_NAME in hdul:
        variance = np.asarray(hdul[_VARIANCE_NAME].data, dtype=np.float64)
    mask = None
    if _MASK_NAME in hdul:
        mask = np.asarray(hdul[_MASK_NAME].data, dtype=bool)
    meta = loads_meta(h["METAJSON"]) if "METAJSON" in h else {}
    return Datacube(
        flux=np.asarray(hdu.data, dtype=np.float64),
        spectral_axis=spectral_axis,
        spectral_kind=h["SPECKIND"],
        pixel_scale=h["PIXSCALE"] * u.arcsec,
        redshift=h["REDSHIFT"],
        flux_unit=u.Unit(h["BUNIT"]),
        variance=variance,
        mask=mask,
        meta=meta,
    )


def write_datacube(cube: Datacube, path: str | Path, extname: str = "FLUX") -> None:
    """Write ``cube`` to ``path`` as a FITS file (see module docstring)."""
    primary = fits.PrimaryHDU()
    hdus = [primary, *_cube_to_hdus(cube, extname)]
    fits.HDUList(hdus).writeto(path, overwrite=True)


def read_datacube(path: str | Path, extname: str = "FLUX") -> Datacube:
    """Inverse of :func:`write_datacube`."""
    with fits.open(path) as hdul:
        return _hdus_to_cube(hdul, extname)


def write_slit_observation(obs: SlitObservation, path: str | Path) -> None:
    """Write ``obs`` to ``path`` as a FITS file (see module docstring)."""
    primary = fits.PrimaryHDU()
    hdus: list[fits.hdu.base._BaseHDU] = [primary, *_cube_to_hdus(obs.spectrum, "SPECTRUM")]
    image_hdu = fits.ImageHDU(data=np.asarray(obs.image, dtype=np.float64), name="IMAGE")
    ih = image_hdu.header
    ih["IMGSCALE"] = obs.image_pixel_scale.to_value(u.arcsec)
    ih["SLITW"] = obs.slit_width.to_value(u.arcsec)
    ih["SLITLEN"] = obs.slit_length.to_value(u.arcsec)
    ih["SLITPA"] = obs.slit_pa
    ih["SLITOFFX"] = obs.slit_offset[0].to_value(u.arcsec)
    ih["SLITOFFY"] = obs.slit_offset[1].to_value(u.arcsec)
    hdus.append(image_hdu)
    fits.HDUList(hdus).writeto(path, overwrite=True)


def read_slit_observation(path: str | Path) -> SlitObservation:
    """Inverse of :func:`write_slit_observation`."""
    from kino.instruments.slit import SlitObservation

    with fits.open(path) as hdul:
        spectrum = _hdus_to_cube(hdul, "SPECTRUM")
        ih = hdul["IMAGE"].header
        return SlitObservation(
            spectrum=spectrum,
            image=np.asarray(hdul["IMAGE"].data, dtype=np.float64),
            image_pixel_scale=ih["IMGSCALE"] * u.arcsec,
            slit_width=ih["SLITW"] * u.arcsec,
            slit_length=ih["SLITLEN"] * u.arcsec,
            slit_pa=ih["SLITPA"],
            slit_offset=(ih["SLITOFFX"] * u.arcsec, ih["SLITOFFY"] * u.arcsec),
        )
