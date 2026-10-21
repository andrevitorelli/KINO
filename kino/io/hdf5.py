"""HDF5 read/write for KINO's own mock data products (README.md Phase 4).

Mirrors ``kino.io.fitsio``'s four function names and round-trip contract
(see that module's docstring for the ``Datacube``/``SlitObservation``
layout rationale) on an HDF5 backend instead of FITS. Array-valued fields
(``flux``, ``variance``, ``mask``, ``spectral_axis``, ``image``) are HDF5
datasets; scalar fields (units, ``spectral_kind``, ``pixel_scale``,
``redshift``, slit geometry) are group attributes; ``Datacube.meta`` is
JSON-encoded into one ``meta_json`` attribute, exactly as ``kino.io.fitsio``
JSON-encodes it into one ``METAJSON`` header card.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import astropy.units as u
import h5py
import numpy as np

from kino.io._meta import dumps_meta, loads_meta
from kino.io.base import Datacube

if TYPE_CHECKING:
    from kino.instruments.slit import SlitObservation

_VARIANCE_NAME = "variance"
_MASK_NAME = "mask"


def _write_cube_group(group: h5py.Group, cube: Datacube) -> None:
    group.create_dataset("flux", data=np.asarray(cube.flux, dtype=np.float64))
    group.create_dataset(
        "spectral_axis", data=np.asarray(cube.spectral_axis.value, dtype=np.float64)
    )
    if cube.variance is not None:
        group.create_dataset(_VARIANCE_NAME, data=np.asarray(cube.variance, dtype=np.float64))
    if cube.mask is not None:
        group.create_dataset(_MASK_NAME, data=np.asarray(cube.mask, dtype=np.uint8))
    group.attrs["flux_unit"] = str(cube.flux_unit)
    group.attrs["spectral_kind"] = cube.spectral_kind
    group.attrs["spectral_axis_unit"] = str(cube.spectral_axis.unit)
    group.attrs["pixel_scale"] = cube.pixel_scale.to_value(u.arcsec)
    group.attrs["redshift"] = cube.redshift
    group.attrs["meta_json"] = dumps_meta(cube.meta)


def _read_cube_group(group: h5py.Group) -> Datacube:
    has_variance = _VARIANCE_NAME in group
    variance = np.asarray(group[_VARIANCE_NAME][()], dtype=np.float64) if has_variance else None
    mask = np.asarray(group[_MASK_NAME][()], dtype=bool) if _MASK_NAME in group else None
    meta = loads_meta(group.attrs["meta_json"]) if "meta_json" in group.attrs else {}
    return Datacube(
        flux=np.asarray(group["flux"][()], dtype=np.float64),
        spectral_axis=np.asarray(group["spectral_axis"][()], dtype=np.float64)
        * u.Unit(group.attrs["spectral_axis_unit"]),
        spectral_kind=group.attrs["spectral_kind"],
        pixel_scale=float(group.attrs["pixel_scale"]) * u.arcsec,
        redshift=float(group.attrs["redshift"]),
        flux_unit=u.Unit(group.attrs["flux_unit"]),
        variance=variance,
        mask=mask,
        meta=meta,
    )


def write_datacube(cube: Datacube, path: str | Path, extname: str = "FLUX") -> None:
    """Write ``cube`` to ``path`` as an HDF5 file (see module docstring).

    ``extname`` names the top-level group holding the cube's datasets/attrs
    (mirroring :func:`kino.io.fitsio.write_datacube`'s ``extname``).
    """
    with h5py.File(path, "w") as f:
        _write_cube_group(f.create_group(extname), cube)


def read_datacube(path: str | Path, extname: str = "FLUX") -> Datacube:
    """Inverse of :func:`write_datacube`."""
    with h5py.File(path, "r") as f:
        return _read_cube_group(f[extname])


def write_slit_observation(obs: SlitObservation, path: str | Path) -> None:
    """Write ``obs`` to ``path`` as an HDF5 file (see module docstring)."""
    with h5py.File(path, "w") as f:
        _write_cube_group(f.create_group("SPECTRUM"), obs.spectrum)
        image_group = f.create_group("IMAGE")
        image_group.create_dataset("image", data=np.asarray(obs.image, dtype=np.float64))
        image_group.attrs["image_pixel_scale"] = obs.image_pixel_scale.to_value(u.arcsec)
        image_group.attrs["slit_width"] = obs.slit_width.to_value(u.arcsec)
        image_group.attrs["slit_length"] = obs.slit_length.to_value(u.arcsec)
        image_group.attrs["slit_pa"] = obs.slit_pa
        image_group.attrs["slit_offset_x"] = obs.slit_offset[0].to_value(u.arcsec)
        image_group.attrs["slit_offset_y"] = obs.slit_offset[1].to_value(u.arcsec)


def read_slit_observation(path: str | Path) -> SlitObservation:
    """Inverse of :func:`write_slit_observation`."""
    from kino.instruments.slit import SlitObservation

    with h5py.File(path, "r") as f:
        spectrum = _read_cube_group(f["SPECTRUM"])
        ig = f["IMAGE"]
        return SlitObservation(
            spectrum=spectrum,
            image=np.asarray(ig["image"][()], dtype=np.float64),
            image_pixel_scale=float(ig.attrs["image_pixel_scale"]) * u.arcsec,
            slit_width=float(ig.attrs["slit_width"]) * u.arcsec,
            slit_length=float(ig.attrs["slit_length"]) * u.arcsec,
            slit_pa=float(ig.attrs["slit_pa"]),
            slit_offset=(
                float(ig.attrs["slit_offset_x"]) * u.arcsec,
                float(ig.attrs["slit_offset_y"]) * u.arcsec,
            ),
        )
