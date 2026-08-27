"""Common in-memory container for IFU-style spectral datacubes.

Every KINO data source (``kino.io.kinms``, ``kino.io.manga``, ...) converts
its cube into a :class:`Datacube` on read/generation, so the rest of KINO
(mocks, inference, analysis) consumes one documented representation
regardless of provenance. See ``CLAUDE.md`` for the underlying unit and
coordinate conventions, and ``TEST_DATA.md`` for the validation-data plan
this container supports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import astropy.units as u
import numpy as np


@dataclass
class Datacube:
    """A flux datacube with an explicit spatial and spectral axis.

    Axis order is ``(n_spec, n_y, n_x)`` — spectral axis first, then the two
    spatial axes — matching common FITS/IFU cube layout.

    Parameters
    ----------
    flux : numpy.ndarray, shape (n_spec, n_y, n_x)
        Surface brightness per spaxel per spectral channel, in ``flux_unit``.
    spectral_axis : astropy.units.Quantity, shape (n_spec,)
        Wavelength or velocity of each channel, see ``spectral_kind``.
    spectral_kind : {"wavelength", "velocity"}
        What ``spectral_axis`` represents.
    pixel_scale : astropy.units.Quantity
        Spatial pixel size (angle, e.g. arcsec/pixel); square spaxels assumed.
    redshift : float
        Systemic redshift of the source.
    flux_unit : astropy.units.Unit
        Physical unit of ``flux``.
    variance : numpy.ndarray or None, shape (n_spec, n_y, n_x)
        Per-voxel variance, in ``flux_unit ** 2``. ``None`` if unknown.
    mask : numpy.ndarray of bool or None, shape (n_spec, n_y, n_x)
        ``True`` where the voxel is invalid/masked. ``None`` if unmasked.
    meta : dict
        Free-form provenance: source, object ID, download date, original
        path/URL, citation (see ``TEST_DATA.md`` Section 5).
    """

    flux: np.ndarray
    spectral_axis: u.Quantity
    spectral_kind: str
    pixel_scale: u.Quantity
    redshift: float
    flux_unit: u.UnitBase
    variance: np.ndarray | None = None
    mask: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.flux.ndim != 3:
            raise ValueError(f"flux must be 3-D (n_spec, n_y, n_x), got shape {self.flux.shape}")
        n_spec = self.flux.shape[0]
        if self.spectral_axis.shape != (n_spec,):
            raise ValueError(
                f"spectral_axis shape {self.spectral_axis.shape} does not match "
                f"flux's spectral axis length {n_spec}"
            )
        if self.spectral_kind not in ("wavelength", "velocity"):
            raise ValueError(
                f"spectral_kind must be 'wavelength' or 'velocity', got {self.spectral_kind!r}"
            )
        expected_unit = u.km / u.s if self.spectral_kind == "velocity" else u.m
        if not self.spectral_axis.unit.is_equivalent(expected_unit):
            raise u.UnitsError(
                f"spectral_axis unit {self.spectral_axis.unit} is not compatible with "
                f"spectral_kind={self.spectral_kind!r}"
            )
        if not self.pixel_scale.isscalar or not self.pixel_scale.unit.is_equivalent(u.arcsec):
            raise u.UnitsError("pixel_scale must be a scalar angle (e.g. arcsec)")
        if self.variance is not None and self.variance.shape != self.flux.shape:
            raise ValueError("variance must have the same shape as flux")
        if self.mask is not None and self.mask.shape != self.flux.shape:
            raise ValueError("mask must have the same shape as flux")

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(n_spec, n_y, n_x)``."""
        return self.flux.shape
