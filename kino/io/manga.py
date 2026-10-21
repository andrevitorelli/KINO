"""Readers for the real MaNGIA (TNG50 mock MaNGA, Sarmiento et al. 2022)
data products, confirmed against live downloads from the TNG-project
portal this session (2026-10-21) -- see ``kino.io.download``'s module
docstring for where/how these are fetched.

**This replaces an earlier version of this module** that assumed a single
flux ``Datacube`` with ``FLUX``/``ERROR``/``MASK``/``GAS`` extensions,
written from the mock-generator code's documentation without a real file
to check against. That assumption was wrong: MaNGIA is released as two
different product types, neither of which matches it:

- **"maps"** (:func:`read_mangia_maps` -> :class:`MangiaMaps`): already
  spatially gridded, but as a stack of **41 named 2-D property maps**
  (``VEL``, ``DISP``, ``MASS``, ``SFR``, metallicity, age, dust, ... -- see
  ``MangiaMaps.maps``'s keys), not a wavelength-indexed flux cube. There is
  no spectral axis in this product at all.
- **"rss"** (:func:`read_mangia_rss` -> :class:`MangiaRSS`): a genuine
  flux-calibrated spectrum per fibre (shape ``(n_wave, n_fiber)``, real
  units, a real wavelength grid) plus each fibre's real sky position --
  but **not pre-gridded into a cube**. :func:`regrid_rss_to_cube` turns
  this into a real :class:`~kino.io.base.Datacube` via a simplified
  Gaussian-kernel ("drizzle"-like) regridding -- see its own docstring for
  exactly what it does and doesn't match about the official MaNGA DRP
  cubing algorithm (Law et al. 2016) or MaNGIA's own generator code
  (``reginasar/TNG_MaNGA_mocks``, GPL-3.0), which this does not call or
  port, only take inspiration from.

File structure, confirmed directly (not from documentation) by downloading
real files for ten TNG50-1 galaxies:

``maps`` (``maps/TNG50-{snap}-{subhalo}-{view}-{n_fibers}.cube_maps.fits``):
    ``INTRINSIC_ASSIGNED`` (primary): shape ``(41, ny, nx)``, one 2-D map
    per ``TYPE<i>`` header key (e.g. ``TYPE2 = VEL``) -- simulation ground
    truth, not a pyPipe3D fit. Header: ``CD1_1``/``CD2_2`` (pixel scale,
    degrees), ``REDSHIFT``, ``PSF`` (fibre PSF FWHM, arcsec), ``FOV``
    (arcsec). Other extensions (``MASS_PER_AGE_INTR``,
    ``MASS_PER_AGE_MET_INTR``, ``LUM_PER_AGE_MET_ASSI``, and the
    pyPipe3D-fit ``SSP``/``SFH``/``indices`` ``_pyPipe3D_REC`` extensions)
    are **not parsed** by :func:`read_mangia_maps` -- out of scope for now,
    add them if a use for them comes up.

``rss`` (``RSS/TNG50-{snap}-{subhalo}-{view}-{n_fibers}.cube_RSS.fits``):
    Primary: flux, shape ``(n_wave=6603, n_fiber)``, ``UNITS`` header says
    ``1E-16 erg/s/cm^2`` (assumed per-Angstrom, i.e. the implicit unit is
    ``1e-16 erg/s/cm^2/Angstrom`` -- the header does not spell out the
    ``/Angstrom``, so this is still an assumption, just a far better
    grounded one than the previous version's). Wavelength grid: ``CRVAL2``/
    ``CDELT2`` along axis 2 (3749-10352 Angstrom, 1 Angstrom steps for
    every file checked). ``Gas emission`` extension: same shape, gas-only
    component (primary is assumed total stars+gas, as real MaNGA cubes
    are). ``x_ifu``/``y_ifu``: real fibre sky positions, arcsec, **not**
    confirmed to follow KINO's East-of-North sky convention -- same caution
    as the previous version of this module: verify before using
    orientation from these. ``Real values``: the same 41
    ``TYPE<i>``-keyed ground-truth properties as ``maps``'
    ``INTRINSIC_ASSIGNED``, here per-fibre (shape ``(41, n_fiber)``)
    instead of per-spaxel -- this is what makes cross-checking
    :func:`regrid_rss_to_cube` possible: regrid ``Real values``' ``VEL``/
    ``DISP`` the same way as ``flux`` and compare against the matching
    ``maps`` file's own gridded ``VEL``/``DISP`` for the *same* galaxy.
    ``N_part_fiber``: simulation-particle count per fibre, not used here.
    ``mass_t``/``mass_full``/``light_full`` (age/metallicity-resolved,
    per-fibre) are **not parsed** -- same scope limit as ``maps``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.io import fits

from kino.io.base import Datacube

_REDSHIFT_KEYWORDS = ("REDSHIFT", "Z", "ZREDSHIFT")

#: Assumed per the "rss" file's UNITS header card -- see module docstring.
DEFAULT_RSS_FLUX_UNIT = 1e-16 * u.erg / u.s / u.cm**2 / u.AA


def _type_maps(header: fits.Header, data: np.ndarray) -> dict[str, np.ndarray]:
    """``{TYPE<i> name: data[i]}`` for every ``TYPE<i>`` header key, i=0.."""
    maps: dict[str, np.ndarray] = {}
    i = 0
    while f"TYPE{i}" in header:
        maps[header[f"TYPE{i}"]] = data[i]
        i += 1
    if i == 0:
        raise ValueError("no TYPE<i> header keys found -- not a recognised MaNGIA cube")
    if i != data.shape[0]:
        raise ValueError(
            f"found {i} TYPE<i> keys but data's leading axis has length {data.shape[0]}"
        )
    return maps


def _redshift(header: fits.Header) -> float:
    for key in _REDSHIFT_KEYWORDS:
        if key in header:
            return float(header[key])
    return 0.0


@dataclass
class MangiaMaps:
    """The ``INTRINSIC_ASSIGNED`` extension of a MaNGIA "maps" FITS file:
    ``n_types`` real, ground-truth 2-D property maps on a common spatial
    grid. See :func:`read_mangia_maps`/the module docstring for exactly
    which maps/metadata are parsed.

    Parameters
    ----------
    maps : dict[str, numpy.ndarray]
        ``{TYPE name: (ny, nx) array}``, e.g. ``maps["VEL"]``,
        ``maps["DISP"]``, ``maps["MASS"]`` -- see a file's own ``TYPE<i>``
        header keys for the full set (41 for every file checked this
        session).
    pixel_scale : Quantity (angle)
    redshift : float
    psf_fwhm : Quantity (angle)
        The fibre PSF FWHM assumed when this mock was generated.
    fov : Quantity (angle)
    meta : dict
        Free-form provenance (source, path, the file's own header).
    """

    maps: dict[str, np.ndarray]
    pixel_scale: u.Quantity
    redshift: float
    psf_fwhm: u.Quantity
    fov: u.Quantity
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class MangiaRSS:
    """The raw, un-gridded contents of a MaNGIA "rss" FITS file: a
    flux-calibrated spectrum per fibre plus each fibre's real sky
    position. See :func:`read_mangia_rss`/the module docstring for exactly
    which extensions are parsed, and :func:`regrid_rss_to_cube` to turn
    this into a :class:`~kino.io.base.Datacube`.

    Parameters
    ----------
    wavelength : Quantity (wavelength), shape (n_wave,)
    flux, flux_gas : numpy.ndarray, shape (n_wave, n_fiber)
        Total (stars+gas, assumed) and gas-only flux, in ``flux_unit``.
        ``flux_gas`` is ``None`` if the file has no ``Gas emission``
        extension.
    flux_unit : Unit
    x_fiber, y_fiber : Quantity (angle), shape (n_fiber,)
        Real fibre sky positions -- East/North convention unconfirmed,
        see module docstring.
    fiber_maps : dict[str, numpy.ndarray]
        ``{TYPE name: (n_fiber,) array}``, the per-fibre analogue of
        ``MangiaMaps.maps`` -- the same 41 ground-truth properties,
        not yet gridded.
    psf_fwhm : Quantity (angle)
    redshift : float
    meta : dict
    """

    wavelength: u.Quantity
    flux: np.ndarray
    flux_gas: np.ndarray | None
    flux_unit: u.UnitBase
    x_fiber: u.Quantity
    y_fiber: u.Quantity
    fiber_maps: dict[str, np.ndarray]
    psf_fwhm: u.Quantity
    redshift: float
    meta: dict[str, Any] = field(default_factory=dict)


def read_mangia_maps(path: str | Path) -> MangiaMaps:
    """Read a MaNGIA "maps" FITS file (see module docstring) into a
    :class:`MangiaMaps`."""
    with fits.open(path) as hdul:
        primary = hdul[0]
        header = primary.header
        data = np.asarray(primary.data, dtype=float)
        if data.ndim != 3:
            raise ValueError(f"expected a 3-D INTRINSIC_ASSIGNED HDU, got shape {data.shape}")
        maps = _type_maps(header, data)
        pixel_scale = (0.5 * (abs(header["CD1_1"]) + abs(header["CD2_2"])) * u.deg).to(u.arcsec)
        return MangiaMaps(
            maps=maps,
            pixel_scale=pixel_scale,
            redshift=_redshift(header),
            psf_fwhm=header["PSF"] * u.arcsec,
            fov=header["FOV"] * u.arcsec,
            meta={"source": "MaNGIA maps (Sarmiento et al. 2022)", "path": str(path)},
        )


def read_mangia_rss(path: str | Path) -> MangiaRSS:
    """Read a MaNGIA "rss" FITS file (see module docstring) into a
    :class:`MangiaRSS`."""
    with fits.open(path) as hdul:
        primary = hdul[0]
        header = primary.header
        flux = np.asarray(primary.data, dtype=float)  # (n_wave, n_fiber)
        n_wave = flux.shape[0]
        wavelength = (
            header["CRVAL2"] + (np.arange(n_wave) + 1 - header["CRPIX2"]) * header["CDELT2"]
        ) * u.AA

        flux_gas = None
        if "Gas emission" in hdul:
            flux_gas = np.asarray(hdul["Gas emission"].data, dtype=float)

        fiber_maps = _type_maps(
            hdul["Real values"].header, np.asarray(hdul["Real values"].data, dtype=float)
        )

        return MangiaRSS(
            wavelength=wavelength,
            flux=flux,
            flux_gas=flux_gas,
            flux_unit=DEFAULT_RSS_FLUX_UNIT,
            x_fiber=np.asarray(hdul["x_ifu"].data, dtype=float) * u.arcsec,
            y_fiber=np.asarray(hdul["y_ifu"].data, dtype=float) * u.arcsec,
            fiber_maps=fiber_maps,
            psf_fwhm=header["PSF"] * u.arcsec,
            redshift=_redshift(header),
            meta={"source": "MaNGIA rss (Sarmiento et al. 2022)", "path": str(path)},
        )


def regrid_fibers_to_grid(
    x_fiber: u.Quantity,
    y_fiber: u.Quantity,
    values: np.ndarray,
    *,
    pixel_scale: u.Quantity,
    nx: int,
    ny: int,
    kernel_fwhm: u.Quantity,
    x0: u.Quantity = 0.0 * u.arcsec,
    y0: u.Quantity = 0.0 * u.arcsec,
) -> np.ndarray:
    """Regrid per-fibre ``values`` (last axis = fibre) onto a regular
    ``(..., ny, nx)`` grid by a Gaussian-kernel weighted average --
    ``out[..., y, x] = sum_f w_f(x,y) * values[..., f] / sum_f w_f(x,y)``,
    ``w_f(x,y) = exp(-0.5 * dist((x,y),(x_f,y_f))^2 / sigma^2)``,
    ``sigma = kernel_fwhm / 2.3548``.

    This is a simplified "drizzle"-like regridding, **not** a reproduction
    of the real MaNGA DRP cubing algorithm (Law et al. 2016) or MaNGIA's
    own generator (``reginasar/TNG_MaNGA_mocks``): it ignores each fibre's
    finite aperture (treating it as a point sample) and does not conserve
    flux through any aperture-overlap accounting. **It also has no fibre-
    bundle footprint mask** -- the Gaussian kernel has no hard cutoff, so
    output pixels outside the real fibre coverage still get a (smoothly
    extrapolated, increasingly unreliable) value rather than being flagged
    invalid; mask the output to the fibre bundle's own footprint yourself
    if you need one (e.g. a matching ``MangiaMaps`` map's own ``!= 0``
    footprint, as ``notebooks/06_mangia_datacube_demo.ipynb`` and
    ``scripts/validate_mangia_regrid.py`` do). Validated empirically (not
    by construction) against MaNGIA's own pre-gridded ``maps`` product for
    the same galaxies, within that footprint -- see ``notebooks/`` and
    ``tests/io/test_manga.py`` for the comparison.

    Parameters
    ----------
    x_fiber, y_fiber : Quantity (angle), shape (n_fiber,)
    values : numpy.ndarray, shape (..., n_fiber)
    pixel_scale : Quantity (angle)
    nx, ny : int
    kernel_fwhm : Quantity (angle)
    x0, y0 : Quantity (angle), optional
        Grid centre, same convention as ``x_fiber``/``y_fiber``.

    Returns
    -------
    numpy.ndarray, shape (..., ny, nx)
    """
    n_fiber = x_fiber.shape[0]
    if values.shape[-1] != n_fiber:
        raise ValueError(
            f"values' last axis ({values.shape[-1]}) must match the number of fibres ({n_fiber})"
        )
    px = pixel_scale.to_value(u.arcsec)
    xf = x_fiber.to_value(u.arcsec)
    yf = y_fiber.to_value(u.arcsec)
    x_axis = (np.arange(nx) - (nx - 1) / 2.0) * px + x0.to_value(u.arcsec)
    y_axis = (np.arange(ny) - (ny - 1) / 2.0) * px + y0.to_value(u.arcsec)
    xx, yy = np.meshgrid(x_axis, y_axis)  # (ny, nx)

    sigma = kernel_fwhm.to_value(u.arcsec) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    dx = xx[..., np.newaxis] - xf[np.newaxis, np.newaxis, :]
    dy = yy[..., np.newaxis] - yf[np.newaxis, np.newaxis, :]
    w = np.exp(-0.5 * (dx**2 + dy**2) / sigma**2)  # (ny, nx, n_fiber)
    w_sum = w.sum(axis=-1)
    w_sum_safe = np.where(w_sum > 0, w_sum, 1.0)

    out = np.einsum("...f,yxf->...yx", values, w) / w_sum_safe
    return np.where(w_sum > 0, out, 0.0)


def regrid_rss_to_cube(
    rss: MangiaRSS,
    *,
    nx: int,
    ny: int,
    pixel_scale: u.Quantity,
    kernel_fwhm: u.Quantity | None = None,
) -> Datacube:
    """Regrid ``rss``'s per-fibre flux spectra into a real
    :class:`~kino.io.base.Datacube`, via :func:`regrid_fibers_to_grid`
    (see its docstring for exactly what the regridding does and doesn't
    model).

    Parameters
    ----------
    rss : MangiaRSS
    nx, ny : int
        Output grid size. MaNGIA's own "maps" product for the same galaxy
        (if you have it) gives a natural choice, for an apples-to-apples
        comparison.
    pixel_scale : Quantity (angle)
    kernel_fwhm : Quantity (angle) or None, optional
        ``None`` (default) uses ``rss.psf_fwhm`` -- the fibre PSF FWHM
        this mock was itself generated with.

    Returns
    -------
    Datacube
        ``flux`` shape ``(n_wave, ny, nx)``, in ``rss.flux_unit``.
    """
    fwhm = kernel_fwhm if kernel_fwhm is not None else rss.psf_fwhm
    flux_grid = regrid_fibers_to_grid(
        rss.x_fiber, rss.y_fiber, rss.flux, pixel_scale=pixel_scale, nx=nx, ny=ny, kernel_fwhm=fwhm
    )
    return Datacube(
        flux=flux_grid,
        spectral_axis=rss.wavelength,
        spectral_kind="wavelength",
        pixel_scale=pixel_scale,
        redshift=rss.redshift,
        flux_unit=rss.flux_unit,
        meta={**rss.meta, "regridded_from": "MangiaRSS", "kernel_fwhm": str(fwhm)},
    )
