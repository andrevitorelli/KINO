"""Flux/Vlos/dispersion moment maps, with noise, from any ``Datacube``.

Implements the "extract moment maps (flux, Vlos, sigma) with noise" item of
``README.md`` Phase 3. Deliberately generic over *any* :class:`~kino.io.base.Datacube`
(an ``IFU``/``SlitSpectrograph`` mock, a KinMS cube, or real survey data) --
this is a measurement *on* a cube, not part of the forward model, so it does
not depend on ``kino.instruments`` at all.

Method
------
Standard flux-weighted spectral moments per spaxel: moment 0 (flux) is the
background-subtracted sum over the spectral axis; moment 1 (``Vlos``) is the
flux-weighted mean velocity; moment 2 (``dispersion``) is the flux-weighted
second moment about that mean, optionally corrected for instrumental
broadening (``lsf_sigma``) in quadrature. See ``CLAUDE.md``'s ``Vlos``
sign convention -- positive is redshifted/receding, which this module
inherits directly from ``Datacube.spectral_axis``.

Per-voxel ``variance`` (if present on the cube) is propagated to
``flux``/``vlos``/``dispersion`` uncertainties via the standard delta-method
(first-order) approximation for a ratio of correlated sums -- exact for
linear moments (flux), a linearisation for the nonlinear ones (Vlos,
dispersion). Spaxels below ``snr_threshold`` (flux significance) have their
``vlos``/``dispersion`` masked to NaN -- kinematics are not reliably
measurable there, even though a (noisy) flux value still exists.

This is *not* a line-fitting pipeline (no continuum model, no line-profile
fit) -- it is the same moment-based measurement ``kino.lensing.moments``
uses for spatial shape, applied along the spectral axis instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.constants import c as C_LIGHT

from kino.io.base import Datacube


@dataclass
class MomentMaps:
    """Per-spaxel flux, Vlos, and velocity-dispersion maps, with uncertainties.

    All spatial arrays have shape ``(n_y, n_x)`` (see ``Datacube``). ``flux``
    is in the cube's own ``flux_unit``; ``vlos`` and ``dispersion`` are in
    km/s. ``*_err``/``snr``/``mask`` are ``None`` if the input cube had no
    ``variance``. ``mask`` is ``True`` where ``vlos``/``dispersion`` are
    masked (below ``snr_threshold`` or otherwise ill-defined) -- ``flux`` is
    never masked.
    """

    flux: np.ndarray
    vlos: np.ndarray
    dispersion: np.ndarray
    flux_err: np.ndarray | None
    vlos_err: np.ndarray | None
    dispersion_err: np.ndarray | None
    snr: np.ndarray | None
    mask: np.ndarray | None
    flux_unit: u.UnitBase


def compute_moment_maps(
    cube: Datacube,
    *,
    line_wavelength: u.Quantity | None = None,
    redshift: float | None = None,
    lsf_sigma: u.Quantity | None = None,
    snr_threshold: float = 3.0,
    subtract_background: bool = True,
) -> MomentMaps:
    """Flux/Vlos/dispersion moment maps from ``cube``.

    Parameters
    ----------
    cube : Datacube
    line_wavelength : Quantity (wavelength), optional
        Rest-frame line wavelength -- required if ``cube.spectral_kind ==
        "wavelength"`` (ignored, and unnecessary, for ``"velocity"``, where
        ``cube.spectral_axis`` already *is* a velocity).
    redshift : float or None, optional
        Overrides ``cube.redshift`` for the wavelength -> velocity
        conversion; ``None`` uses ``cube.redshift``.
    lsf_sigma : Quantity (velocity) or None, optional
        Instrumental broadening to remove from ``dispersion`` in quadrature
        (``sqrt(max(measured**2 - lsf_sigma**2, 0))``). ``None`` skips the
        correction, i.e. ``dispersion`` is the *observed* (instrument +
        intrinsic) width.
    snr_threshold : float, optional
        Spaxels with flux significance below this are masked (``vlos``,
        ``dispersion`` -> NaN). Only applied if ``cube.variance`` is set.
    subtract_background : bool, optional
        Subtract a per-spaxel background estimated as the median flux
        across the spectral axis before computing moments (appropriate for
        a roughly flat sky/dark pedestal occupying most of the band, e.g.
        ``kino.instruments.ifu.IFU``'s output; turn off for an
        already-background-subtracted cube).

    Returns
    -------
    MomentMaps
    """
    if cube.spectral_kind == "velocity":
        velocity = cube.spectral_axis.to(u.km / u.s)
    else:
        if line_wavelength is None:
            raise ValueError("line_wavelength is required for a wavelength-kind Datacube")
        z = cube.redshift if redshift is None else redshift
        line_center = line_wavelength * (1.0 + z)
        velocity = (C_LIGHT * (cube.spectral_axis / line_center - 1.0)).to(u.km / u.s)
    v = velocity.to_value(u.km / u.s)[:, np.newaxis, np.newaxis]

    flux_cube = cube.flux
    background = np.median(flux_cube, axis=0, keepdims=True) if subtract_background else 0.0
    signal = flux_cube - background

    with np.errstate(invalid="ignore", divide="ignore"):
        flux_map = signal.sum(axis=0)
        vlos_map = (signal * v).sum(axis=0) / flux_map
        dv = v - vlos_map[np.newaxis, :, :]
        sigma2_map = (signal * dv**2).sum(axis=0) / flux_map
    dispersion_map = np.sqrt(np.clip(sigma2_map, 0.0, None))

    if lsf_sigma is not None:
        lsf_kms = lsf_sigma.to_value(u.km / u.s)
        dispersion_map = np.sqrt(np.clip(dispersion_map**2 - lsf_kms**2, 0.0, None))

    flux_err = vlos_err = dispersion_err = snr = mask = None
    if cube.variance is not None:
        var = cube.variance
        var_d = var.sum(axis=0)
        flux_err = np.sqrt(var_d)

        var_n = (v**2 * var).sum(axis=0)
        cov_nd = (v * var).sum(axis=0)
        n_num = (signal * v).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            var_vlos = (
                var_n / flux_map**2
                - 2.0 * n_num * cov_nd / flux_map**3
                + n_num**2 * var_d / flux_map**4
            )
        vlos_err = np.sqrt(np.clip(var_vlos, 0.0, None))

        var_m = (dv**4 * var).sum(axis=0)
        cov_md = (dv**2 * var).sum(axis=0)
        m_num = (signal * dv**2).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            var_sigma2 = (
                var_m / flux_map**2
                - 2.0 * m_num * cov_md / flux_map**3
                + m_num**2 * var_d / flux_map**4
            )
            var_sigma = np.where(sigma2_map > 0, var_sigma2 / (4.0 * sigma2_map), np.nan)
        dispersion_err = np.sqrt(np.clip(var_sigma, 0.0, None))

        with np.errstate(invalid="ignore", divide="ignore"):
            snr = flux_map / flux_err
        mask = ~(snr >= snr_threshold)  # NaN-safe: NaN snr also masks out
        vlos_map = np.where(mask, np.nan, vlos_map)
        dispersion_map = np.where(mask, np.nan, dispersion_map)
        vlos_err = np.where(mask, np.nan, vlos_err)
        dispersion_err = np.where(mask, np.nan, dispersion_err)

    return MomentMaps(
        flux=flux_map,
        vlos=vlos_map,
        dispersion=dispersion_map,
        flux_err=flux_err,
        vlos_err=vlos_err,
        dispersion_err=dispersion_err,
        snr=snr,
        mask=mask,
        flux_unit=cube.flux_unit,
    )
