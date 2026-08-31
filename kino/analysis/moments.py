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

Window width matters -- a real usage caveat
----------------------------------------------
``dispersion`` (a second moment, weighted by ``(v - vlos)**2``) is far more
sensitive than ``flux``/``vlos`` to how much of ``cube.spectral_axis`` is
*pure noise far from the line*: a channel's noise, however small, gets
multiplied by a potentially huge ``dv**2`` there, so summed over many such
channels it can dominate the numerator and drive ``dispersion`` to garbage
(commonly clipped to exactly 0) even on an otherwise high-S/N line. This is
a real property of the naive second-moment estimator, not a bug -- keep
``cube``'s spectral window to a few times the line's expected total extent
(Doppler range across the field, plus a handful of line widths of margin)
before calling this function, rather than handing it, say, a full MaNGA
wavelength range. ``flux`` and ``vlos`` degrade far more gracefully with a
too-wide window; ``dispersion`` is the one to watch.

The failure isn't one-sided, either: a window that's too *narrow* truncates
the line's own wings before they reach the background, biasing
``dispersion`` low -- and a biased-low variance estimate is just as easily
pushed negative (and clipped to 0) by noise as a noise-dominated wide one
is. There is a sweet spot, not just an upper bound; see
``notebooks/07_moment_window_width_caveat.ipynb`` for a quantified sweep
across window widths (collapse rate, recovered dispersion, and ``vlos``
scatter for comparison) that maps it out on a synthetic line.

Two mitigations follow directly from that finding: :func:`crop_to_line`
makes the safe path (crop to near the line before measuring moments) the
easy one, and :func:`compute_moment_maps` itself runs a cheap, cube-level
diagnostic -- via ``check_window`` -- that warns when the spectral window
looks implausibly wide relative to where the flux actually is, *before*
handing back a `dispersion` map that may be silently full of clipped
zeros. The diagnostic only catches the too-wide failure mode (the one that
matches "hand it a full instrument bandpass by accident," the concrete
trap in this repo's own fiducial configs -- see ``CLAUDE.md``); it does not
detect a too-narrow window, which has no comparably robust, false-positive-free
signal to check for automatically.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.constants import c as C_LIGHT

from kino.io.base import Datacube

#: Below this fraction of "effectively contributing" channels (see
#: `_effective_channel_fraction`), `compute_moment_maps` warns that the
#: window may be too wide. Calibrated empirically against
#: `notebooks/07_moment_window_width_caveat.ipynb`'s sweep: comfortably
#: separates its "sweet spot" windows (fraction >= 0.12, 0% dispersion
#: collapse) from windows already well into collapse (fraction <= 0.09,
#: >85% collapse). Not a universal constant -- a heuristic guideline, per
#: this module's own "not a bug" framing of the underlying effect.
_MIN_EFFECTIVE_CHANNEL_FRACTION = 0.1


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


def crop_to_line(cube: Datacube, center: u.Quantity, half_width: u.Quantity) -> Datacube:
    """Crop ``cube``'s spectral axis to ``[center - half_width, center + half_width]``.

    The safe-path counterpart to this module's window-width caveat: size
    ``half_width`` to a handful of the line's expected total extent (Doppler
    range across the field, plus a few line widths of margin) and crop
    before calling :func:`compute_moment_maps`, rather than handing it an
    instrument's full bandpass.

    Parameters
    ----------
    cube : Datacube
    center : Quantity
        Line centre, in the same domain as ``cube.spectral_axis`` (a
        wavelength for ``spectral_kind="wavelength"``, a velocity for
        ``"velocity"``).
    half_width : Quantity
        Half-width of the retained window, same domain as ``center``.

    Returns
    -------
    Datacube
        A new ``Datacube`` with ``flux``/``variance``/``mask`` sliced to
        match the cropped ``spectral_axis``. Assumes ``cube.spectral_axis``
        is monotonically increasing (true of every cube produced elsewhere
        in kino).
    """
    axis = cube.spectral_axis
    lo, hi = center - half_width, center + half_width
    in_window = (axis >= lo) & (axis <= hi)
    if not np.any(in_window):
        raise ValueError(
            f"no channels of cube.spectral_axis (range {axis.min()}-{axis.max()}) "
            f"fall within [{lo}, {hi}]"
        )
    idx = np.flatnonzero(in_window)
    sl = slice(idx[0], idx[-1] + 1)
    return Datacube(
        flux=cube.flux[sl],
        spectral_axis=axis[sl],
        spectral_kind=cube.spectral_kind,
        pixel_scale=cube.pixel_scale,
        redshift=cube.redshift,
        flux_unit=cube.flux_unit,
        variance=cube.variance[sl] if cube.variance is not None else None,
        mask=cube.mask[sl] if cube.mask is not None else None,
        meta=dict(cube.meta),
    )


def _effective_channel_fraction(flux_cube: np.ndarray) -> tuple[float, float, int]:
    """Fraction of channels "effectively" carrying the (stacked) spectrum's flux.

    Stacks ``flux_cube`` over the spatial axes, subtracts a median
    background, and computes the participation-ratio effective channel
    count ``n_eff = (sum w)**2 / sum(w**2)`` -- large when a few channels
    hold most of the (background-subtracted) flux, small when it's spread
    thin (either a genuine broad/weak line, or -- the failure mode this
    guards against -- a narrow line lost in a wide window of noise).
    Returns ``(fraction, n_eff, n_total)``.
    """
    stacked = flux_cube.sum(axis=(1, 2))
    n_total = stacked.shape[0]
    w = stacked - np.median(stacked)
    denom = np.sum(w**2)
    n_eff = (np.sum(w) ** 2 / denom) if denom > 0 else 0.0
    return n_eff / n_total, n_eff, n_total


def compute_moment_maps(
    cube: Datacube,
    *,
    line_wavelength: u.Quantity | None = None,
    redshift: float | None = None,
    lsf_sigma: u.Quantity | None = None,
    snr_threshold: float = 3.0,
    subtract_background: bool = True,
    check_window: bool = True,
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
    check_window : bool, optional
        Run the cheap, cube-level too-wide-window diagnostic described in
        this module's docstring and emit a ``UserWarning`` if triggered.
        Set ``False`` to silence it -- e.g. when deliberately exploring a
        bad window (as `notebooks/07_moment_window_width_caveat.ipynb`
        does), or once you've already sized the window with
        :func:`crop_to_line` and don't need to be told twice.

    Returns
    -------
    MomentMaps
    """
    if check_window:
        fraction, n_eff, n_total = _effective_channel_fraction(cube.flux)
        if fraction < _MIN_EFFECTIVE_CHANNEL_FRACTION:
            warnings.warn(
                f"compute_moment_maps: only {fraction:.1%} of the {n_total} spectral "
                f"channels ({n_eff:.0f} effective) contribute to the stacked spectrum's "
                "flux -- the spectral window may be too wide relative to the line. "
                "dispersion is especially vulnerable to noise from the rest (see this "
                "module's 'window width matters' docstring section). Consider "
                "kino.analysis.moments.crop_to_line() first, or pass check_window=False "
                "to silence this.",
                UserWarning,
                stacklevel=2,
            )

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
