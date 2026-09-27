"""IFU shear-inference likelihood.

Implements ``README.md`` Phase 5's "Likelihood for the IFU case" bullet, via
the "moment maps" option it names explicitly (as opposed to fitting the raw
cube pixel-by-pixel): for each trial parameter set, forward-model a
:class:`~kino.galaxy.DiscGalaxy` + shear + a **fixed, known** IFU instrument
into a noiseless model cube, reduce both the model and the (once, up front)
data cube to flux/Vlos/dispersion moment maps via the exact same
:func:`kino.analysis.moments.compute_moment_maps` call, and compare them
with a Gaussian log-likelihood using the *data*'s own propagated
uncertainties.

Rendering the model through the identical instrument pipeline (not just
evaluating the intrinsic ``DiscGalaxy.vlos`` analytically) is deliberate:
the data's own moment maps carry PSF-driven beam smearing (see
``notebooks/05_ifu_instrument_demo.ipynb`` and ``notebooks/06_mangia_datacube_demo.ipynb``),
and comparing against an un-smeared analytic model would bias every fit by
exactly that amount. Both cubes go through :func:`compute_moment_maps` with
``check_window=False`` on every per-call model evaluation (the window was
already validated once, up front, on the data cube -- re-checking it
thousands of times during sampling is wasted work and log spam; see
``kino.analysis.moments``'s "window width matters" caveat) and with
``snr_threshold=0`` (a noiseless model has no real per-voxel S/N to
threshold on) -- fitting only uses spaxels the *data*'s own mask already
judged trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import astropy.units as u
import numpy as np

from kino.analysis.moments import MomentMaps, compute_moment_maps
from kino.galaxy import DiscGalaxy
from kino.instruments.ifu import IFU
from kino.instruments.slit import SlitSpectrograph
from kino.io.base import Datacube
from kino.lensing import apply_shear

#: Voxel variances (electrons^2) are floored here in the cube likelihood: a voxel whose
#: expected counts and read noise are both ~0 (a far wing of a noiseless test cube) carries no
#: information, and 1/0 would poison the sum.
VARIANCE_FLOOR = 1e-3


def _scene_from_params(fixed_galaxy_params: dict, params: dict):
    """Build a sheared ``DiscGalaxy`` from ``fixed_galaxy_params`` overridden by any
    ``GALAXY_FIELDS`` entry in ``params``, sheared by ``params``'s ``g1``/``g2``
    (0 if absent). Shared by every likelihood in this module -- instrument-agnostic."""
    galaxy_kwargs = dict(fixed_galaxy_params)
    for key, value in params.items():
        if key in GALAXY_FIELDS:
            galaxy_kwargs[key] = value
    galaxy = DiscGalaxy(**galaxy_kwargs)
    return apply_shear(galaxy, params.get("g1", 0.0), params.get("g2", 0.0))


def _resolve_line_flux(
    params: dict, fixed_line_flux: u.Quantity | None, line_flux_unit: u.UnitBase | None
) -> u.Quantity | None:
    """``params["line_flux"]`` (with a unit attached) or ``fixed_line_flux``; ``None``
    if unresolvable (the caller should then return ``-inf`` rather than raising, since
    this runs inside a sampler's per-trial call)."""
    line_flux = params.get("line_flux")
    if line_flux is None:
        if fixed_line_flux is None:
            raise ValueError("line_flux must be free (in params) or fixed_line_flux set")
        return fixed_line_flux
    if isinstance(line_flux, u.Quantity):
        return line_flux
    if line_flux_unit is None:
        raise ValueError(
            "line_flux is free but line_flux_unit (or fixed_line_flux, to infer "
            "the unit from) was not set"
        )
    return line_flux * line_flux_unit


#: DiscGalaxy constructor fields a likelihood may treat as free or fixed.
GALAXY_FIELDS = (
    "r_e",
    "inc",
    "pa",
    "v_flat",
    "z",
    "sersic_n",
    "flux",
    "rotation_curve",
    "r_t",
    "x0",
    "y0",
    "v_sys",
    "sigma0",
    "sigma_grad",
)

#: Parameter names the likelihood handles itself rather than passing to
#: ``DiscGalaxy`` -- the shear, and the physical line flux (kept separate
#: from ``DiscGalaxy.flux``, its own arbitrary-unit shape amplitude, per the
#: "Instrument flux calibration" convention in ``CLAUDE.md``).
NON_GALAXY_FIELDS = ("g1", "g2", "line_flux")


@dataclass
class IFUMomentLikelihood:
    """A callable ``log_likelihood(params: dict) -> float`` for ``nautilus``.

    Parameters
    ----------
    ifu : IFU
        The fixed, known instrument the data was (or, for a mock, will be)
        observed with. Its config is never varied during the fit.
    line_wavelength : Quantity (wavelength)
        Rest-frame line wavelength -- fixed, known (e.g. from a redshift
        estimate and a targeted line).
    data : Datacube
        The observed (or mock) cube to fit against.
    fixed_galaxy_params : dict
        ``DiscGalaxy`` constructor kwargs held fixed for every likelihood
        evaluation (e.g. ``{"sersic_n": 1.0, "r_t": 0.2, "x0": 0.0, "y0": 0.0}``).
        Any key also present in a call's ``params`` dict is overridden by
        the sampled value -- so a parameter is "free" simply by including
        it in the sampler's prior, not by omitting it here.
    fixed_line_flux : Quantity (energy / time / area) or None, optional
        Used when ``"line_flux"`` is not a free parameter (absent from a
        call's ``params``).
    line_flux_unit : Unit or None, optional
        Unit a free ``"line_flux"`` (necessarily a bare float -- samplers
        only ever produce plain numbers) is interpreted in. Defaults to
        ``fixed_line_flux.unit`` if that's set; required if ``line_flux``
        is free and ``fixed_line_flux`` is not.
    lsf_sigma : Quantity (velocity) or None, optional
        Forwarded to :func:`compute_moment_maps` for both data and model --
        should match whatever was used (if anything) to reduce the data.
    snr_threshold : float, optional
        Passed to the *data*'s own one-time moment-map reduction (not the
        per-call model one, see the module docstring); spaxels below this
        are excluded from the fit.
    known_baseline : bool, optional
        Subtract the instrument's *known* sky + dark baseline
        (:meth:`~kino.instruments.ifu.IFU.baseline_electrons`) when reducing both
        the data and every model cube, instead of estimating a per-spaxel median
        from each. Default ``True``: the median is biased and its noise unpropagated
        when the line fills much of the window (see ``compute_moment_maps``), which
        made the reported errors up to ~4.6x too small. Set ``False`` only for data
        whose baseline you cannot model.
    fit_flux : bool, optional
        Include the flux moment in the chi-square alongside Vlos and
        dispersion. Off by default -- flux is the moment most sensitive to
        the overall normalisation (``line_flux``) and least informative
        about ``(g1, g2)``, so many fits will not need it.
    """

    ifu: IFU
    line_wavelength: u.Quantity
    data: Datacube
    fixed_galaxy_params: dict = field(default_factory=dict)
    fixed_line_flux: u.Quantity | None = None
    line_flux_unit: u.UnitBase | None = None
    lsf_sigma: u.Quantity | None = None
    snr_threshold: float = 3.0
    fit_flux: bool = False
    known_baseline: bool = True
    data_moments: MomentMaps = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.line_flux_unit is None and self.fixed_line_flux is not None:
            self.line_flux_unit = self.fixed_line_flux.unit
        self._background = self.ifu.baseline_electrons() if self.known_baseline else True
        self.data_moments = compute_moment_maps(
            self.data,
            line_wavelength=self.line_wavelength,
            lsf_sigma=self.lsf_sigma,
            snr_threshold=self.snr_threshold,
            subtract_background=self._background,
            check_window=True,
        )
        if self.data_moments.mask is None:
            self._valid = np.ones_like(self.data_moments.vlos, dtype=bool)
        else:
            self._valid = ~self.data_moments.mask
        # a spaxel can pass the flux-S/N mask yet still have a NaN vlos_err/
        # dispersion_err (e.g. sigma2_map <= 0 numerically there) -- exclude
        # those too, rather than let one NaN poison the whole chi-square.
        self._valid &= np.isfinite(self.data_moments.vlos_err)
        self._valid &= np.isfinite(self.data_moments.dispersion_err)
        if self.fit_flux:
            self._valid &= np.isfinite(self.data_moments.flux_err)
        if not np.any(self._valid):
            raise ValueError("no spaxel in `data` passes snr_threshold -- nothing to fit against")

    def _build_scene(self, params: dict):
        return _scene_from_params(self.fixed_galaxy_params, params)

    def __call__(self, params: dict) -> float:
        line_flux = params.get("line_flux")
        if line_flux is None:
            if self.fixed_line_flux is None:
                raise ValueError("line_flux must be free (in params) or fixed_line_flux set")
            line_flux = self.fixed_line_flux
        elif not isinstance(line_flux, u.Quantity):
            if self.line_flux_unit is None:
                raise ValueError(
                    "line_flux is free but line_flux_unit (or fixed_line_flux, to infer "
                    "the unit from) was not set"
                )
            line_flux = line_flux * self.line_flux_unit

        try:
            # scene construction can raise too (e.g. |g| >= 1, or an
            # unphysical DiscGalaxy field from a wide/improper prior) -- an
            # unphysical trial point, reject rather than crash the sampler.
            scene = self._build_scene(params)
            model_cube = self.ifu.observe(
                scene, line_flux=line_flux, line_wavelength=self.line_wavelength, rng=None
            )
        except ValueError:
            return -np.inf

        model_moments = compute_moment_maps(
            model_cube,
            line_wavelength=self.line_wavelength,
            lsf_sigma=self.lsf_sigma,
            snr_threshold=0.0,
            subtract_background=self._background,
            check_window=False,
        )

        valid = self._valid
        chi2 = _chi_square(
            self.data_moments.vlos, model_moments.vlos, self.data_moments.vlos_err, valid
        )
        chi2 += _chi_square(
            self.data_moments.dispersion,
            model_moments.dispersion,
            self.data_moments.dispersion_err,
            valid,
        )
        if self.fit_flux:
            chi2 += _chi_square(
                self.data_moments.flux, model_moments.flux, self.data_moments.flux_err, valid
            )

        if not np.isfinite(chi2):
            return -np.inf
        return -0.5 * chi2


def _chi_square(data: np.ndarray, model: np.ndarray, err: np.ndarray, valid: np.ndarray) -> float:
    resid = (data - model)[valid]
    sigma = err[valid]
    return float(np.sum((resid / sigma) ** 2))


@dataclass
class IFUCubeLikelihood:
    """Voxel-level Gaussian log-likelihood of an IFU cube.

    ``log L = -1/2 sum[(d - m)^2 / v + ln v]`` over every voxel.

    Each trial parameter set is forward-modelled through the fixed ``ifu`` into the
    expected cube ``m`` (sky + dark baseline included) and compared with the data
    cube voxel by voxel, with ``v = m + read noise^2`` the expected variance (Poisson
    source + sky + dark plus read noise, Gaussian-approximated). Nothing is
    reduced to moments first, so -- unlike :class:`IFUMomentLikelihood` -- there is no
    estimator noise bias: on many noise realisations the maximum-likelihood shear
    was unbiased (to ~1e-4) and its scatter matched the Cramer-Rao bound
    (:func:`kino.analysis.fisher.shear_fisher`, ``estimator="cube"``) at every S/N
    tested, from per-spaxel S/N ~ 5 to ~100. It also uses the flux image and the
    line wings that the moment maps discard.

    Parameters are as for :class:`IFUMomentLikelihood`: ``g1``/``g2``/``line_flux``
    handled here, any other free parameter a ``DiscGalaxy`` field.

    Parameters
    ----------
    ifu : IFU
        The fixed, known instrument the data was observed with.
    line_wavelength : Quantity (wavelength)
        Rest-frame line wavelength.
    data : Datacube
        The observed (or mock) cube; its ``flux`` must be in the same units as
        ``ifu.observe`` returns (electrons).
    fixed_galaxy_params, fixed_line_flux, line_flux_unit
        As for :class:`IFUMomentLikelihood`.
    """

    ifu: IFU
    line_wavelength: u.Quantity
    data: Datacube
    fixed_galaxy_params: dict = field(default_factory=dict)
    fixed_line_flux: u.Quantity | None = None
    line_flux_unit: u.UnitBase | None = None

    def __post_init__(self) -> None:
        if self.line_flux_unit is None and self.fixed_line_flux is not None:
            self.line_flux_unit = self.fixed_line_flux.unit
        expected = (self.ifu.n_wave, self.ifu.ny, self.ifu.nx)
        if self.data.flux.shape != expected:
            raise ValueError(
                f"data cube shape {self.data.flux.shape} != the instrument's {expected}"
            )
        self._flux = np.asarray(self.data.flux, dtype=float)

    def _build_scene(self, params: dict):
        return _scene_from_params(self.fixed_galaxy_params, params)

    def __call__(self, params: dict) -> float:
        line_flux = _resolve_line_flux(params, self.fixed_line_flux, self.line_flux_unit)
        try:
            scene = self._build_scene(params)
            model = self.ifu.observe(
                scene, line_flux=line_flux, line_wavelength=self.line_wavelength, rng=None
            )
        except ValueError:
            return -np.inf
        var = np.maximum(model.variance, VARIANCE_FLOOR)
        chi2 = float(np.sum((self._flux - model.flux) ** 2 / var + np.log(var)))
        if not np.isfinite(chi2):
            return -np.inf
        return -0.5 * chi2


@dataclass
class SlitCubeLikelihood:
    """Voxel-level Gaussian log-likelihood of a slit spectrograph's dispersed spectrum.

    The :class:`SlitSpectrograph` analogue of :class:`IFUCubeLikelihood`: every trial
    parameter set is forward-modelled through the fixed ``slit`` (the companion
    broadband image is discarded -- only ``SlitObservation.spectrum`` enters the fit)
    and compared voxel by voxel with the same Gaussian likelihood, so slit losses
    (a source wider than the slit, or a misaligned ``slit_pa``/``slit_offset``) are
    inside the model exactly as they are inside the data, rather than a bias.

    Parameters
    ----------
    slit : SlitSpectrograph
        The fixed, known instrument the data was observed with.
    line_wavelength : Quantity (wavelength)
        Rest-frame line wavelength.
    data : Datacube
        ``SlitObservation.spectrum`` (or an equivalent real reduction), shape
        ``(slit.n_wave, slit.n_spatial, 1)``, in electrons.
    fixed_galaxy_params, fixed_line_flux, line_flux_unit
        As for :class:`IFUMomentLikelihood`.
    """

    slit: SlitSpectrograph
    line_wavelength: u.Quantity
    data: Datacube
    fixed_galaxy_params: dict = field(default_factory=dict)
    fixed_line_flux: u.Quantity | None = None
    line_flux_unit: u.UnitBase | None = None

    def __post_init__(self) -> None:
        if self.line_flux_unit is None and self.fixed_line_flux is not None:
            self.line_flux_unit = self.fixed_line_flux.unit
        expected = (self.slit.n_wave, self.slit.n_spatial, 1)
        if self.data.flux.shape != expected:
            raise ValueError(
                f"data cube shape {self.data.flux.shape} != the instrument's {expected}"
            )
        self._flux = np.asarray(self.data.flux, dtype=float)

    def _build_scene(self, params: dict):
        return _scene_from_params(self.fixed_galaxy_params, params)

    def __call__(self, params: dict) -> float:
        line_flux = _resolve_line_flux(params, self.fixed_line_flux, self.line_flux_unit)
        try:
            scene = self._build_scene(params)
            model = self.slit.observe(
                scene, line_flux=line_flux, line_wavelength=self.line_wavelength, rng=None
            ).spectrum
        except ValueError:
            return -np.inf
        var = np.maximum(model.variance, VARIANCE_FLOOR)
        chi2 = float(np.sum((self._flux - model.flux) ** 2 / var + np.log(var)))
        if not np.isfinite(chi2):
            return -np.inf
        return -0.5 * chi2
