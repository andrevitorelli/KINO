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
from kino.io.base import Datacube
from kino.lensing import apply_shear

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
    data_moments: MomentMaps = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.line_flux_unit is None and self.fixed_line_flux is not None:
            self.line_flux_unit = self.fixed_line_flux.unit
        self.data_moments = compute_moment_maps(
            self.data,
            line_wavelength=self.line_wavelength,
            lsf_sigma=self.lsf_sigma,
            snr_threshold=self.snr_threshold,
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
        galaxy_kwargs = dict(self.fixed_galaxy_params)
        for key, value in params.items():
            if key in GALAXY_FIELDS:
                galaxy_kwargs[key] = value
        galaxy = DiscGalaxy(**galaxy_kwargs)
        g1 = params.get("g1", 0.0)
        g2 = params.get("g2", 0.0)
        return apply_shear(galaxy, g1, g2)

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
