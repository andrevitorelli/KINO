"""Target-level forecast: shear S/N as a function of seeing, for a given night.

``forecast_ifu`` and ``forecast_slit`` are the entry points that tie the
observing-conditions layer (:mod:`kino.observing`) to an instrument backend
(an :class:`~kino.instruments.ifu.IFU` or a
:class:`~kino.instruments.slit.SlitSpectrograph`) and the fast Fisher estimate
(:mod:`kino.analysis.fisher`). Given

* an instrument (its exposure, throughput, detector, PSF machinery),
* a target: sky position, a :class:`~kino.galaxy.DiscGalaxy` (geometry and
  redshift) and the line flux measured from its spectrum
  (:func:`kino.io.spectrum.measure_line_flux`),
* a night, and the shear you expect,

each returns, at times through the night and for each seeing you ask about, the
airmass, the sky, the effective seeing, the line S/N and the shear precision and
S/N -- all computed with the sky, moon, extinction and seeing appropriate to
that moment. Both are thin wrappers around ``_forecast`` (below), which does the
actual work and is agnostic to which instrument type it was given.

What happens at each (time, seeing) point:

1. airmass, moon phase/altitude/separation and ecliptic position from the
   target's track (``kino.observing``);
2. a sky spectrum from the offline SkyCalc library, converted to detected
   electrons per channel on the instrument's own collecting area/throughput
   (per spaxel for an IFU, per along-slit pixel for a slit);
3. the line flux dimmed by CTIO atmospheric extinction (``include_extinction``);
4. the quoted (500 nm, zenith) seeing converted to the observed wavelength and
   airmass (Kolmogorov scaling), set as the instrument's seeing PSF;
5. :func:`~kino.analysis.fisher.shear_fisher` on the noiseless expected cube
   (the ``"cube"`` estimator works for either instrument; slit losses are
   inside the model exactly as they are inside a real slit observation).

Assumptions you must be comfortable with (each is also in ``CLAUDE.md``):

* the instrument's throughput curve is *end to end excluding the atmosphere*;
  extinction is applied separately. If your throughput already includes the
  atmosphere (a standard-star efficiency often does), pass
  ``include_extinction=False``;
* the sky model is Paranal's, not the site's; airglow is a whole-night mean;
* ``line_flux`` is the *whole galaxy's intrinsic* flux (aperture-corrected,
  magnification neglected -- a few per cent for weak lensing);
* only the emission line constrains the shear -- no continuum;
* the spectral window is cropped to ``window_kms`` around the line, keeping the
  instrument's native dispersion, purely so the many renders a forecast needs
  stay fast -- not a physical limitation (see ``narrowed_ifu``/``narrowed_slit``);
* the primary, default forecast (``intrinsic_sigma=None``) is the **idealised
  statistical** bound: real-galaxy floor calibration (non-circular motions,
  disc thickness, ...) is out of scope for now (2026-09-27 direction) and left
  as an optional sensitivity comparison, not a required input.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time

from kino.analysis.fisher import LOW_SNR_WARNING, FisherResult, shear_fisher
from kino.galaxy import DiscGalaxy
from kino.instruments.base import PSF
from kino.io.spectrum import LINES
from kino.observing.conditions import (
    ecliptic_from_sun,
    moon_state,
    seeing_at,
    target_track,
)
from kino.observing.extinction import atmospheric_transmission
from kino.observing.night import Night
from kino.observing.site import SOAR, Site
from kino.observing.skymodel import (
    SkyConditions,
    SkyLibrary,
    ifu_sky_electron_rate,
    slit_sky_electron_rate,
)

C_KMS = 299792.458


@dataclass(frozen=True)
class ForecastTarget:
    """What the forecast needs to know about a target.

    Attributes
    ----------
    coord : SkyCoord
    galaxy : DiscGalaxy
        Intrinsic geometry; ``galaxy.z`` is the redshift.
    line_flux : Quantity (erg / s / cm^2)
        Whole-galaxy intrinsic line flux (see the module docstring).
    line : str or Quantity
        A key of :data:`kino.io.spectrum.LINES` or a rest vacuum wavelength.
    name : str
        A label only -- **never put private target names in code or commits**.
    """

    coord: SkyCoord
    galaxy: DiscGalaxy
    line_flux: u.Quantity
    line: str | u.Quantity = "Halpha"
    name: str = ""

    @property
    def rest_wavelength(self) -> u.Quantity:
        rest = LINES[self.line] * u.AA if isinstance(self.line, str) else u.Quantity(self.line)
        return rest.to(u.AA)

    @property
    def observed_wavelength(self) -> u.Quantity:
        return self.rest_wavelength * (1.0 + self.galaxy.z)


@dataclass(frozen=True)
class TargetForecast:
    """Forecast on a grid of ``(time, seeing)``.

    Array shapes: per-time quantities ``(n_t,)``; per-point quantities
    ``(n_t, n_s)``. Points where the target is below ``max_airmass`` or
    undetectable are NaN (``detectable`` marks the latter).

    Attributes
    ----------
    times : Time
    airmass, altitude, moon_illumination, moon_altitude, moon_separation : arrays
    seeing_ref : Quantity (arcsec)
        The quoted seeing (500 nm, zenith) that was swept.
    effective_seeing : Quantity (arcsec)
        Seeing at the observed line wavelength and each time's airmass.
    line_snr, peak_spaxel_snr : arrays
        Whole-field integrated and best-single-spaxel line S/N.
    sigma_g : array (n_t, n_s, 2)
        Marginal ``(sigma_g1, sigma_g2)``; ``sigma_g_fixed`` the same with
        nuisance parameters known.
    shear_snr : array
        ``|g| / sigma_total`` along the expected shear, where ``sigma_total`` adds, in
        quadrature, the statistical marginal error, the ``intrinsic_sigma`` floor (if
        given) and the PSF-mismatch bias (if ``psf_error`` was given).
    shear_snr_stat : array
        The same with the statistical error only.
    psf_bias : array (n_t, n_s, 2)
        Shear bias ``(dg1, dg2)`` from a wrong assumed PSF (NaN if not requested).
    n_spaxels : array
    median_spaxel_snr : array
        Median per-spaxel line S/N of the spaxels used; see :attr:`low_snr`.
    g : tuple[float, float]
        The expected shear.
    """

    times: Time
    airmass: np.ndarray
    altitude: u.Quantity
    moon_illumination: np.ndarray
    moon_altitude: u.Quantity
    moon_separation: u.Quantity
    seeing_ref: u.Quantity
    effective_seeing: u.Quantity
    line_snr: np.ndarray
    peak_spaxel_snr: np.ndarray
    sigma_g: np.ndarray
    sigma_g_fixed: np.ndarray
    shear_snr: np.ndarray
    shear_snr_stat: np.ndarray
    psf_bias: np.ndarray
    n_spaxels: np.ndarray
    median_spaxel_snr: np.ndarray
    detectable: np.ndarray
    g: tuple[float, float]
    observed_wavelength: u.Quantity
    meta: dict

    @property
    def low_snr(self) -> np.ndarray:
        """Moment estimator only: True where the median spaxel S/N is below
        :data:`kino.analysis.fisher.LOW_SNR_WARNING`, where the forecast is optimistic and
        the estimator biased (see ``kino.analysis.fisher``). Always False for the cube
        estimator, which has no such bias."""
        if self.meta.get("estimator") != "moments":
            return np.zeros(self.median_spaxel_snr.shape, dtype=bool)
        with np.errstate(invalid="ignore"):
            return self.median_spaxel_snr < LOW_SNR_WARNING

    def best_time_index(self, seeing_index: int) -> int | None:
        """Index of the time with the highest shear S/N at one seeing (None if none)."""
        col = self.shear_snr[:, seeing_index]
        if not np.any(np.isfinite(col)):
            return None
        return int(np.nanargmax(col))

    def snr_vs_seeing(self, time_index: int | None = None) -> np.ndarray:
        """Shear S/N against seeing at ``time_index`` (default: the best time per seeing)."""
        if time_index is not None:
            return self.shear_snr[time_index]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmax(self.shear_snr, axis=0)

    def night_summary(self) -> dict:
        """One-line-per-seeing summary: best S/N and when it happens."""
        out = {}
        for j, s in enumerate(self.seeing_ref):
            i = self.best_time_index(j)
            out[float(s.to_value(u.arcsec))] = (
                None if i is None else (float(self.shear_snr[i, j]), self.times[i].iso)
            )
        return out


def _narrowed(instrument, observed_wavelength: u.Quantity, window_kms: float):
    """``instrument`` (IFU or SlitSpectrograph) re-windowed to ``+/- window_kms``
    around the line, at its native dispersion. Shared by :func:`narrowed_ifu` and
    :func:`narrowed_slit` -- both types expose the same ``wavelength_min``/
    ``wavelength_max``/``n_wave`` fields and support ``dataclasses.replace``."""
    channel = ((instrument.wavelength_max - instrument.wavelength_min) / instrument.n_wave).to(u.AA)
    half = (observed_wavelength * window_kms / C_KMS).to(u.AA)
    n_wave = max(int(round((2 * half / channel).decompose().value)), 8)
    half = 0.5 * n_wave * channel
    return dataclasses.replace(
        instrument,
        wavelength_min=(observed_wavelength - half).to(instrument.wavelength_min.unit),
        wavelength_max=(observed_wavelength + half).to(instrument.wavelength_max.unit),
        n_wave=n_wave,
    )


def narrowed_ifu(ifu, observed_wavelength: u.Quantity, window_kms: float):
    """``ifu`` re-windowed to ``+/- window_kms`` around the line, at its native dispersion."""
    return _narrowed(ifu, observed_wavelength, window_kms)


def narrowed_slit(slit, observed_wavelength: u.Quantity, window_kms: float):
    """``slit`` re-windowed to ``+/- window_kms`` around the line, at its native dispersion."""
    return _narrowed(slit, observed_wavelength, window_kms)


def _forecast(
    instrument,
    target: ForecastTarget,
    night: Night,
    library: SkyLibrary,
    *,
    g: tuple[float, float],
    seeing_ref: u.Quantity,
    solar_flux: float,
    site: Site,
    max_airmass: float,
    time_step: u.Quantity,
    window_kms: float,
    include_extinction: bool,
    snr_threshold: float,
    priors: dict | None,
    estimator: str,
    intrinsic_sigma: float | None,
    psf_error: float | None,
    narrow_fn,
    sky_rate_fn,
) -> TargetForecast:
    """Shared implementation of :func:`forecast_ifu` and :func:`forecast_slit` --
    identical logic, parameterised only by which narrowing/sky-rate helper matches
    ``instrument``'s type. See either public function's docstring for the parameters."""
    seeing_ref = u.Quantity(seeing_ref).to(u.arcsec)
    if seeing_ref.ndim != 1 or len(seeing_ref) == 0:
        raise ValueError("seeing_ref must be a non-empty 1-D array")
    lam_obs = target.observed_wavelength
    lam_um = lam_obs.to(u.um)
    lo, hi = library.wavelength_range
    if not lo <= lam_obs <= hi:
        raise ValueError(
            f"observed line at {lam_obs.to(u.nm):.0f} is outside the sky library's "
            f"{lo:.0f}-{hi:.0f} range; the near-infrared is not supported yet"
        )
    if include_extinction and lam_obs > 1087.0 * u.nm:
        raise ValueError("the CTIO extinction curve ends at 1087 nm; pass include_extinction=False")
    instrument_line = narrow_fn(instrument, lam_obs, window_kms)
    if float(np.max(instrument_line.throughput(np.array([lam_um.value]) * u.um))) <= 0.0:
        raise ValueError(
            f"the instrument's throughput curve is zero at the observed line ({lam_obs:.0f}): "
            "this target's line is outside its tabulated coverage"
        )

    step = max(int(round((time_step / (night.times[1] - night.times[0])).decompose().value)), 1)
    times = night.times[::step]
    track = target_track(site, target.coord, times)
    moon = moon_state(site, times, target.coord)
    ecl_lon, ecl_lat = ecliptic_from_sun(target.coord, times)
    n_t, n_s = len(times), len(seeing_ref)

    shape = (n_t, n_s)
    nan = np.full(shape, np.nan)
    out = {
        "line_snr": nan.copy(),
        "peak": nan.copy(),
        "snr": nan.copy(),
        "snr_stat": nan.copy(),
        "pbias": np.full((n_t, n_s, 2), np.nan),
        "nsp": nan.copy(),
        "msnr": nan.copy(),
        "sig": np.full((n_t, n_s, 2), np.nan),
        "sigfix": np.full((n_t, n_s, 2), np.nan),
    }
    detectable = np.zeros(shape, dtype=bool)
    eff_seeing = np.full(shape, np.nan) * u.arcsec

    for i in range(n_t):
        x = float(track.airmass[i])
        if not np.isfinite(x) or x > max_airmass:
            continue
        zm = 90.0 * u.deg - moon.altitude[i]
        cond = SkyConditions(
            airmass=x,
            solar_flux=solar_flux,
            moon_phase_angle=moon.phase_angle[i],
            moon_target_separation=moon.separation[i],
            moon_zenith_distance=zm,
            ecliptic_lon_from_sun=ecl_lon[i],
            ecliptic_lat=ecl_lat[i],
        )
        sky_rate = sky_rate_fn(instrument_line, cond, library)
        flux = target.line_flux
        if include_extinction:
            flux = flux * float(atmospheric_transmission(lam_obs, x)[0])
        for j, s_ref in enumerate(seeing_ref):
            eff = seeing_at(s_ref, x, lam_obs)
            eff_seeing[i, j] = eff
            instrument_t = dataclasses.replace(
                instrument_line, psf=PSF(seeing_fwhm=eff), sky_electron_rate=sky_rate
            )
            try:
                res: FisherResult = shear_fisher(
                    instrument_t,
                    target.galaxy,
                    line_flux=flux,
                    line_wavelength=target.rest_wavelength,
                    g=g,
                    estimator=estimator,
                    snr_threshold=snr_threshold,
                    priors=priors,
                    psf_error=psf_error if estimator == "cube" else None,
                )
            except ValueError:
                continue  # undetectable at this point: leave NaN, detectable stays False
            detectable[i, j] = True
            out["line_snr"][i, j] = res.line_snr
            out["peak"][i, j] = res.peak_spaxel_snr
            out["nsp"][i, j] = res.n_spaxels
            out["msnr"][i, j] = res.median_spaxel_snr
            out["sig"][i, j] = res.sigma_marginal
            out["sigfix"][i, j] = res.sigma_fixed
            out["pbias"][i, j] = res.psf_bias
            gv = np.asarray(g, dtype=float)
            gnorm = float(np.linalg.norm(gv))
            if gnorm > 0:
                ghat = gv / gnorm
                var_stat = float(ghat @ res.covariance[:2, :2] @ ghat)
                extra = (intrinsic_sigma or 0.0) ** 2
                if np.all(np.isfinite(res.psf_bias)):
                    extra += float(np.sum(np.square(res.psf_bias)))
                out["snr_stat"][i, j] = gnorm / np.sqrt(var_stat)
                out["snr"][i, j] = gnorm / np.sqrt(var_stat + extra)

    return TargetForecast(
        times=times,
        airmass=np.asarray(track.airmass),
        altitude=track.altitude,
        moon_illumination=np.asarray(moon.illumination),
        moon_altitude=moon.altitude,
        moon_separation=moon.separation,
        seeing_ref=seeing_ref,
        effective_seeing=eff_seeing,
        line_snr=out["line_snr"],
        peak_spaxel_snr=out["peak"],
        sigma_g=out["sig"],
        sigma_g_fixed=out["sigfix"],
        shear_snr=out["snr"],
        shear_snr_stat=out["snr_stat"],
        psf_bias=out["pbias"],
        n_spaxels=out["nsp"],
        median_spaxel_snr=out["msnr"],
        detectable=detectable,
        g=(float(g[0]), float(g[1])),
        observed_wavelength=lam_obs,
        meta={
            "site": site.name,
            "solar_flux": solar_flux,
            "window_kms": window_kms,
            "include_extinction": include_extinction,
            "max_airmass": max_airmass,
            "exposure_total": (instrument.exposure_time * instrument.n_exposures).to(u.s),
            "target": target.name,
            "estimator": estimator,
            "intrinsic_sigma": intrinsic_sigma,
            "psf_error": psf_error,
        },
    )


def forecast_ifu(
    ifu,
    target: ForecastTarget,
    night: Night,
    library: SkyLibrary,
    *,
    g: tuple[float, float],
    seeing_ref: u.Quantity,
    solar_flux: float,
    site: Site = SOAR,
    max_airmass: float = 2.0,
    time_step: u.Quantity = 60.0 * u.min,
    window_kms: float = 500.0,
    include_extinction: bool = True,
    snr_threshold: float = 3.0,
    priors: dict | None = None,
    estimator: str = "cube",
    intrinsic_sigma: float | None = None,
    psf_error: float | None = None,
) -> TargetForecast:
    """Shear S/N against seeing through ``night`` for ``target`` with ``ifu``.

    Parameters
    ----------
    ifu : IFU
        The instrument configuration (exposure, throughput, ... are used as
        given; its PSF seeing and sky rate are overwritten at each point).
    target : ForecastTarget
    night : Night
        From :func:`kino.observing.night_window`.
    library : SkyLibrary
    g : tuple[float, float]
        The shear you expect (fiducial for the derivatives and the S/N).
    seeing_ref : Quantity (arcsec), array
        Seeing values *as site seeing is quoted*: FWHM at 500 nm at zenith.
    solar_flux : float
        Monthly-mean 10.7 cm flux, s.f.u. (see :mod:`kino.observing.solar`).
    max_airmass : float
        Times with a higher airmass are skipped (the moonlight grid stops at 2).
    time_step : Quantity (time)
        Spacing of the sampled times within the night.
    window_kms : float
        Half-width of the spectral window around the line, km/s.
    include_extinction : bool
        Dim the line flux by the CTIO extinction at each airmass.
    priors : dict, optional
        Nuisance-parameter priors for the Fisher estimate
        (:data:`kino.analysis.fisher.DEFAULT_CUBE_PRIORS`).
    estimator : {"cube", "moments"}
        Fisher estimator (see :func:`kino.analysis.fisher.shear_fisher`); the cube
        one is the default: unbiased and using the flux image as well as the
        kinematics (``"moments"`` is IFU only).
    intrinsic_sigma : float, optional
        A per-galaxy shear floor from what the model does not capture -- non-circular
        motions, disc thickness, warps, everything a perfect-model Cramer-Rao bound
        ignores -- added in quadrature. Default ``None`` (no floor): the primary
        forecast is the **idealised, statistical** S/N (``shear_snr_stat``); real-galaxy
        floor calibration is a future project's problem, not this pipeline's (2026-09-27
        direction). Pass a value only for an explicit *sensitivity* comparison (the
        README's published-forecast target is ``sigma_eps`` ~ 0.02-0.04, for reference).
    psf_error : float, optional
        Fractional error on the assumed PSF FWHM (truth larger by this fraction); adds
        the resulting shear bias, in quadrature, to the error and reports it.
    """
    return _forecast(
        ifu, target, night, library, g=g, seeing_ref=seeing_ref, solar_flux=solar_flux, site=site,
        max_airmass=max_airmass, time_step=time_step, window_kms=window_kms,
        include_extinction=include_extinction, snr_threshold=snr_threshold, priors=priors,
        estimator=estimator, intrinsic_sigma=intrinsic_sigma, psf_error=psf_error,
        narrow_fn=narrowed_ifu, sky_rate_fn=ifu_sky_electron_rate,
    )  # fmt: skip


def forecast_slit(
    slit,
    target: ForecastTarget,
    night: Night,
    library: SkyLibrary,
    *,
    g: tuple[float, float],
    seeing_ref: u.Quantity,
    solar_flux: float,
    site: Site = SOAR,
    max_airmass: float = 2.0,
    time_step: u.Quantity = 60.0 * u.min,
    window_kms: float = 500.0,
    include_extinction: bool = True,
    snr_threshold: float = 3.0,
    priors: dict | None = None,
    estimator: str = "cube",
    intrinsic_sigma: float | None = None,
    psf_error: float | None = None,
) -> TargetForecast:
    """Shear S/N against seeing through ``night`` for ``target`` with ``slit``.

    Slit losses (a source wider than the slit, or a misaligned ``slit_pa``/
    ``slit_offset``) fall out of the same forward render the IFU case uses -- no
    separate loss factor. ``estimator`` must be ``"cube"``
    (:class:`~kino.inference.likelihood.SlitCubeLikelihood`'s bound); there is no
    slit moment-map likelihood.

    Parameters
    ----------
    slit : SlitSpectrograph
        The instrument configuration (exposure, throughput, slit geometry, ... are
        used as given; its PSF seeing and sky rate are overwritten at each point).
    target : ForecastTarget
    night : Night
        From :func:`kino.observing.night_window`.
    library : SkyLibrary
    g : tuple[float, float]
        The shear you expect (fiducial for the derivatives and the S/N).
    seeing_ref : Quantity (arcsec), array
        Seeing values *as site seeing is quoted*: FWHM at 500 nm at zenith.
    solar_flux : float
        Monthly-mean 10.7 cm flux, s.f.u. (see :mod:`kino.observing.solar`).
    max_airmass : float
        Times with a higher airmass are skipped (the moonlight grid stops at 2).
    time_step : Quantity (time)
        Spacing of the sampled times within the night.
    window_kms : float
        Half-width of the spectral window around the line, km/s.
    include_extinction : bool
        Dim the line flux by the CTIO extinction at each airmass.
    priors : dict, optional
        Nuisance-parameter priors for the Fisher estimate
        (:data:`kino.analysis.fisher.DEFAULT_CUBE_PRIORS`).
    estimator : {"cube", "moments"}
        Fisher estimator (see :func:`kino.analysis.fisher.shear_fisher`); the cube
        one is the default: unbiased and using the flux image as well as the
        kinematics (``"moments"`` is IFU only).
    intrinsic_sigma : float, optional
        A per-galaxy shear floor from what the model does not capture -- non-circular
        motions, disc thickness, warps, everything a perfect-model Cramer-Rao bound
        ignores -- added in quadrature. Default ``None`` (no floor): the primary
        forecast is the **idealised, statistical** S/N (``shear_snr_stat``); real-galaxy
        floor calibration is a future project's problem, not this pipeline's (2026-09-27
        direction). Pass a value only for an explicit *sensitivity* comparison (the
        README's published-forecast target is ``sigma_eps`` ~ 0.02-0.04, for reference).
    psf_error : float, optional
        Fractional error on the assumed PSF FWHM (truth larger by this fraction); adds
        the resulting shear bias, in quadrature, to the error and reports it.
    """
    if estimator != "cube":
        raise ValueError(
            "forecast_slit only supports estimator='cube' -- no slit moment likelihood"
        )
    return _forecast(
        slit, target, night, library, g=g, seeing_ref=seeing_ref, solar_flux=solar_flux, site=site,
        max_airmass=max_airmass, time_step=time_step, window_kms=window_kms,
        include_extinction=include_extinction, snr_threshold=snr_threshold, priors=priors,
        estimator=estimator, intrinsic_sigma=intrinsic_sigma, psf_error=psf_error,
        narrow_fn=narrowed_slit, sky_rate_fn=slit_sky_electron_rate,
    )  # fmt: skip
