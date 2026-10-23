"""Batch forecasting across a date range and/or a target list.

``kino.analysis.forecast``'s ``forecast_ifu``/``forecast_slit`` forecast ONE
target through ONE night. Real proposal target selection needs the step
above that: repeat the same forecast over many nights, to find a semester's
best observing windows, and over many candidate targets, to rank them
(requested interface, 2026-09-26: "given a telescope/detector, a galaxy
spectrum, its RA/Dec, the night and the expected shear, the pipeline returns
S/N as a function of seeing... then batch it over a candidate list to find
the optimal candidates across a semester"). This module adds exactly that
batching layer -- no new physics: it loops ``forecast_ifu``/``forecast_slit``
over dates and targets and collapses each night's full (time, seeing) grid to
its single best point.

Real candidate lists are private (``CLAUDE.md``: never commit one, or use a
real target name in code, tests or notebooks); this module and its tests use
only synthetic targets.

**Cost.** A full semester (~180 nights) at a fine seeing/time grid is slow:
each night forecast evaluates several times x seeing points, each its own
``shear_fisher`` call (a few seconds for an IFU, much slower for a slit --
see ``kino.analysis.fisher``'s module docstring). Default to a single
representative ``seeing_ref``, a coarse ``time_step``, and a strided date
range (:func:`semester_dates`'s ``step_days``) for an exploratory sweep over
many candidates; only widen either once a shortlist exists.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.time import Time

from kino.analysis.forecast import ForecastTarget, forecast_ifu, forecast_slit
from kino.instruments.slit import SlitSpectrograph
from kino.observing.night import night_window
from kino.observing.site import SOAR, Site
from kino.observing.skymodel import SkyLibrary
from kino.observing.solar import solar_flux as _tabulated_solar_flux


def semester_dates(start: str, end: str, step_days: int = 7) -> list[str]:
    """Calendar dates (``YYYY-MM-DD``) from ``start`` to ``end`` inclusive, every ``step_days``.

    A target's visibility changes slowly night to night, so a weekly stride
    (the default) is usually enough to map out a semester's observable
    window without evaluating every single night -- narrow ``step_days``
    once a promising stretch is found.
    """
    if step_days < 1:
        raise ValueError(f"step_days must be >= 1, got {step_days}")
    t0, t1 = Time(start), Time(end)
    if t1 < t0:
        raise ValueError(f"end ({end}) is before start ({start})")
    n_days = int(round((t1 - t0).to_value(u.day)))
    return [(t0 + i * u.day).iso[:10] for i in range(0, n_days + 1, step_days)]


@dataclass(frozen=True)
class NightlyForecast:
    """One target's best-case shear forecast on each night of a date range.

    Arrays/lists are indexed by night, in the order of ``dates``. A night
    where the target never clears ``max_airmass``, the line was undetectable
    at every point tried, or no F10.7 prediction is tabulated for that month
    (see ``forecast_over_nights``'s ``solar_flux_fn``) is NaN.

    Attributes
    ----------
    target : ForecastTarget
    dates : list of str
    seeing_ref : Quantity (arcsec)
        The seeing grid that was swept each night.
    g : tuple[float, float]
        The expected shear used for every night's forecast.
    shear_snr, shear_snr_stat : array (n_dates,)
        Best-point total and statistical-only shear S/N (see
        ``TargetForecast.shear_snr``/``shear_snr_stat``).
    seeing : Quantity (arcsec), array (n_dates,)
        The seeing (from ``seeing_ref``) at which the best point occurred.
    airmass : array (n_dates,)
    time : list of str or None
        ISO time of the best point, or None for a NaN night.
    """

    target: ForecastTarget
    dates: list[str]
    seeing_ref: u.Quantity
    g: tuple[float, float]
    shear_snr: np.ndarray
    shear_snr_stat: np.ndarray
    seeing: u.Quantity
    airmass: np.ndarray
    time: list[str | None]

    @property
    def observable_mask(self) -> np.ndarray:
        """``True`` for nights with a finite best-point shear S/N."""
        return np.isfinite(self.shear_snr)

    @property
    def best_night_index(self) -> int | None:
        """Index of the single best night (None if the target was never detectable)."""
        if not np.any(self.observable_mask):
            return None
        return int(np.nanargmax(self.shear_snr))

    def summary(self) -> dict:
        """The single best night across the whole range (``detectable=False`` if none)."""
        i = self.best_night_index
        if i is None:
            return {"detectable": False, "name": self.target.name}
        return {
            "detectable": True,
            "name": self.target.name,
            "date": self.dates[i],
            "shear_snr": float(self.shear_snr[i]),
            "shear_snr_stat": float(self.shear_snr_stat[i]),
            "seeing": self.seeing[i],
            "airmass": float(self.airmass[i]),
            "time": self.time[i],
        }


def forecast_over_nights(
    instrument,
    target: ForecastTarget,
    dates: list[str],
    library: SkyLibrary,
    *,
    g: tuple[float, float],
    seeing_ref: u.Quantity,
    site: Site = SOAR,
    solar_flux_fn=None,
    **forecast_kwargs,
) -> NightlyForecast:
    """``target``'s best-case shear forecast on each night in ``dates``.

    Loops ``forecast_ifu``/``forecast_slit`` (auto-selected from
    ``instrument``'s type) over ``dates``, and collapses each night's full
    (time, seeing) grid to its single best point (highest ``shear_snr``).

    Parameters
    ----------
    instrument : IFU or SlitSpectrograph
    dates : list of str
        Calendar dates (``"2027-03-14"``); see :func:`semester_dates` to
        build an evenly-spaced range.
    solar_flux_fn : callable, optional
        ``date -> F10.7`` (s.f.u.). Default:
        ``kino.observing.solar.solar_flux(date)[0]`` (the NOAA-predicted
        monthly mean). A night outside its tabulated months is skipped (left
        NaN) with one warning for the whole run, not an exception, so a long
        date range doesn't abort partway through -- pass your own function
        to cover out-of-table months.
    **forecast_kwargs
        Passed through to ``forecast_ifu``/``forecast_slit`` (``max_airmass``,
        ``time_step``, ``window_kms``, ``include_extinction``,
        ``snr_threshold``, ``priors``, ``estimator``, ``intrinsic_sigma``,
        ``psf_error``).

    A target/instrument mismatch that's independent of the date (the line
    outside the sky library's or the instrument's own wavelength coverage)
    raises on the very first night, exactly as ``forecast_ifu``/
    ``forecast_slit`` already would -- this is a configuration error, not a
    per-night NaN, so it is not caught here (see :func:`rank_candidates` for
    batching over several targets where one bad target shouldn't abort the
    rest).
    """
    forecast_fn = forecast_slit if isinstance(instrument, SlitSpectrograph) else forecast_ifu
    sf_fn = solar_flux_fn or (lambda d: _tabulated_solar_flux(d)[0])
    seeing_ref = u.Quantity(seeing_ref)
    if seeing_ref.isscalar:
        seeing_ref = seeing_ref.reshape(1)

    n = len(dates)
    shear_snr = np.full(n, np.nan)
    shear_snr_stat = np.full(n, np.nan)
    seeing = np.full(n, np.nan) * u.arcsec
    airmass = np.full(n, np.nan)
    time: list[str | None] = [None] * n
    skipped_months: set[str] = set()

    for i, date in enumerate(dates):
        try:
            sf = sf_fn(date)
        except KeyError:
            skipped_months.add(str(date)[:7])
            continue
        night = night_window(site, date)
        fc = forecast_fn(
            instrument, target, night, library, g=g, seeing_ref=seeing_ref,
            solar_flux=sf, site=site, **forecast_kwargs,
        )  # fmt: skip
        if not np.any(np.isfinite(fc.shear_snr)):
            continue
        i_t, j_s = np.unravel_index(np.nanargmax(fc.shear_snr), fc.shear_snr.shape)
        shear_snr[i] = fc.shear_snr[i_t, j_s]
        shear_snr_stat[i] = fc.shear_snr_stat[i_t, j_s]
        seeing[i] = fc.seeing_ref[j_s]
        airmass[i] = fc.airmass[i_t]
        time[i] = fc.times[i_t].iso

    if skipped_months:
        warnings.warn(
            f"no F10.7 prediction for {sorted(skipped_months)}; those nights are NaN "
            "(pass solar_flux_fn to cover them)",
            stacklevel=2,
        )

    return NightlyForecast(
        target=target, dates=list(dates), seeing_ref=seeing_ref, g=g,
        shear_snr=shear_snr, shear_snr_stat=shear_snr_stat, seeing=seeing,
        airmass=airmass, time=time,
    )  # fmt: skip


def rank_candidates(
    instrument,
    targets: list[ForecastTarget],
    dates: list[str],
    library: SkyLibrary,
    *,
    g: tuple[float, float],
    seeing_ref: u.Quantity,
    site: Site = SOAR,
    solar_flux_fn=None,
    **forecast_kwargs,
) -> list[NightlyForecast]:
    """:func:`forecast_over_nights` for every target, ranked by best-night shear S/N.

    Unlike :func:`forecast_over_nights` itself, a target/instrument
    mismatch here (``ValueError`` from the line falling outside the sky
    library's or the instrument's wavelength coverage) marks *that* target
    undetectable (``summary()["detectable"] = False``) rather than aborting
    the whole candidate list -- one bad candidate (e.g. a typo'd redshift)
    shouldn't block ranking the rest. Returns one :class:`NightlyForecast`
    per target, most observable first (``summary()["detectable"] = False``
    entries last, in input order).
    """
    results = []
    for t in targets:
        try:
            results.append(
                forecast_over_nights(
                    instrument,
                    t,
                    dates,
                    library,
                    g=g,
                    seeing_ref=seeing_ref,
                    site=site,
                    solar_flux_fn=solar_flux_fn,
                    **forecast_kwargs,
                )  # fmt: skip
            )
        except ValueError:
            n = len(dates)
            results.append(
                NightlyForecast(
                    target=t,
                    dates=list(dates),
                    seeing_ref=u.Quantity(seeing_ref),
                    g=g,
                    shear_snr=np.full(n, np.nan),
                    shear_snr_stat=np.full(n, np.nan),
                    seeing=np.full(n, np.nan) * u.arcsec,
                    airmass=np.full(n, np.nan),
                    time=[None] * n,
                )  # fmt: skip
            )

    def rank_key(r: NightlyForecast) -> float:
        i = r.best_night_index
        return -r.shear_snr[i] if i is not None else np.inf

    return sorted(results, key=rank_key)
