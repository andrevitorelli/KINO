"""Injection-recovery: apply a known shear, re-observe, and check recovery.

Implements ``README.md`` Phase 6's injection-recovery, shape-noise, (m, c)
bias, and null-test bullets for the **IFU** case. Each trial takes an
intrinsic :class:`~kino.galaxy.DiscGalaxy`, applies a **known** reduced shear
``(g1, g2)`` via :func:`kino.lensing.apply_shear`, observes it through an
:class:`~kino.instruments.ifu.IFU` (one noisy realisation), runs
:func:`kino.inference.fit_shear`, and records recovered vs. input shear.
:func:`summarize_trials` then reduces an ensemble of trials to the
calibration numbers a shear measurement needs:

* multiplicative/additive bias ``(m, c)`` from ``g_rec = (1 + m) g_true + c``
  (needs trials at two or more distinct true shears),
* residual shape noise ``sigma_eps``, the per-galaxy scatter of
  ``g_rec - g_true`` (the shear-equivalent noise of one galaxy),
* the mean bias at a single shear (with ``g_true = 0`` this is the null test),
* the pull ``(g_rec - g_true) / sigma_post``, whose standard deviation should
  be ~1 if the posteriors' widths are honest.

Design notes
------------
* The input is a ``DiscGalaxy``, **not** a ``Datacube`` as the original stub
  sketched: a cube is already an observation and cannot be re-sheared, so
  injection needs the underlying model. (Known-truth *cube* recovery, e.g.
  from ``kino.io.kinms``, is a different check -- fitting an unlensed model to
  a cube -- and is not what this module does.)
* Recovered shear is the posterior *median*. With a finite-width uniform prior
  and weakly-constraining data the median is pulled toward the prior centre,
  so a measured ``m < 0`` (shrinkage) is expected at low S/N and is a property
  of the estimator + prior, not necessarily a forward-model bug -- compare
  ``m`` across S/N before interpreting it.
* Every trial gets its own child seed (``numpy.random.SeedSequence.spawn``)
  for both the noise realisation and the sampler, so a run is fully
  reproducible from one ``seed`` and trials are independent of ensemble
  order/size. Trials are embarrassingly parallel; callers wanting to
  parallelise can call :func:`recover_shear` themselves with their own pool.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import astropy.units as u
import numpy as np

from kino.galaxy import DiscGalaxy
from kino.inference.likelihood import GALAXY_FIELDS, IFUMomentLikelihood
from kino.inference.priors import build_prior
from kino.inference.sampler import Posterior, fit_shear
from kino.instruments.ifu import IFU
from kino.lensing import apply_shear

FreePriors = dict | Callable[[DiscGalaxy], dict]


@dataclass
class InjectionTrial:
    """One injection-recovery trial.

    Parameters
    ----------
    g_true, g_rec : tuple[float, float]
        Injected and recovered (posterior-median) reduced shear ``(g1, g2)``.
    g_err : tuple[float, float]
        Symmetrised 1-sigma posterior width ``(minus + plus) / 2`` per component.
    galaxy : DiscGalaxy or None
        The injected galaxy, kept so results can be binned by its properties
        (inclination, size, ...).
    posterior : Posterior or None
        Full posterior, if retained.
    """

    g_true: tuple[float, float]
    g_rec: tuple[float, float]
    g_err: tuple[float, float]
    galaxy: DiscGalaxy | None = None
    posterior: Posterior | None = None


@dataclass
class ComponentSummary:
    """Ensemble statistics for one shear component.

    ``m``/``c`` (and their errors) are NaN if the trials span fewer than two
    distinct true shears (or fewer than three trials, for the errors).
    ``sigma_eps`` is NaN for a single trial.
    """

    m: float
    m_err: float
    c: float
    c_err: float
    mean_bias: float
    mean_bias_err: float
    sigma_eps: float
    mean_posterior_sigma: float
    pull_mean: float
    pull_std: float


@dataclass
class EnsembleSummary:
    """Per-component summaries for an ensemble of trials."""

    g1: ComponentSummary
    g2: ComponentSummary
    n_trials: int

    def __repr__(self) -> str:
        lines = [f"EnsembleSummary(n_trials={self.n_trials})"]
        for name, s in (("g1", self.g1), ("g2", self.g2)):
            lines.append(
                f"  {name}: m={s.m:+.3f}±{s.m_err:.3f}  c={s.c:+.4f}±{s.c_err:.4f}  "
                f"bias={s.mean_bias:+.4f}±{s.mean_bias_err:.4f}  sigma_eps={s.sigma_eps:.4f}  "
                f"pull={s.pull_mean:+.2f}±{s.pull_std:.2f}"
            )
        return "\n".join(lines)


def _resolve_free_priors(free_priors: FreePriors | None, galaxy: DiscGalaxy) -> dict:
    if free_priors is None:
        return {}
    return dict(free_priors(galaxy) if callable(free_priors) else free_priors)


def recover_shear(
    galaxy: DiscGalaxy,
    g1: float,
    g2: float,
    *,
    ifu: IFU,
    line_wavelength: u.Quantity,
    line_flux: u.Quantity,
    free_priors: FreePriors | None = None,
    shear_prior_range: float = 0.15,
    rng: np.random.Generator | None = None,
    sampler_seed: int | None = None,
    snr_threshold: float = 3.0,
    n_live: int = 500,
    n_eff: float = 2000,
    keep_posterior: bool = False,
) -> InjectionTrial:
    """Inject ``(g1, g2)`` into ``galaxy``, observe it with ``ifu``, and infer it back.

    Parameters
    ----------
    galaxy : DiscGalaxy
        The intrinsic (unlensed) truth.
    g1, g2 : float
        Injected reduced shear.
    ifu : IFU
        Instrument used both to make the data and (inside the likelihood) to
        forward-model every trial point -- see
        :class:`~kino.inference.likelihood.IFUMomentLikelihood`.
    line_wavelength : Quantity (wavelength)
        Rest-frame line wavelength.
    line_flux : Quantity (energy / time / area)
        Intrinsic (unlensed) line flux, assumed known (fixed in the fit).
    free_priors : dict or callable, optional
        Priors (in :func:`~kino.inference.priors.build_prior`'s format) for
        free parameters *other than* the shear -- e.g. ``{"inc": (30, 80)}``.
        A callable ``f(galaxy) -> dict`` lets priors depend on the galaxy
        (e.g. a Tully-Fisher prior centred on its flux). Every
        ``DiscGalaxy`` field not named here is held fixed at the galaxy's
        true value. Default: shear-only fit.
    shear_prior_range : float, optional
        Uniform prior half-width on both ``g1`` and ``g2``.
    rng : numpy.random.Generator or None, optional
        Drives the data's noise realisation; ``None`` gives noiseless data.
    sampler_seed : int or None, optional
        Seed for ``nautilus``.
    snr_threshold, n_live, n_eff : optional
        Forwarded to the likelihood / :func:`~kino.inference.sampler.fit_shear`.
    keep_posterior : bool, optional
        Store the full :class:`~kino.inference.sampler.Posterior` on the trial.

    Returns
    -------
    InjectionTrial
    """
    data = ifu.observe(
        apply_shear(galaxy, g1, g2), line_flux=line_flux, line_wavelength=line_wavelength, rng=rng
    )

    extra_priors = _resolve_free_priors(free_priors, galaxy)
    bad = [k for k in extra_priors if k not in GALAXY_FIELDS]
    if bad:
        raise ValueError(f"free_priors may only name DiscGalaxy fields, got {bad}")
    fixed = {f: getattr(galaxy, f) for f in GALAXY_FIELDS if f not in extra_priors}

    likelihood = IFUMomentLikelihood(
        ifu=ifu,
        line_wavelength=line_wavelength,
        data=data,
        fixed_galaxy_params=fixed,
        fixed_line_flux=line_flux,
        snr_threshold=snr_threshold,
    )
    prior = build_prior(
        {
            "g1": (-shear_prior_range, shear_prior_range),
            "g2": (-shear_prior_range, shear_prior_range),
            **extra_priors,
        }
    )
    posterior = fit_shear(prior, likelihood, n_live=n_live, seed=sampler_seed, n_eff=n_eff)

    summary = posterior.summary()
    g_rec = (summary["g1"][0], summary["g2"][0])
    g_err = tuple(0.5 * (summary[k][1] + summary[k][2]) for k in ("g1", "g2"))
    return InjectionTrial(
        g_true=(float(g1), float(g2)),
        g_rec=g_rec,
        g_err=g_err,
        galaxy=galaxy,
        posterior=posterior if keep_posterior else None,
    )


def _trial_seeds(seed: int, n: int) -> list[tuple[np.random.Generator, int]]:
    """Independent ``(noise generator, sampler seed)`` pairs, one per trial."""
    children = np.random.SeedSequence(seed).spawn(n)
    return [(np.random.default_rng(c), int(c.generate_state(1)[0] & 0x7FFFFFFF)) for c in children]


def run_injection(
    galaxies: Sequence[DiscGalaxy],
    shears: Sequence[tuple[float, float]],
    *,
    seed: int,
    verbose: bool = False,
    **recover_kwargs,
) -> list[InjectionTrial]:
    """Run :func:`recover_shear` for every ``(shear, galaxy)`` pair.

    Trials are ordered shear-major (all galaxies at ``shears[0]``, then at
    ``shears[1]``, ...). Trial ``k`` always uses the ``k``-th child seed of
    ``seed``, so results are reproducible; pass the same galaxy list at every
    shear to compare like with like.

    ``recover_kwargs`` are forwarded to :func:`recover_shear` (``ifu``,
    ``line_wavelength`` and ``line_flux`` are required; ``rng`` and
    ``sampler_seed`` are set here and may not be passed).
    """
    if not galaxies or not shears:
        raise ValueError("need at least one galaxy and one shear")
    for reserved in ("rng", "sampler_seed"):
        if reserved in recover_kwargs:
            raise TypeError(f"{reserved!r} is set per-trial from `seed`; do not pass it")

    pairs = [(s, g) for s in shears for g in galaxies]
    seeds = _trial_seeds(seed, len(pairs))
    trials = []
    for k, ((shear, galaxy), (rng, sampler_seed)) in enumerate(zip(pairs, seeds, strict=True)):
        trial = recover_shear(
            galaxy, shear[0], shear[1], rng=rng, sampler_seed=sampler_seed, **recover_kwargs
        )
        trials.append(trial)
        if verbose:
            print(
                f"[{k + 1}/{len(pairs)}] true=({shear[0]:+.3f},{shear[1]:+.3f}) "
                f"rec=({trial.g_rec[0]:+.3f},{trial.g_rec[1]:+.3f}) "
                f"err=({trial.g_err[0]:.3f},{trial.g_err[1]:.3f})"
            )
    return trials


def _summarize_component(true: np.ndarray, rec: np.ndarray, err: np.ndarray) -> ComponentSummary:
    n = len(true)
    resid = rec - true
    nan = float("nan")

    mean_bias = float(resid.mean())
    sigma_eps = float(resid.std(ddof=1)) if n > 1 else nan
    mean_bias_err = sigma_eps / np.sqrt(n) if n > 1 else nan

    m = m_err = c = c_err = nan
    if len(np.unique(true)) >= 2:
        if n >= 3:
            (slope, c), cov = np.polyfit(true, rec, 1, cov=True)
            m_err, c_err = float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))
        else:
            slope, c = np.polyfit(true, rec, 1)
        m, c = float(slope - 1.0), float(c)

    pull = resid / err
    return ComponentSummary(
        m=m,
        m_err=m_err,
        c=c,
        c_err=c_err,
        mean_bias=mean_bias,
        mean_bias_err=float(mean_bias_err),
        sigma_eps=sigma_eps,
        mean_posterior_sigma=float(err.mean()),
        pull_mean=float(pull.mean()),
        pull_std=float(pull.std(ddof=1)) if n > 1 else nan,
    )


def summarize_trials(trials: Sequence[InjectionTrial]) -> EnsembleSummary:
    """Reduce injection trials to ``(m, c)``, ``sigma_eps``, mean bias and pulls.

    ``mean_bias`` pools all trials regardless of true shear -- meaningful as a
    null test when every trial has ``g_true = 0``, otherwise prefer ``(m, c)``.
    """
    if not trials:
        raise ValueError("need at least one trial")
    true = np.array([t.g_true for t in trials], dtype=float)
    rec = np.array([t.g_rec for t in trials], dtype=float)
    err = np.array([t.g_err for t in trials], dtype=float)
    return EnsembleSummary(
        g1=_summarize_component(true[:, 0], rec[:, 0], err[:, 0]),
        g2=_summarize_component(true[:, 1], rec[:, 1], err[:, 1]),
        n_trials=len(trials),
    )
