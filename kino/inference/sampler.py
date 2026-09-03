"""``nautilus`` sampler wrapper: turn a ``Prior`` + likelihood into a ``Posterior``.

Implements ``README.md`` Phase 5's sampler-wrapper and posterior-summary
bullets. ``nautilus`` (neural-network-boosted importance nested sampling,
https://nautilus-sampler.readthedocs.io) is kino's sole sampler
(``CLAUDE.md``'s "Language / stack" convention) -- this module is a thin
wrapper, not an abstraction over multiple samplers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from nautilus import Sampler


@dataclass
class Posterior:
    """Equal-weighted posterior samples from a ``fit_shear`` run.

    Parameters
    ----------
    samples : dict[str, numpy.ndarray]
        ``{parameter_name: samples}``, equal-weighted (i.e. usable directly,
        with no further importance-reweighting, for summary statistics or
        a corner plot).
    log_z : float
        Log Bayesian evidence (``nautilus.Sampler.log_z``).
    param_names : list[str]
        Parameter names, in a fixed display order.
    """

    samples: dict[str, np.ndarray]
    log_z: float
    param_names: list[str]

    def summary(self) -> dict[str, tuple[float, float, float]]:
        """``{name: (median, -1sigma, +1sigma)}`` from the 16/50/84th percentiles."""
        out = {}
        for name in self.param_names:
            lo, med, hi = np.percentile(self.samples[name], [16.0, 50.0, 84.0])
            out[name] = (float(med), float(med - lo), float(hi - med))
        return out

    def __repr__(self) -> str:
        lines = [f"Posterior(log_z={self.log_z:.3f}, n_samples={self.n_samples})"]
        for name, (med, minus, plus) in self.summary().items():
            lines.append(f"  {name} = {med:.4g}  (-{minus:.3g} / +{plus:.3g})")
        return "\n".join(lines)

    @property
    def n_samples(self) -> int:
        return len(self.samples[self.param_names[0]])


def fit_shear(
    prior,
    likelihood,
    *,
    n_live: int = 2000,
    seed: int | None = None,
    verbose: bool = False,
    n_eff: float = 10000,
    **sampler_kwargs,
) -> Posterior:
    """Run ``nautilus`` nested sampling and return a :class:`Posterior`.

    Parameters
    ----------
    prior : nautilus.Prior
        E.g. from :func:`kino.inference.priors.build_prior`.
    likelihood : Callable[[dict], float]
        E.g. an :class:`~kino.inference.likelihood.IFUMomentLikelihood`.
    n_live, seed, verbose : optional
        Forwarded to ``nautilus.Sampler``/``.run()``.
    n_eff : float, optional
        Target effective sample size, forwarded to ``.run()`` -- the usual
        knob for trading runtime against posterior sample count/evidence
        precision (lower for a fast smoke test, higher for a real run).
    **sampler_kwargs
        Forwarded to ``nautilus.Sampler`` (e.g. ``pool`` for
        multiprocessing).

    Returns
    -------
    Posterior
    """
    sampler = Sampler(prior, likelihood, n_live=n_live, pass_dict=True, seed=seed, **sampler_kwargs)
    sampler.run(verbose=verbose, n_eff=n_eff)
    points, _log_w, _log_l = sampler.posterior(return_as_dict=True, equal_weight=True)
    return Posterior(samples=points, log_z=float(sampler.log_z), param_names=list(points.keys()))
