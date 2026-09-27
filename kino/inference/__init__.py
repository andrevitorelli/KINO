"""Shear inference.

Likelihoods, priors, and a sampler wrapper behind a common API, used to
recover the (g1, g2) posterior from mock or real data. ``nautilus``
(neural-network-boosted importance nested sampling) is kino's sole sampler
-- see ``CLAUDE.md``'s "Language / stack" convention.
"""

from kino.inference.likelihood import (
    GALAXY_FIELDS,
    IFUCubeLikelihood,
    IFUMomentLikelihood,
    SlitCubeLikelihood,
)
from kino.inference.priors import build_prior, tully_fisher_prior, tully_fisher_velocity
from kino.inference.sampler import Posterior, fit_shear

__all__ = [
    "GALAXY_FIELDS",
    "IFUCubeLikelihood",
    "IFUMomentLikelihood",
    "Posterior",
    "SlitCubeLikelihood",
    "build_prior",
    "fit_shear",
    "tully_fisher_prior",
    "tully_fisher_velocity",
]
