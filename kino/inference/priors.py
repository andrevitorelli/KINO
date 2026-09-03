"""Priors for shear inference: a thin ``nautilus.Prior`` builder, plus an
explicitly uncalibrated ("toy") Tully-Fisher prior for ``v_flat``.

Implements ``README.md`` Phase 5's priors bullet. No calibrated
Tully-Fisher relation exists in kino yet (Phase 1's own consistency-check
item is deferred here for exactly that reason) -- :func:`tully_fisher_prior`
uses the same toy power law as
``notebooks/03_tully_fisher_shear_signal.ipynb`` (``v = v0 (F/F0)^{1/alpha}``,
``alpha=4``) with a log-normal scatter, clearly not a real calibration.
Use a wide/uninformative prior on ``v_flat`` instead if you don't want to
implicitly assume this toy relation is right.
"""

from __future__ import annotations

import numpy as np
from nautilus import Prior
from scipy.stats import rv_continuous

#: The same toy Tully-Fisher power-law index used in
#: notebooks/03_tully_fisher_shear_signal.ipynb.
TOY_TF_ALPHA = 4.0


def tully_fisher_velocity(
    flux: float, *, v0: float, flux0: float, alpha: float = TOY_TF_ALPHA
) -> float:
    """Toy Tully-Fisher relation: ``v = v0 * (flux / flux0) ** (1 / alpha)``.

    Not a calibrated relation -- see the module docstring.
    """
    return v0 * (flux / flux0) ** (1.0 / alpha)


def tully_fisher_prior(
    flux: float,
    *,
    v0: float,
    flux0: float,
    alpha: float = TOY_TF_ALPHA,
    scatter_dex: float = 0.1,
) -> rv_continuous:
    """A log-normal ``scipy.stats`` distribution for ``v_flat``, centred on
    the toy Tully-Fisher relation's prediction at ``flux``, with
    ``scatter_dex`` scatter in :math:`\\log_{10} v_{\\rm flat}`.

    Pass the result as a parameter's ``dist`` in :func:`build_prior` (or
    directly to ``nautilus.Prior.add_parameter``).
    """
    from scipy.stats import norm

    v_center = tully_fisher_velocity(flux, v0=v0, flux0=flux0, alpha=alpha)
    log_v_center = np.log10(v_center)

    class _LogNormalV(rv_continuous):
        def _ppf(self, q):
            return 10.0 ** norm.ppf(q, loc=log_v_center, scale=scatter_dex)

        def _cdf(self, x):
            with np.errstate(divide="ignore"):
                return norm.cdf(np.log10(x), loc=log_v_center, scale=scatter_dex)

    return _LogNormalV(a=0.0, name="toy_tully_fisher_v_flat")


def build_prior(free_params: dict) -> Prior:
    """Build a ``nautilus.Prior`` from ``{name: dist}``.

    ``dist`` is either a ``(low, high)`` tuple (uniform) or any object with
    ``.ppf``/``.rvs`` (e.g. a ``scipy.stats`` frozen distribution, or
    :func:`tully_fisher_prior`'s return value) -- exactly what
    ``nautilus.Prior.add_parameter(dist=...)`` accepts.
    """
    prior = Prior()
    for name, dist in free_params.items():
        prior.add_parameter(name, dist=dist)
    return prior
