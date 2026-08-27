"""Injection-recovery: apply a known shear, re-observe, and check recovery.

Takes any Datacube (or a sim galaxy) (TEST_DATA.md's validation ladder,
Section 0), applies a **known** shear via KINO's forward model, re-observes
it through an instrument backend, runs inference, and reports recovered vs.
input (g1, g2), multiplicative/additive bias (m, c), and residual shape
noise sigma_eps. This is the core validation check for every rung of the
ladder (TEST_DATA.md Section 2).

Not implemented yet: depends on kino.lensing.apply_shear,
kino.instruments.Instrument, and kino.inference.fit_shear, none of which
exist yet (roadmap Phases 2, 3, and 5 in README.md).
"""

from __future__ import annotations

from kino.io.base import Datacube


def recover_shear(cube: Datacube, g1: float, g2: float, *, sampler: str = "nautilus"):
    """Inject (g1, g2) into ``cube``'s underlying model, re-observe, and infer it back."""
    raise NotImplementedError(
        "kino.analysis.injection is not implemented yet -- it depends on kino.lensing, "
        "kino.instruments, and kino.inference (README.md Phases 2, 3, 5)."
    )
