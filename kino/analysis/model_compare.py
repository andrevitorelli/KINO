"""Bayesian model comparison: lensed vs. unlensed vs. intrinsically disturbed.

The distinctive KL systematic (TEST_DATA.md Section 3): a distorted velocity
field can arise from lensing *or* from intrinsic causes. Runs a comparable
set of forward models on one dataset via the nested sampler and returns
ranked evidences + Bayes factors.

Minimum model set (TEST_DATA.md Section 3):
    1. Lensed regular disc (the KL hypothesis: intrinsic disc + shear g1, g2).
    2. Unlensed regular disc (null: shear fixed to zero).
    3. Tidally disturbed / interacting disc (intrinsic non-lensing distortion).
    4. (stretch) warp, bar/non-circular streaming, beam-smearing-only.

Validation expectations to encode as assertions once this exists: on an
intrinsically regular TNG50 galaxy with a known shear injected, evidence
should favour model 1 over 2 and 3; on an intrinsically interacting/tidal
TNG50 galaxy with no shear, evidence should favour 3, not 1. Also report a
prior-sensitivity analysis (Bayes factors move under reasonable prior
variations, per the Jeffreys-Lindley effect).

Sampler: **nautilus only** (the project dropped the dynesty cross-check
originally proposed here in favour of a single sampler stack — see
CLAUDE.md's "Language / stack" convention).

Not implemented yet. kino.inference (README.md Phase 5) and the regular-disc
model (models 1-2 above, README.md Phase 1) are both done now -- the actual
remaining blocker is narrower than this module's original scope suggested:
a tidally-disturbed/interacting disc model (model 3) does not exist in
kino.galaxy yet, and nothing here can be implemented without it.
"""

from __future__ import annotations

from kino.io.base import Datacube


def compare_models(cube: Datacube, models: list[str] | None = None):
    """Rank ``models`` by nautilus log-evidence on ``cube``; return Bayes factors."""
    raise NotImplementedError(
        "kino.analysis.model_compare is not implemented yet -- it depends on "
        "kino.inference and kino.galaxy's model set (README.md Phases 1, 5)."
    )
