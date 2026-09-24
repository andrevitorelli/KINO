"""Ensemble analysis.

Shape-noise characterisation, multiplicative/additive (m, c) bias
calibration, instrument forecasts, and plotting over galaxy ensembles.

``kino.analysis.injection`` is deliberately **not** re-exported here: it
depends on ``kino.inference``, which itself depends on
``kino.analysis.moments``, so importing it eagerly from this package would
create an import cycle. Import it as ``from kino.analysis.injection import
run_injection, summarize_trials`` instead.
"""

from kino.analysis.moments import MomentMaps, compute_moment_maps, crop_to_line

__all__ = [
    "MomentMaps",
    "compute_moment_maps",
    "crop_to_line",
]
