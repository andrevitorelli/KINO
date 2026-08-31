"""Ensemble analysis.

Shape-noise characterisation, multiplicative/additive (m, c) bias
calibration, instrument forecasts, and plotting over galaxy ensembles.
"""

from kino.analysis.moments import MomentMaps, compute_moment_maps, crop_to_line

__all__ = [
    "MomentMaps",
    "compute_moment_maps",
    "crop_to_line",
]
