"""Unweighted flux-moment shape measurement, for validating/visualizing shear.

Not a science-grade shape-measurement pipeline (no PSF correction, no
weight function to suppress profile-wing noise) -- just enough to confirm a
rendered, sheared image has the ellipticity ``kino.lensing.shear`` predicts
analytically, and to draw the ellipse/axis overlays used in demo plots.
Returns ellipticity in the same ``(eps1, eps2)`` convention as
``kino.lensing.shear`` (standard math-angle phase, see that module's
docstring) -- not directly in ``kino.galaxy``'s East-of-North ``pa``.
"""

from __future__ import annotations

import numpy as np


def measure_moments(image: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Flux-weighted centroid and second moments of ``image`` sampled at ``(x, y)``.

    Returns ``(xbar, ybar, qxx, qyy, qxy)``.
    """
    flux = np.asarray(image, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    total = flux.sum()
    xbar = (flux * x).sum() / total
    ybar = (flux * y).sum() / total
    dx = x - xbar
    dy = y - ybar
    qxx = (flux * dx**2).sum() / total
    qyy = (flux * dy**2).sum() / total
    qxy = (flux * dx * dy).sum() / total
    return xbar, ybar, qxx, qyy, qxy


def measure_ellipticity(image: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Centroid, complex ellipticity, and semi-axes of ``image`` at ``(x, y)``.

    Returns ``(xbar, ybar, eps1, eps2, a, b)``, with ``eps = eps1 + i*eps2``
    in ``kino.lensing.shear``'s standard-math-angle convention.
    """
    xbar, ybar, qxx, qyy, qxy = measure_moments(image, x, y)
    trace_half = 0.5 * (qxx + qyy)
    diff_half = 0.5 * (qxx - qyy)
    radius = np.hypot(diff_half, qxy)
    a = np.sqrt(trace_half + radius)
    b = np.sqrt(max(trace_half - radius, 0.0))
    eps_mag = (a - b) / (a + b)
    phi = 0.5 * np.arctan2(2.0 * qxy, qxx - qyy)
    eps1 = eps_mag * np.cos(2.0 * phi)
    eps2 = eps_mag * np.sin(2.0 * phi)
    return xbar, ybar, eps1, eps2, a, b
