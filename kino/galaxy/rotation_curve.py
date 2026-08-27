"""Rotation-curve models for an intrinsic disc.

``r`` and ``r_t`` (turnover radius) in arcsec, ``v_flat`` and the return
value in km/s -- see ``CLAUDE.md`` for the ``Vlos`` geometry these feed into.
All three models satisfy ``Vrot(0) = 0`` and are monotonically non-decreasing.
"""

from __future__ import annotations

import numpy as np


def arctan(r: float | np.ndarray, v_flat: float, r_t: float) -> np.ndarray:
    """``Vrot(r) = (2/pi) * v_flat * arctan(r / r_t)``."""
    r = np.asarray(r, dtype=float)
    return (2.0 / np.pi) * v_flat * np.arctan(r / r_t)


def tanh(r: float | np.ndarray, v_flat: float, r_t: float) -> np.ndarray:
    """``Vrot(r) = v_flat * tanh(r / r_t)``."""
    r = np.asarray(r, dtype=float)
    return v_flat * np.tanh(r / r_t)


def flat_turnover(r: float | np.ndarray, v_flat: float, r_t: float) -> np.ndarray:
    """Linear rise to ``v_flat`` at ``r=r_t``, then flat: ``v_flat * clip(r/r_t, 0, 1)``."""
    r = np.asarray(r, dtype=float)
    return v_flat * np.clip(r / r_t, 0.0, 1.0)


ROTATION_CURVES = {
    "arctan": arctan,
    "tanh": tanh,
    "flat_turnover": flat_turnover,
}
