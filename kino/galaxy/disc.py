"""Intrinsic disc galaxy: surface brightness, rotation curve, and Vlos field.

Implements ``README.md`` Phase 1. Positions in arcsec, velocities in km/s,
angles in degrees (``inc``, ``pa``) -- see ``CLAUDE.md`` for the full
convention list. The position-angle deprojection here uses the same
East-of-North rotation sense as ``kino/io/kinms.py`` (see that module's
docstring for the derivation): a ``DiscGalaxy`` and a KinMS cube built with
the same ``(inc, pa)`` put the kinematic major axis in the same place and
agree on which end is redshifted -- pinned by
``tests/galaxy/test_disc.py::test_pa_zero_redshifted_side_matches_kinms_convention``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kino.galaxy.profiles import sersic_effective_intensity, sersic_profile
from kino.galaxy.rotation_curve import ROTATION_CURVES

_EDGE_ON_COS_I_TOL = 1e-9


@dataclass
class DiscGalaxy:
    """An intrinsic (unlensed) rotating disc galaxy.

    Parameters
    ----------
    r_e : float
        Sersic effective radius, arcsec.
    inc : float
        Inclination, degrees; 0 = face-on, 90 = edge-on. Must be in [0, 90].
    pa : float
        Position angle of the disc major axis, degrees East of North.
    v_flat : float
        Asymptotic circular velocity, km/s.
    z : float, optional
        Redshift.
    sersic_n : float, optional
        Sersic index (1 = exponential disc, the default).
    flux : float, optional
        Total integrated flux, arbitrary units.
    rotation_curve : {"arctan", "tanh", "flat_turnover"}, optional
        Rotation-curve shape; see :mod:`kino.galaxy.rotation_curve`.
    r_t : float or None, optional
        Rotation-curve turnover radius, arcsec. Defaults to ``r_e``.
    x0, y0 : float, optional
        Centroid, arcsec.
    v_sys : float, optional
        Systemic velocity, km/s.
    sigma0, sigma_grad : float, optional
        Intrinsic velocity dispersion at r=0 (km/s) and its gradient with
        disc-plane radius (km/s/arcsec); the dispersion floor is 0.
    """

    r_e: float
    inc: float
    pa: float
    v_flat: float
    z: float = 0.0
    sersic_n: float = 1.0
    flux: float = 1.0
    rotation_curve: str = "arctan"
    r_t: float | None = None
    x0: float = 0.0
    y0: float = 0.0
    v_sys: float = 0.0
    sigma0: float = 0.0
    sigma_grad: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.inc <= 90.0):
            raise ValueError(f"inc must be in [0, 90] degrees, got {self.inc}")
        if self.r_e <= 0:
            raise ValueError(f"r_e must be positive, got {self.r_e}")
        if self.sersic_n <= 0:
            raise ValueError(f"sersic_n must be positive, got {self.sersic_n}")
        if self.rotation_curve not in ROTATION_CURVES:
            raise ValueError(
                f"rotation_curve must be one of {sorted(ROTATION_CURVES)}, "
                f"got {self.rotation_curve!r}"
            )
        if self.r_t is None:
            self.r_t = self.r_e
        if self.r_t <= 0:
            raise ValueError(f"r_t must be positive, got {self.r_t}")

    @property
    def axis_ratio(self) -> float:
        """Projected minor/major axis ratio ``q = cos(inc)`` for a thin disc."""
        return float(np.cos(np.radians(self.inc)))

    @property
    def ellipticity(self) -> float:
        """Distortion ellipticity ``eps = (a-b)/(a+b) = (1-q)/(1+q)`` from inclination."""
        q = self.axis_ratio
        return (1.0 - q) / (1.0 + q)

    def _major_minor(self, x, y):
        """Sky-plane offsets from the centroid, rotated onto the disc's
        major/minor axes (East-of-North ``pa``, see module docstring)."""
        x = np.asarray(x, dtype=float) - self.x0
        y = np.asarray(y, dtype=float) - self.y0
        pa_rad = np.radians(self.pa)
        x_major = x * np.sin(pa_rad) + y * np.cos(pa_rad)
        x_minor = x * np.cos(pa_rad) - y * np.sin(pa_rad)
        return x_major, x_minor

    def _disc_plane(self, x, y):
        """Deprojected disc-plane radius and cos(theta), theta measured from
        the major axis. See the edge-on note in the class docstring: at
        inc=90 the minor-axis deprojection is singular everywhere except
        exactly on the major axis (x_minor=0), which is the only line with
        nonzero flux for a zero-thickness disc seen exactly edge-on.
        """
        x_major, x_minor = self._major_minor(x, y)
        cos_i = self.axis_ratio
        if cos_i < _EDGE_ON_COS_I_TOL:
            y_disc = np.where(x_minor == 0.0, 0.0, np.inf)
        else:
            y_disc = x_minor / cos_i
        r_disc = np.hypot(x_major, y_disc)
        cos_theta = np.divide(x_major, r_disc, out=np.zeros_like(r_disc), where=r_disc > 0)
        return r_disc, cos_theta

    def surface_brightness(self, x, y) -> np.ndarray:
        """Surface brightness at sky-plane position ``(x, y)``, arcsec."""
        r_disc, _ = self._disc_plane(x, y)
        i_e = sersic_effective_intensity(
            self.flux, self.r_e, n=self.sersic_n, axis_ratio=self.axis_ratio
        )
        return sersic_profile(r_disc, self.r_e, n=self.sersic_n, amplitude=i_e)

    def vlos(self, x, y) -> np.ndarray:
        """Line-of-sight velocity at ``(x, y)``, arcsec -> km/s.

        ``Vlos = v_sys + Vrot(r) * sin(inc) * cos(theta)``, theta measured
        from the major axis in the disc plane (``CLAUDE.md`` convention).
        Positive = redshifted (receding).
        """
        r_disc, cos_theta = self._disc_plane(x, y)
        v_rot = ROTATION_CURVES[self.rotation_curve](r_disc, self.v_flat, self.r_t)
        sin_i = np.sin(np.radians(self.inc))
        return self.v_sys + v_rot * sin_i * cos_theta

    def velocity_dispersion(self, x, y) -> np.ndarray:
        """Intrinsic velocity dispersion at ``(x, y)``, arcsec -> km/s."""
        r_disc, _ = self._disc_plane(x, y)
        return np.clip(self.sigma0 + self.sigma_grad * r_disc, 0.0, None)
