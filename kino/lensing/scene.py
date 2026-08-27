"""Apply reduced shear to a galaxy scene: image and velocity field jointly.

Implements ``README.md`` Phase 2. See ``kino.lensing.shear`` for the matrix
and ellipticity-composition conventions this relies on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kino.lensing.shear import shear_ellipticity


@dataclass
class LensedScene:
    """A galaxy observed through reduced shear ``(g1, g2)``.

    Wraps any object exposing ``surface_brightness(x, y)``, ``vlos(x, y)``,
    ``velocity_dispersion(x, y)``, ``x0``, ``y0`` (e.g. a
    :class:`kino.galaxy.DiscGalaxy`) and re-evaluates it at the delensed
    coordinate for each observed sky position, per ``kino.lensing.shear``'s
    ``beta = A @ theta`` convention -- **both** the surface brightness and
    the velocity field go through the identical spatial remapping (see
    ``CLAUDE.md``: "Shear applied to velocity, not just image"). The shear
    is centred on the wrapped galaxy's own centroid ``(x0, y0)``, i.e. the
    galaxy is assumed to be its own local lens centre (KINO does not model
    convergence/magnification or a separate external lens position).
    """

    galaxy: object
    g1: float
    g2: float

    def __post_init__(self) -> None:
        if self.g1**2 + self.g2**2 >= 1.0:
            raise ValueError(f"|g| must be < 1 (reduced shear), got g1={self.g1}, g2={self.g2}")

    def _delens(self, x, y):
        dx = np.asarray(x, dtype=float) - self.galaxy.x0
        dy = np.asarray(y, dtype=float) - self.galaxy.y0
        beta_dx = (1.0 - self.g1) * dx - self.g2 * dy
        beta_dy = -self.g2 * dx + (1.0 + self.g1) * dy
        return self.galaxy.x0 + beta_dx, self.galaxy.y0 + beta_dy

    def surface_brightness(self, x, y) -> np.ndarray:
        beta_x, beta_y = self._delens(x, y)
        return self.galaxy.surface_brightness(beta_x, beta_y)

    def vlos(self, x, y) -> np.ndarray:
        beta_x, beta_y = self._delens(x, y)
        return self.galaxy.vlos(beta_x, beta_y)

    def velocity_dispersion(self, x, y) -> np.ndarray:
        beta_x, beta_y = self._delens(x, y)
        return self.galaxy.velocity_dispersion(beta_x, beta_y)

    @property
    def photometric_ellipticity(self) -> tuple[float, float]:
        """Analytic (magnitude, PA-East-of-North-degrees) of the lensed image.

        Requires the wrapped galaxy to expose ``ellipticity`` (magnitude)
        and ``pa`` (degrees, East of North). Converts to/from
        ``kino.lensing.shear``'s standard-math-angle phase convention
        (``phi = 90deg - pa``, see that module's docstring), applies
        :func:`kino.lensing.shear.shear_ellipticity`, and converts back.
        """
        pa_rad = np.radians(self.galaxy.pa)
        phi_rad = np.pi / 2.0 - pa_rad
        eps_int = self.galaxy.ellipticity * np.exp(2j * phi_rad)
        eps_obs = shear_ellipticity(eps_int, self.g1, self.g2)
        magnitude = abs(eps_obs)
        phi_obs_rad = np.angle(eps_obs) / 2.0
        pa_obs_deg = np.degrees(np.pi / 2.0 - phi_obs_rad) % 180.0
        return magnitude, pa_obs_deg


def apply_shear(galaxy, g1: float, g2: float) -> LensedScene:
    """Apply reduced shear ``(g1, g2)`` to ``galaxy``'s image and velocity field."""
    return LensedScene(galaxy=galaxy, g1=g1, g2=g2)
