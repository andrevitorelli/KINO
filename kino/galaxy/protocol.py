"""Structural protocol a pluggable intrinsic galaxy model must satisfy.

Formalizes a contract `kino.lensing.scene.LensedScene`/`apply_shear` already
rely on implicitly (see that module's docstring): any object exposing
`surface_brightness(x, y)`, `vlos(x, y)`, `velocity_dispersion(x, y)`, and a
centroid `(x0, y0)` can be sheared and rendered through any KINO instrument
backend. `DiscGalaxy` is the only implementation today, but nothing in
`kino.lensing` or `kino.instruments` is specific to it -- the forward model
(galaxy -> shear -> instrument) is already model-agnostic by construction.
This module gives that existing contract a name and a place to check
against, without changing any call site's behaviour.

**What this does NOT yet do.** `kino.inference.likelihood` (`GALAXY_FIELDS`,
`_scene_from_params`), `kino.mocks.catalogue`, and `kino.analysis.injection`
all still construct `DiscGalaxy` directly (`DiscGalaxy(**galaxy_kwargs)`) and
use a `DiscGalaxy`-specific field list rather than asking an arbitrary model
for its own fields. A model implemented as a `@dataclass` already gets a
field list "for free" via `dataclasses.fields(model_cls)` -- no new
registration mechanism is needed -- so generalizing those three call sites
is mechanically straightforward once undertaken: pass the model *class*
through (`IFUCubeLikelihood(..., galaxy_model=DiscGalaxy)` instead of a
hardcoded import) and replace `GALAXY_FIELDS` with
`{f.name for f in dataclasses.fields(galaxy_model)}` at call time. This is
real, deliberately deferred future work (README.md), not started here --
KINO's priority is being a general-purpose kinematic-lensing forward model
and inference engine, not a single fixed disc-model implementation, so a
user's own (e.g. tidally-disturbed, warped, barred) model should eventually
be a drop-in `GalaxyModel`, not a fork of `DiscGalaxy`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class GalaxyModel(Protocol):
    """Structural contract for any intrinsic galaxy model KINO can shear and observe.

    A class satisfies this Protocol (no inheritance required -- Python
    checks it structurally) by defining `x0`/`y0` and the three methods
    below with the right signatures. `isinstance(obj, GalaxyModel)` works
    at runtime (`@runtime_checkable`) but only checks that the named
    attributes/methods exist, not their exact signatures or units --
    satisfy the conventions below by convention, not by the type checker.

    Attributes
    ----------
    x0, y0 : float
        Centroid, arcsec -- `kino.lensing.scene.LensedScene` shears around
        this point (`CLAUDE.md`: image-plane Cartesian `(x, y)`, arcsec,
        origin at the galaxy centroid).

    Methods
    -------
    surface_brightness(x, y) -> numpy.ndarray
        Evaluated at sky-plane position `(x, y)`, arcsec. The model's own
        arbitrary units -- physical calibration happens at
        `Instrument.observe()` time (`CLAUDE.md`: "Instrument flux
        calibration").
    vlos(x, y) -> numpy.ndarray
        Line-of-sight velocity, km/s, positive = redshifted (receding).
    velocity_dispersion(x, y) -> numpy.ndarray
        Intrinsic velocity dispersion, km/s.
    """

    x0: float
    y0: float

    def surface_brightness(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...

    def vlos(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...

    def velocity_dispersion(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...
