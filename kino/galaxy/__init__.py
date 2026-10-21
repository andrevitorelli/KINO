"""Intrinsic disc galaxy model.

Surface brightness, rotation curve, and line-of-sight velocity field for an
unlensed disc galaxy. Positions in arcsec, velocities in km/s, angles in
degrees — see ``CLAUDE.md`` for the full convention list.

``GalaxyModel`` (``kino.galaxy.protocol``) formalizes the structural
contract ``DiscGalaxy`` satisfies and any future pluggable model (e.g. a
tidally-disturbed or warped disc) would need to satisfy too — see that
module's docstring for what's already model-agnostic (the forward model)
and what's still ``DiscGalaxy``-specific (the inference/mocks layers).
"""

from kino.galaxy.disc import DiscGalaxy
from kino.galaxy.protocol import GalaxyModel

__all__ = ["DiscGalaxy", "GalaxyModel"]
