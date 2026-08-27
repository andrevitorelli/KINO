"""Reduced-shear transformation of the image plane.

Applies the shear jointly to the surface brightness and the velocity field.
Convention: reduced shear ``g = g1 + i*g2``; ellipticity/distortion
``eps = (a-b)/(a+b)``. See ``CLAUDE.md`` before changing any sign or
convention, and ``kino.lensing.shear``'s module docstring for the full
derivation of the matrix and ellipticity-composition conventions.
"""

from kino.lensing.scene import LensedScene, apply_shear

__all__ = ["LensedScene", "apply_shear"]
