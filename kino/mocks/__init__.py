"""Mock data assembly.

Combines a galaxy, an applied shear, and an instrument backend into mock
data products (cubes, spectra, images) with provenance
(:func:`simulate_observation`), plus batch generation from a YAML
catalogue of galaxies (:func:`generate_batch`) -- README.md Phase 4.
"""

from kino.mocks.catalogue import generate_batch
from kino.mocks.single import simulate_observation

__all__ = ["generate_batch", "simulate_observation"]
