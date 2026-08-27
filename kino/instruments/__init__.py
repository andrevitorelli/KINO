"""Instrument backends.

Defines the common ``Instrument.observe(scene) -> MockData`` API and the IFU
and slit/grism spectrograph implementations, including PSF, LSF, sampling,
throughput, and noise handling.
"""

from kino.instruments.base import LSF, PSF, Detector, Instrument, Throughput, photon_energy
from kino.instruments.ifu import IFU

__all__ = [
    "Detector",
    "IFU",
    "Instrument",
    "LSF",
    "PSF",
    "Throughput",
    "photon_energy",
]
