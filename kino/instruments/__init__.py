"""Instrument backends.

Defines the common ``Instrument.observe(scene) -> MockData`` API and the IFU
and slit/grism spectrograph implementations, including PSF, LSF, sampling,
throughput, and noise handling.
"""
