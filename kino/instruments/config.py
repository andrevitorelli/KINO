"""Shared YAML config parsing for instrument backends.

Every physical quantity in a config is a string ``astropy.units.Quantity``
can parse, e.g. ``"0.0125 arcsec"``, ``"25.4 m"``, ``"900 s"``. See
``kino/config/gmtifs.yaml`` and ``kino/config/gmacs.yaml`` for worked
examples.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np

from kino.instruments.base import LSF, PSF, Detector, Throughput

_OPTICAL_ABERRATION_KEYS = (
    "defocus",
    "astig1",
    "astig2",
    "coma1",
    "coma2",
    "trefoil1",
    "trefoil2",
    "spher",
)


def parse_quantity(value: str | u.Quantity) -> u.Quantity:
    """Parse a ``"<value> <unit>"`` config string into a Quantity."""
    return u.Quantity(value) if isinstance(value, str) else value


def psf_from_config(cfg: dict) -> PSF:
    aberrations = {k: float(cfg[k]) for k in _OPTICAL_ABERRATION_KEYS if k in cfg}
    return PSF(seeing_fwhm=parse_quantity(cfg["seeing_fwhm"]), **aberrations)


def lsf_from_config(cfg: dict) -> LSF:
    return LSF(**{k: (parse_quantity(v) if k == "fwhm" else v) for k, v in cfg.items()})


def throughput_from_config(cfg: dict) -> Throughput:
    return Throughput(
        wavelength=u.Quantity([parse_quantity(v) for v in cfg["wavelength"]]),
        transmission=np.asarray(cfg["transmission"], dtype=float),
    )


def detector_from_config(cfg: dict) -> Detector:
    return Detector(
        read_noise=parse_quantity(cfg["read_noise"]),
        dark_current=parse_quantity(cfg["dark_current"]),
    )
