"""Instrument building blocks: PSF, LSF, throughput, detector, and the base API.

Implements the first half of ``README.md`` Phase 3: "Define the ``Instrument``
base API: ``observe(scene) -> MockData``, plus PSF, LSF, spatial sampling,
throughput, and noise hooks." Every dataclass here is a reusable parameter
block shared by any concrete backend (``kino.instruments.ifu.IFU`` today; a
future slit/grism backend reuses the same PSF/LSF/Throughput/Detector types).

Units and conventions follow ``CLAUDE.md``. Two simplifying assumptions,
common to first-generation instrument forecasting tools, apply everywhere in
this module and are documented at the point of use rather than repeated:

- The spatial PSF is **achromatic** (one kernel for the whole band). Real
  diffraction-limited PSFs grow with wavelength; this is a known limitation
  to revisit if a forecast needs wavelength-dependent PSF fidelity.
- The LSF is **Gaussian**, so instrumental and intrinsic (turbulent) line
  broadening combine in quadrature (``kino.instruments.ifu.IFU.observe``)
  rather than by explicit numerical convolution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import astropy.units as u
import galsim
import numpy as np
from astropy.constants import c as C_LIGHT
from astropy.constants import h as PLANCK_H
from scipy.special import erf

from kino.io.base import Datacube

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
_SQRT2 = np.sqrt(2.0)


def photon_energy(wavelength: u.Quantity) -> u.Quantity:
    """``E = h*c/wavelength``, in erg."""
    return (PLANCK_H * C_LIGHT / wavelength).to(u.erg)


@dataclass
class PSF:
    """Spatial PSF: aperture diffraction (``galsim.OpticalPSF``) convolved
    with atmospheric seeing (``galsim.Kolmogorov``).

    Achromatic within a cube: :meth:`kernel` is evaluated once per
    :meth:`~kino.instruments.ifu.IFU.observe` call, at the band's central
    wavelength (see ``kino.instruments.ifu``'s module docstring) -- real
    diffraction- and seeing-limited PSFs both grow with wavelength; this is
    a known simplification to revisit for a forecast needing per-channel
    PSF fidelity.

    Parameters
    ----------
    seeing_fwhm : Quantity (angle)
        Atmospheric (Kolmogorov) seeing FWHM. ``0`` skips the atmospheric
        term entirely, i.e. a diffraction-limited system (space, or an
        AO-corrected core).
    defocus, astig1, astig2, coma1, coma2, trefoil1, trefoil2, spher : float, optional
        Zernike wavefront-error coefficients (waves), passed directly to
        ``galsim.OpticalPSF``. Fiducial default ``0.0`` (diffraction-limited,
        unaberrated optics) -- override with as-built values if known for
        the telescope being forecast.
    """

    seeing_fwhm: u.Quantity
    defocus: float = 0.0
    astig1: float = 0.0
    astig2: float = 0.0
    coma1: float = 0.0
    coma2: float = 0.0
    trefoil1: float = 0.0
    trefoil2: float = 0.0
    spher: float = 0.0

    def __post_init__(self) -> None:
        if self.seeing_fwhm < 0:
            raise ValueError("seeing_fwhm must be non-negative")
        if not self.seeing_fwhm.unit.is_equivalent(u.arcsec):
            raise u.UnitsError("PSF.seeing_fwhm must be an angle")

    def kernel(
        self,
        pixel_scale: u.Quantity,
        wavelength: u.Quantity,
        aperture_diameter: u.Quantity,
        obscuration: float = 0.0,
    ) -> np.ndarray:
        """Normalised (sum=1) 2-D kernel, rasterised at ``pixel_scale``.

        ``wavelength`` and ``aperture_diameter`` set the diffraction scale;
        ``obscuration`` is the fraction of the aperture blocked by a
        secondary mirror (pass the instrument's own value for consistency
        with the collecting area used for the photon budget).
        """
        optical = galsim.OpticalPSF(
            lam=wavelength.to_value(u.nm),
            diam=aperture_diameter.to_value(u.m),
            obscuration=obscuration,
            defocus=self.defocus,
            astig1=self.astig1,
            astig2=self.astig2,
            coma1=self.coma1,
            coma2=self.coma2,
            trefoil1=self.trefoil1,
            trefoil2=self.trefoil2,
            spher=self.spher,
        )
        profile = optical
        if self.seeing_fwhm > 0:
            atmosphere = galsim.Kolmogorov(fwhm=self.seeing_fwhm.to_value(u.arcsec))
            profile = galsim.Convolve([optical, atmosphere])
        image = profile.drawImage(scale=pixel_scale.to_value(u.arcsec))
        kernel = image.array.astype(float)
        return kernel / kernel.sum()


@dataclass
class LSF:
    """Spectral line-spread function (assumed Gaussian, see module docstring).

    Exactly one of ``resolving_power`` or ``fwhm`` must be set.

    Parameters
    ----------
    resolving_power : float or None, optional
        ``R = wavelength / FWHM_wavelength``, assumed constant across the
        band (the standard first-order approximation for a grating/prism
        spectrograph).
    fwhm : Quantity (wavelength), optional
        Fixed-wavelength-width alternative to ``resolving_power``, for
        instruments better described by a constant ``d(lambda)`` per
        resolution element than a constant ``R``.
    """

    resolving_power: float | None = None
    fwhm: u.Quantity | None = None

    def __post_init__(self) -> None:
        if (self.resolving_power is None) == (self.fwhm is None):
            raise ValueError("LSF requires exactly one of resolving_power or fwhm")
        if self.resolving_power is not None and self.resolving_power <= 0:
            raise ValueError(f"resolving_power must be positive, got {self.resolving_power}")
        if self.fwhm is not None:
            if self.fwhm <= 0:
                raise ValueError("LSF.fwhm must be positive")
            if not self.fwhm.unit.is_equivalent(u.m):
                raise u.UnitsError("LSF.fwhm must be a wavelength")

    def sigma(self, wavelength: u.Quantity) -> u.Quantity:
        """Gaussian sigma at ``wavelength`` (broadcastable), in ``wavelength``'s unit."""
        fwhm = wavelength / self.resolving_power if self.resolving_power is not None else self.fwhm
        return fwhm * _FWHM_TO_SIGMA


@dataclass
class Throughput:
    """Wavelength-dependent end-to-end throughput (optics x coatings x detector QE).

    Does **not** include the telescope's collecting area (``IFU.aperture_diameter``
    accounts for that separately) or, for a ground-based instrument,
    atmospheric transmission -- fold an atmospheric curve into ``transmission``
    directly if you need one, by multiplying it in before constructing this
    object.

    Parameters
    ----------
    wavelength : Quantity (wavelength), shape (n,)
        Strictly increasing wavelength grid the curve is tabulated on.
    transmission : numpy.ndarray, shape (n,)
        Fraction in [0, 1] at each ``wavelength``. Linearly interpolated;
        zero outside the tabulated range.
    """

    wavelength: u.Quantity
    transmission: np.ndarray

    def __post_init__(self) -> None:
        self.transmission = np.asarray(self.transmission, dtype=float)
        if self.wavelength.shape != self.transmission.shape:
            raise ValueError("wavelength and transmission must have the same shape")
        if np.any(np.diff(self.wavelength.value) <= 0):
            raise ValueError("wavelength must be strictly increasing")
        if np.any(self.transmission < 0) or np.any(self.transmission > 1):
            raise ValueError("transmission must be in [0, 1]")

    def __call__(self, wavelength: u.Quantity) -> np.ndarray:
        wl = wavelength.to_value(self.wavelength.unit)
        return np.interp(
            wl,
            self.wavelength.to_value(self.wavelength.unit),
            self.transmission,
            left=0.0,
            right=0.0,
        )


@dataclass
class Detector:
    """Detector noise properties.

    Parameters
    ----------
    read_noise : Quantity (electron)
        Per-exposure read noise, per pixel.
    dark_current : Quantity (electron / s)
        Dark-current rate, per pixel.
    """

    read_noise: u.Quantity
    dark_current: u.Quantity = 0.0 * u.electron / u.s

    def __post_init__(self) -> None:
        if self.read_noise < 0:
            raise ValueError("read_noise must be non-negative")
        if self.dark_current < 0:
            raise ValueError("dark_current must be non-negative")


def doppler_center(vlos: u.Quantity, line_wavelength: u.Quantity, redshift: float) -> u.Quantity:
    """Observed-frame wavelength of a line at rest wavelength ``line_wavelength``,
    redshift ``redshift``, and line-of-sight velocity ``vlos`` (``CLAUDE.md``:
    positive = redshifted/receding)."""
    c_kms = C_LIGHT.to(u.km / u.s)
    return (
        line_wavelength
        * (1.0 + redshift)
        * (1.0 + (vlos / c_kms).to_value(u.dimensionless_unscaled))
    )


def doppler_sigma(sigma_v: u.Quantity, line_center: u.Quantity) -> u.Quantity:
    """Wavelength-domain sigma of a turbulent line of velocity dispersion
    ``sigma_v`` centred at ``line_center``."""
    c_kms = C_LIGHT.to(u.km / u.s)
    return line_center * (sigma_v / c_kms).to_value(u.dimensionless_unscaled)


def gaussian_channel_fractions(
    line_center: u.Quantity,
    intrinsic_sigma: u.Quantity,
    lsf_sigma: u.Quantity,
    wave_edges: u.Quantity,
) -> np.ndarray:
    """Fraction of a Gaussian line's flux falling in each wavelength channel.

    The line's width is ``intrinsic_sigma`` (turbulent broadening) and
    ``lsf_sigma`` (instrumental) combined in quadrature (exact for two
    Gaussians -- see ``kino.instruments.ifu``'s module docstring), with
    each channel's share found analytically via the Gaussian CDF rather
    than by discretising and convolving.

    ``line_center`` and ``intrinsic_sigma`` have some spatial shape ``S``
    (e.g. ``(ny, nx)`` for an IFU, ``(n_u,)`` along a slit); ``lsf_sigma``
    has shape ``(n_wave,)``; ``wave_edges`` has shape ``(n_wave + 1,)``.
    Returns an ``(n_wave,) + S`` array of fractions in ``[0, 1]``.
    """
    unit = line_center.unit
    pad = (1,) * line_center.ndim
    center = line_center.to_value(unit)[np.newaxis, ...]
    sigma_tot = np.sqrt(
        intrinsic_sigma.to_value(unit)[np.newaxis, ...] ** 2
        + lsf_sigma.to_value(unit).reshape((-1,) + pad) ** 2
    )
    lo = wave_edges[:-1].to_value(unit).reshape((-1,) + pad)
    hi = wave_edges[1:].to_value(unit).reshape((-1,) + pad)
    return 0.5 * (
        erf((hi - center) / (_SQRT2 * sigma_tot)) - erf((lo - center) / (_SQRT2 * sigma_tot))
    )


def photon_flux_to_electrons(
    flux: u.Quantity,
    wave_centers: u.Quantity,
    throughput: Throughput,
    effective_area: u.Quantity,
    exposure_total: u.Quantity,
) -> np.ndarray:
    """Physical flux per wavelength channel -> detected electrons.

    ``flux`` (energy / time / area) has shape ``(n_wave,) + S``;
    ``wave_centers`` has shape ``(n_wave,)``. Returns a plain ``(n_wave,) + S``
    array (electron counts, unitless).
    """
    pad = (1,) * (flux.ndim - 1)
    e_photon = photon_energy(wave_centers).reshape((-1,) + pad)
    t = throughput(wave_centers).reshape((-1,) + pad)
    return (flux / e_photon * t * effective_area * exposure_total).to_value(
        u.dimensionless_unscaled
    )


def apply_shot_and_read_noise(
    expected_electron: np.ndarray,
    read_noise: u.Quantity,
    n_exposures: int,
    rng: np.random.Generator | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Poisson shot noise (on ``expected_electron``, e.g. source + sky + dark)
    plus Gaussian read noise (``sqrt(n_exposures) * read_noise``).

    Returns ``(counts, variance)``; ``rng=None`` returns ``expected_electron``
    itself as ``counts`` (noiseless) with ``variance`` still populated.
    """
    read_sigma = read_noise.to_value(u.electron) * np.sqrt(n_exposures)
    variance = expected_electron + read_sigma**2
    if rng is None:
        return expected_electron, variance
    counts = rng.poisson(np.clip(expected_electron, 0.0, None)).astype(float)
    counts = counts + rng.normal(0.0, read_sigma, size=counts.shape)
    return counts, variance


class Instrument(ABC):
    """Common API: turn a lensed (or intrinsic) galaxy scene into mock data."""

    @abstractmethod
    def observe(self, scene, *, rng: np.random.Generator | None = None, **kwargs) -> Datacube:
        """Render ``scene`` through this instrument.

        ``rng=None`` returns the noiseless expected-count cube (``variance``
        is still populated, for computing S/N without drawing a random
        realisation); pass a seeded ``numpy.random.Generator`` to draw one
        noisy realisation (``CLAUDE.md``: no global RNG use).
        """
        raise NotImplementedError
