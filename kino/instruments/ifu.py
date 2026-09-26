"""IFU instrument backend: spaxel grid, PSF/LSF, throughput, and noise.

Implements ``README.md`` Phase 3's IFU backend and noise-model bullets.

Flux calibration
-----------------
A ``kino.galaxy.DiscGalaxy`` (or ``kino.lensing.scene.LensedScene``)
``surface_brightness`` is in the galaxy's own arbitrary flux units (see
``kino.galaxy.disc``), not a physical flux. :meth:`IFU.observe` takes an
explicit physical ``line_flux`` (the emission line's total, **intrinsic**
flux, e.g. in ``erg/s/cm**2``) and rescales the (arbitrary-unit) rendered
shape by ``line_flux / galaxy.flux``.

Because a :class:`~kino.lensing.scene.LensedScene` is evaluated by sampling
the *unlensed* galaxy's surface brightness at each observed (delensed) sky
position, the total flux integrated over the observed image already comes
out magnified by ``kino.lensing.shear.magnification`` relative to
``galaxy.flux`` -- this is the same effect documented in ``README.md``
Phase 2 ("verified against the actual rendered/integrated flux ratio of a
lensed vs. unlensed image"). So ``line_flux`` here is the source's
intrinsic, unlensed brightness; the mock cube's observed flux is the
(correctly) magnified value. Do not re-apply magnification separately.

Rendering pipeline (``CLAUDE.md``: "PSF/LSF before sampling")
---------------------------------------------------------------
1. Evaluate the scene on a spatially oversampled grid.
2. Build each oversampled spaxel's line profile in wavelength: centred at
   the redshifted + Doppler-shifted wavelength, with a Gaussian width that
   is the intrinsic (turbulent) and instrumental (LSF) widths combined in
   quadrature (exact for two Gaussians), then partition that Gaussian's
   flux across wavelength channels analytically (via the Gaussian CDF --
   ``kino.instruments.base.gaussian_channel_fractions``) rather than by
   discretising and convolving.
3. Convolve each wavelength channel's oversampled spatial image with the
   PSF kernel, then bin down to the native spaxel grid.
4. Convert physical flux to detected electrons via throughput, aperture,
   and exposure time (``kino.instruments.base.photon_flux_to_electrons``),
   then add sky, dark, and read noise
   (``kino.instruments.base.apply_shot_and_read_noise``).

This same pipeline -- steps 2 and 4 shared verbatim -- underlies
``kino.instruments.slit.SlitSpectrograph``; only the spatial grid (step 1)
and the spatial reduction (step 3) differ (a 2-D spaxel grid here, a
slit-aligned grid truncated to the slit's width there).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
import yaml
from scipy.signal import fftconvolve

from kino.instruments.base import (
    LSF,
    PSF,
    Detector,
    Instrument,
    Throughput,
    apply_shot_and_read_noise,
    doppler_center,
    doppler_sigma,
    gaussian_channel_fractions,
    photon_flux_to_electrons,
)
from kino.instruments.config import (
    detector_from_config,
    lsf_from_config,
    parse_quantity,
    psf_from_config,
    throughput_from_config,
)
from kino.io.base import Datacube

_SPATIAL_OVERSAMPLE = 4


@dataclass
class IFU(Instrument):
    """An integral-field-unit spectrograph.

    Parameters
    ----------
    pixel_scale : Quantity (angle)
        Spaxel size on sky.
    nx, ny : int
        Spatial array size, in spaxels.
    wavelength_min, wavelength_max : Quantity (wavelength)
        Observed-frame spectral coverage.
    n_wave : int
        Number of spectral channels across ``[wavelength_min, wavelength_max]``.
    psf : PSF
        Spatial PSF (assumed achromatic, see module docstring).
    lsf : LSF
        Spectral LSF (assumed Gaussian).
    throughput : Throughput
        End-to-end optics x detector-QE curve (excludes aperture area).
    aperture_diameter : Quantity (length)
        Telescope aperture.
    detector : Detector
        Read noise and dark current.
    exposure_time : Quantity (time)
        Single-exposure integration time.
    obstruction_fraction : float, optional
        Fraction of the aperture blocked (e.g. by a secondary mirror), in
        [0, 1).
    n_exposures : int, optional
        Number of exposures coadded; scales total integration time and read
        noise (``sqrt(n_exposures) * read_noise``).
    sky_electron_rate : Quantity (electron / s), scalar or shape (n_wave,)
        Sky background **already converted to detected electron rate per
        spaxel** (as you would get from an observatory's own exposure-time
        calculator) -- not a sky surface brightness. This sidesteps needing
        a second physical flux-to-electron pipeline (atmosphere/zodiacal
        emission, its own throughput) identical in kind to the source's but
        conceptually separate; fold any such conversion in yourself and pass
        the resulting rate here. Default 0 (e.g. a space telescope, or a
        first pass ignoring sky background).
    """

    pixel_scale: u.Quantity
    nx: int
    ny: int
    wavelength_min: u.Quantity
    wavelength_max: u.Quantity
    n_wave: int
    psf: PSF
    lsf: LSF
    throughput: Throughput
    aperture_diameter: u.Quantity
    detector: Detector
    exposure_time: u.Quantity
    obstruction_fraction: float = 0.0
    n_exposures: int = 1
    sky_electron_rate: u.Quantity = 0.0 * u.electron / u.s

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("nx and ny must be positive")
        if self.n_wave <= 0:
            raise ValueError("n_wave must be positive")
        if self.wavelength_max <= self.wavelength_min:
            raise ValueError("wavelength_max must exceed wavelength_min")
        if not (0.0 <= self.obstruction_fraction < 1.0):
            raise ValueError("obstruction_fraction must be in [0, 1)")
        if self.n_exposures <= 0:
            raise ValueError("n_exposures must be positive")
        if self.aperture_diameter <= 0:
            raise ValueError("aperture_diameter must be positive")
        if self.exposure_time <= 0:
            raise ValueError("exposure_time must be positive")

    @property
    def effective_area(self) -> u.Quantity:
        """Collecting area, accounting for ``obstruction_fraction``."""
        return np.pi * (self.aperture_diameter / 2.0) ** 2 * (1.0 - self.obstruction_fraction)

    @property
    def band_center(self) -> u.Quantity:
        return 0.5 * (self.wavelength_min + self.wavelength_max)

    def baseline_electrons(self) -> np.ndarray:
        """Expected sky + dark electrons per channel per spaxel over the whole
        exposure, shape ``(n_wave, 1, 1)`` -- the *known* baseline that
        ``observe`` adds under the line (before noise). Pass it as
        ``subtract_background`` to :func:`kino.analysis.moments.compute_moment_maps`
        to reduce a cube without estimating the background from the data.
        """
        exposure_total = self.exposure_time * self.n_exposures
        sky = np.broadcast_to(
            (self.sky_electron_rate * exposure_total).to_value(u.electron), (self.n_wave,)
        )
        dark = (self.detector.dark_current * exposure_total).to_value(u.electron)
        return (sky + dark)[:, np.newaxis, np.newaxis]

    def _spatial_grid(self):
        ov = _SPATIAL_OVERSAMPLE
        pix_over = (self.pixel_scale / ov).to(u.arcsec)
        nx_over, ny_over = self.nx * ov, self.ny * ov
        x = (np.arange(nx_over) - (nx_over - 1) / 2.0) * pix_over
        y = (np.arange(ny_over) - (ny_over - 1) / 2.0) * pix_over
        xx, yy = np.meshgrid(x, y, indexing="xy")
        return xx, yy, pix_over

    def observe(
        self,
        scene,
        *,
        line_flux: u.Quantity,
        line_wavelength: u.Quantity,
        rng: np.random.Generator | None = None,
    ) -> Datacube:
        """Render ``scene``'s emission line through this IFU.

        Parameters
        ----------
        scene : object
            A :class:`~kino.lensing.scene.LensedScene` (or an unlensed
            ``DiscGalaxy``): must expose ``surface_brightness(x, y)``,
            ``vlos(x, y)``, ``velocity_dispersion(x, y)`` and, on itself or
            a wrapped ``.galaxy``, ``flux`` and ``z``.
        line_flux : Quantity (energy / time / area)
            Intrinsic (unlensed), total flux of the emission line -- see
            the module docstring for how magnification enters.
        line_wavelength : Quantity (wavelength)
            Rest-frame wavelength of the line.
        rng : numpy.random.Generator or None, optional
            ``None`` returns the noiseless expected-count cube; pass a
            seeded generator for one noisy realisation.

        Returns
        -------
        Datacube
            ``flux`` (and ``variance``) in electrons, ``spectral_axis`` in
            wavelength.
        """
        galaxy = getattr(scene, "galaxy", scene)
        z = galaxy.z
        flux_norm = galaxy.flux

        xx, yy, pix_over = self._spatial_grid()
        sb = scene.surface_brightness(xx, yy)
        vlos = scene.vlos(xx, yy) * (u.km / u.s)
        sigma_v = scene.velocity_dispersion(xx, yy) * (u.km / u.s)

        pixel_area_over = pix_over**2
        flux_per_pixel = sb * pixel_area_over.to_value(u.arcsec**2) * (line_flux / flux_norm)

        line_center = doppler_center(vlos, line_wavelength, z)
        sigma_intrinsic = doppler_sigma(sigma_v, line_center)

        wave_edges = np.linspace(self.wavelength_min, self.wavelength_max, self.n_wave + 1)
        wave_centers = 0.5 * (wave_edges[:-1] + wave_edges[1:])
        lsf_sigma = self.lsf.sigma(wave_centers)

        channel_fraction = gaussian_channel_fractions(
            line_center, sigma_intrinsic, lsf_sigma, wave_edges
        )
        flux_cube_over = (
            flux_per_pixel.to_value(line_flux.unit)[np.newaxis, :, :] * channel_fraction
        )

        kernel = self.psf.kernel(
            pix_over,
            wavelength=self.band_center,
            aperture_diameter=self.aperture_diameter,
            obscuration=self.obstruction_fraction,
        )
        flux_cube_over = np.stack(
            [fftconvolve(flux_cube_over[k], kernel, mode="same") for k in range(self.n_wave)]
        )

        ov = _SPATIAL_OVERSAMPLE
        flux_native = flux_cube_over.reshape(self.n_wave, self.ny, ov, self.nx, ov).sum(axis=(2, 4))
        flux_native = flux_native * line_flux.unit

        exposure_total = self.exposure_time * self.n_exposures
        signal_electron = photon_flux_to_electrons(
            flux_native, wave_centers, self.throughput, self.effective_area, exposure_total
        )
        baseline = self.baseline_electrons()

        expected = signal_electron + baseline
        counts, variance = apply_shot_and_read_noise(
            expected, self.detector.read_noise, self.n_exposures, rng
        )

        return Datacube(
            flux=counts,
            spectral_axis=wave_centers,
            spectral_kind="wavelength",
            pixel_scale=self.pixel_scale.to(u.arcsec),
            redshift=z,
            flux_unit=u.electron,
            variance=variance,
            meta={
                "source": "kino.instruments.ifu.IFU.observe",
                "line_wavelength_rest": line_wavelength,
                "line_flux_intrinsic": line_flux,
                "noiseless": rng is None,
            },
        )

    @classmethod
    def from_config(cls, path: str | Path) -> IFU:
        """Build an ``IFU`` from a YAML config (``kino/config/*.yaml``).

        See ``kino.instruments.config`` for the quantity-string format and
        ``kino/config/gmtifs.yaml`` for a fully worked example.
        """
        with open(path) as f:
            cfg = yaml.safe_load(f)

        return cls(
            pixel_scale=parse_quantity(cfg["pixel_scale"]),
            nx=cfg["nx"],
            ny=cfg["ny"],
            wavelength_min=parse_quantity(cfg["wavelength_min"]),
            wavelength_max=parse_quantity(cfg["wavelength_max"]),
            n_wave=cfg["n_wave"],
            psf=psf_from_config(cfg["psf"]),
            lsf=lsf_from_config(cfg["lsf"]),
            throughput=throughput_from_config(cfg["throughput"]),
            aperture_diameter=parse_quantity(cfg["aperture_diameter"]),
            detector=detector_from_config(cfg["detector"]),
            exposure_time=parse_quantity(cfg["exposure_time"]),
            obstruction_fraction=cfg.get("obstruction_fraction", 0.0),
            n_exposures=cfg.get("n_exposures", 1),
            sky_electron_rate=parse_quantity(cfg.get("sky_electron_rate", "0 electron / s")),
        )
