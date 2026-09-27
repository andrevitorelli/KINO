"""Slit spectrograph instrument backend: slit geometry, dispersion, and a
companion broadband image.

Implements ``README.md`` Phase 3's slit/grism backend bullet. Shares the
line-profile-in-wavelength and photon/noise machinery with
``kino.instruments.ifu.IFU`` (``kino.instruments.base``'s
``gaussian_channel_fractions``, ``photon_flux_to_electrons``,
``apply_shot_and_read_noise``) -- see that module's docstring for the flux
calibration convention (physical ``line_flux`` at ``observe()`` time) and
the PSF/LSF simplifying assumptions, both unchanged here. Only the spatial
handling differs.

Slit geometry and losses
-------------------------
The slit is a rectangle on sky: length ``n_spatial * pixel_scale`` along a
position angle ``slit_pa`` (degrees, East of North -- the same convention as
``kino.galaxy.DiscGalaxy.pa``), width ``slit_width`` across it, centred at
``slit_offset`` from the scene's own origin. Light is rendered on an
oversampled grid **padded** by ``margin`` native pixels beyond the slit's
own footprint on every side (so a source's PSF-blurred wings that
straddle the slit's jaws or end caps are accounted for), PSF-convolved,
then **truncated to the slit's true aperture** -- flux landing outside
``slit_width``/``slit_length`` is discarded. This truncation, at the
oversampled (fine) pixel scale for accuracy, *is* the slit-loss physics: a
source wider than the slit, or a ``slit_offset``/``slit_pa`` misaligned
with the source's own centroid/axis, loses flux exactly as a real slit
would. No separate "slit loss fraction" parameter exists or is needed.

Companion broadband image
---------------------------
``observe()`` also returns an **un-truncated**, wavelength-collapsed image
of the same padded field (so the slit's placement relative to the source
can be checked/visualised) -- exactly the pipeline above, minus the slit's
width/length truncation and the wavelength split. It shares the
spectrum's own throughput, aperture, and (necessarily, since kino has no
separate continuum-flux calibration input yet) the same
``line_flux``-normalised source profile -- treat it as a slit-placement
visualisation aid, not a photometrically calibrated continuum image. It
shares the spectrum's ``exposure_time``/``n_exposures``; call ``observe``
twice with different values if a real acquisition frame's own (typically
much shorter) integration time matters.
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
class SlitObservation:
    """Output of :meth:`SlitSpectrograph.observe`.

    Parameters
    ----------
    spectrum : Datacube
        The dispersed 2-D spectrum, as a ``Datacube`` with a single spatial
        column (``shape = (n_wave, n_spatial, 1)``): wavelength x position
        along the slit.
    image : numpy.ndarray, shape (n_v, n_u)
        Companion broadband (wavelength-collapsed) image of the padded
        field around the slit, in electrons -- see the module docstring.
    image_pixel_scale : Quantity (angle)
        Pixel scale of ``image`` (equal to ``SlitSpectrograph.pixel_scale``).
    slit_width, slit_length : Quantity (angle)
        The slit's physical aperture, for overlaying its footprint on
        ``image``.
    slit_pa : float
        Degrees, East of North.
    slit_offset : tuple[Quantity, Quantity]
        ``(dx, dy)`` offset of the slit centre from the scene origin.
    """

    spectrum: Datacube
    image: np.ndarray
    image_pixel_scale: u.Quantity
    slit_width: u.Quantity
    slit_length: u.Quantity
    slit_pa: float
    slit_offset: tuple[u.Quantity, u.Quantity]


@dataclass
class SlitSpectrograph(Instrument):
    """A single-slit spectrograph.

    Parameters
    ----------
    slit_width : Quantity (angle)
        Cross-slit aperture; sets slit losses for a source wider than this.
    pixel_scale : Quantity (angle)
        Pixel size along the slit's length (and the truncation granularity
        for the image's cross-slit axis).
    n_spatial : int
        Pixels along the slit; ``slit_length = n_spatial * pixel_scale``.
    slit_pa : float, optional
        Slit long-axis position angle, degrees East of North (``CLAUDE.md``
        convention, same as ``kino.galaxy.DiscGalaxy.pa``).
    slit_offset : tuple[Quantity, Quantity], optional
        ``(dx, dy)`` offset of the slit centre from the scene's own origin
        -- non-zero models slit misalignment/mis-centring.
    margin : int, optional
        Native pixels of padding rendered beyond the slit's own footprint
        on every side, to capture PSF-driven flux exchange across the slit
        boundary (see module docstring). Default 10.
    wavelength_min, wavelength_max, n_wave, psf, lsf, throughput,
    aperture_diameter, obstruction_fraction, detector, exposure_time,
    n_exposures, sky_electron_rate :
        Same meaning as :class:`~kino.instruments.ifu.IFU`'s fields of the
        same name (``sky_electron_rate`` here is per along-slit pixel).
    """

    slit_width: u.Quantity
    pixel_scale: u.Quantity
    n_spatial: int
    wavelength_min: u.Quantity
    wavelength_max: u.Quantity
    n_wave: int
    psf: PSF
    lsf: LSF
    throughput: Throughput
    aperture_diameter: u.Quantity
    detector: Detector
    exposure_time: u.Quantity
    slit_pa: float = 0.0
    slit_offset: tuple[u.Quantity, u.Quantity] = (0.0 * u.arcsec, 0.0 * u.arcsec)
    margin: int = 10
    obstruction_fraction: float = 0.0
    n_exposures: int = 1
    sky_electron_rate: u.Quantity = 0.0 * u.electron / u.s

    def __post_init__(self) -> None:
        if self.slit_width <= 0:
            raise ValueError("slit_width must be positive")
        if self.n_spatial <= 0:
            raise ValueError("n_spatial must be positive")
        if self.margin < 1:
            raise ValueError("margin must be >= 1")
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

    @property
    def slit_length(self) -> u.Quantity:
        return self.n_spatial * self.pixel_scale

    def baseline_electrons(self) -> np.ndarray:
        """Expected sky + dark electrons per channel per along-slit pixel over the
        whole exposure, shape ``(n_wave, 1, 1)`` -- the *known* baseline the
        spectrum carries under the line (before noise). Mirrors
        :meth:`kino.instruments.ifu.IFU.baseline_electrons`; pass it as
        ``subtract_background`` to :func:`kino.analysis.moments.compute_moment_maps`
        to reduce ``observe()``'s ``spectrum`` without estimating the background
        from the data.
        """
        exposure_total = self.exposure_time * self.n_exposures
        sky = np.broadcast_to(
            (self.sky_electron_rate * exposure_total).to_value(u.electron), (self.n_wave,)
        )
        dark = (self.detector.dark_current * exposure_total).to_value(u.electron)
        return (sky + dark)[:, np.newaxis, np.newaxis]

    def _n_width_native(self) -> int:
        return max(
            1, int(np.ceil((self.slit_width / self.pixel_scale).to_value(u.dimensionless_unscaled)))
        )

    def _slit_frame_grid(self):
        ov = _SPATIAL_OVERSAMPLE
        pix_over = (self.pixel_scale / ov).to(u.arcsec)
        n_u_native = self.n_spatial + 2 * self.margin
        n_v_native = self._n_width_native() + 2 * self.margin
        n_u_over, n_v_over = n_u_native * ov, n_v_native * ov
        u_axis = (np.arange(n_u_over) - (n_u_over - 1) / 2.0) * pix_over
        v_axis = (np.arange(n_v_over) - (n_v_over - 1) / 2.0) * pix_over
        uu, vv = np.meshgrid(u_axis, v_axis, indexing="xy")
        return uu, vv, u_axis, v_axis, pix_over, n_u_native, n_v_native

    def observe(
        self,
        scene,
        *,
        line_flux: u.Quantity,
        line_wavelength: u.Quantity,
        rng: np.random.Generator | None = None,
    ) -> SlitObservation:
        """Render ``scene``'s emission line through this slit spectrograph.

        Parameters mirror :meth:`kino.instruments.ifu.IFU.observe`; see
        that method and this module's docstring.
        """
        galaxy = getattr(scene, "galaxy", scene)
        z = galaxy.z
        flux_norm = galaxy.flux

        uu, vv, u_axis, v_axis, pix_over, n_u_native, n_v_native = self._slit_frame_grid()
        pa_rad = np.radians(self.slit_pa)
        dx, dy = self.slit_offset
        x_sky = dx + uu * np.sin(pa_rad) + vv * np.cos(pa_rad)
        y_sky = dy + uu * np.cos(pa_rad) - vv * np.sin(pa_rad)

        sb = scene.surface_brightness(x_sky, y_sky)
        vlos = scene.vlos(x_sky, y_sky) * (u.km / u.s)
        sigma_v = scene.velocity_dispersion(x_sky, y_sky) * (u.km / u.s)

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

        exposure_total = self.exposure_time * self.n_exposures
        ov = _SPATIAL_OVERSAMPLE

        # -- spectrum: truncate to the slit width (fine grid), sum across it,
        # bin down along the slit, crop the margin.
        in_slit = np.abs(v_axis.to_value(u.arcsec)) <= (self.slit_width / 2.0).to_value(u.arcsec)
        spectrum_over_u = flux_cube_over[:, in_slit, :].sum(axis=1)
        spectrum_native_padded = spectrum_over_u.reshape(self.n_wave, n_u_native, ov).sum(axis=2)
        spectrum_native = spectrum_native_padded[:, self.margin : self.margin + self.n_spatial]
        spectrum_native = spectrum_native * line_flux.unit

        signal_electron = photon_flux_to_electrons(
            spectrum_native, wave_centers, self.throughput, self.effective_area, exposure_total
        )
        baseline = self.baseline_electrons()  # (n_wave, 1, 1): sky + dark
        expected_spectrum = signal_electron + baseline[:, 0, :]
        spectrum_counts, spectrum_variance = apply_shot_and_read_noise(
            expected_spectrum, self.detector.read_noise, self.n_exposures, rng
        )

        spectrum = Datacube(
            flux=spectrum_counts[:, :, np.newaxis],
            spectral_axis=wave_centers,
            spectral_kind="wavelength",
            pixel_scale=self.pixel_scale.to(u.arcsec),
            redshift=z,
            flux_unit=u.electron,
            variance=spectrum_variance[:, :, np.newaxis],
            meta={
                "source": "kino.instruments.slit.SlitSpectrograph.observe",
                "line_wavelength_rest": line_wavelength,
                "line_flux_intrinsic": line_flux,
                "slit_width": self.slit_width,
                "slit_pa": self.slit_pa,
                "slit_offset": self.slit_offset,
                "noiseless": rng is None,
            },
        )

        # -- companion broadband image: un-truncated, wavelength-collapsed.
        image_native_percow = flux_cube_over.reshape(
            self.n_wave, n_v_native, ov, n_u_native, ov
        ).sum(axis=(2, 4))
        image_native_percow = image_native_percow * line_flux.unit
        image_signal = photon_flux_to_electrons(
            image_native_percow, wave_centers, self.throughput, self.effective_area, exposure_total
        )
        # `baseline` broadcasts (n_wave, 1, 1) against every image pixel -- correct whether
        # sky_electron_rate is a scalar or, as kino.observing.skymodel's helpers return, a
        # per-channel (n_wave,) array (a bare `* np.ones((n_wave, ny, nx))` here would only
        # broadcast the scalar case; that was this branch's bug until an array-valued rate
        # was first exercised through SlitSpectrograph).
        expected_image = (image_signal + baseline).sum(axis=0)
        image_counts, _ = apply_shot_and_read_noise(
            expected_image, self.detector.read_noise, self.n_exposures, rng
        )

        return SlitObservation(
            spectrum=spectrum,
            image=image_counts,
            image_pixel_scale=self.pixel_scale.to(u.arcsec),
            slit_width=self.slit_width,
            slit_length=self.slit_length,
            slit_pa=self.slit_pa,
            slit_offset=self.slit_offset,
        )

    @classmethod
    def from_config(cls, path: str | Path) -> SlitSpectrograph:
        """Build a ``SlitSpectrograph`` from a YAML config (``kino/config/*.yaml``).

        See ``kino.instruments.config`` for the quantity-string format and
        ``kino/config/gmacs.yaml`` for a fully worked example.
        """
        with open(path) as f:
            cfg = yaml.safe_load(f)

        offset_cfg = cfg.get("slit_offset", ["0 arcsec", "0 arcsec"])
        return cls(
            slit_width=parse_quantity(cfg["slit_width"]),
            pixel_scale=parse_quantity(cfg["pixel_scale"]),
            n_spatial=cfg["n_spatial"],
            wavelength_min=parse_quantity(cfg["wavelength_min"]),
            wavelength_max=parse_quantity(cfg["wavelength_max"]),
            n_wave=cfg["n_wave"],
            psf=psf_from_config(cfg["psf"]),
            lsf=lsf_from_config(cfg["lsf"]),
            throughput=throughput_from_config(cfg["throughput"]),
            aperture_diameter=parse_quantity(cfg["aperture_diameter"]),
            detector=detector_from_config(cfg["detector"]),
            exposure_time=parse_quantity(cfg["exposure_time"]),
            slit_pa=cfg.get("slit_pa", 0.0),
            slit_offset=(parse_quantity(offset_cfg[0]), parse_quantity(offset_cfg[1])),
            margin=cfg.get("margin", 10),
            obstruction_fraction=cfg.get("obstruction_fraction", 0.0),
            n_exposures=cfg.get("n_exposures", 1),
            sky_electron_rate=parse_quantity(cfg.get("sky_electron_rate", "0 electron / s")),
        )
