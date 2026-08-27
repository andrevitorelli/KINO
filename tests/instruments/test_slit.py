import astropy.units as u
import numpy as np
import pytest

from kino.galaxy import DiscGalaxy
from kino.instruments.base import LSF, PSF, Detector, Throughput, photon_energy
from kino.instruments.slit import SlitSpectrograph
from kino.lensing import apply_shear

_CONFIG_PATH = "kino/config/gmacs.yaml"


def make_slit(
    *,
    slit_width=8.0 * u.arcsec,
    pixel_scale=0.05 * u.arcsec,
    n_spatial=40,
    slit_pa=0.0,
    slit_offset=(0.0 * u.arcsec, 0.0 * u.arcsec),
    margin=15,
    seeing=0.05 * u.arcsec,
    resolving_power=5000.0,
    wavelength_min=1.09 * u.um,
    wavelength_max=1.11 * u.um,
    n_wave=60,
    exposure_time=600 * u.s,
    n_exposures=1,
    read_noise=0.0 * u.electron,
    dark_current=0.0 * u.electron / u.s,
    sky_electron_rate=0.0 * u.electron / u.s,
    aperture_diameter=10 * u.m,
    throughput_value=1.0,
) -> SlitSpectrograph:
    return SlitSpectrograph(
        slit_width=slit_width,
        pixel_scale=pixel_scale,
        n_spatial=n_spatial,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        n_wave=n_wave,
        psf=PSF(seeing_fwhm=seeing),
        lsf=LSF(resolving_power=resolving_power),
        throughput=Throughput(
            wavelength=[1.0, 1.2] * u.um, transmission=[throughput_value, throughput_value]
        ),
        aperture_diameter=aperture_diameter,
        detector=Detector(read_noise=read_noise, dark_current=dark_current),
        exposure_time=exposure_time,
        slit_pa=slit_pa,
        slit_offset=slit_offset,
        margin=margin,
        n_exposures=n_exposures,
        sky_electron_rate=sky_electron_rate,
    )


def compact_scene(**overrides):
    params = dict(r_e=0.05, inc=30, pa=10, v_flat=150, flux=1.0, z=0.0, sigma0=0.0)
    params.update(overrides)
    return apply_shear(DiscGalaxy(**params), g1=0.0, g2=0.0)


LINE_FLUX = 1e-15 * u.erg / u.s / u.cm**2
LINE_WAVELENGTH = 1.1 * u.um


class TestFluxCalibration:
    def test_total_detected_signal_matches_analytic_photon_budget(self):
        """Compact, centred source; a wide slit and generous margin so
        ~no flux is lost to the slit/window edges. Independently
        recomputes the expected photon count."""
        slit = make_slit()
        result = slit.observe(compact_scene(), line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH)

        e_photon = photon_energy(LINE_WAVELENGTH)
        expected = (LINE_FLUX / e_photon * slit.effective_area * slit.exposure_time).to_value(
            u.dimensionless_unscaled
        )
        assert np.isclose(result.spectrum.flux.sum(), expected, rtol=1e-2)

    def test_signal_scales_linearly_with_line_flux(self):
        slit = make_slit()
        scene = compact_scene()
        r1 = slit.observe(scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH)
        r2 = slit.observe(scene, line_flux=2 * LINE_FLUX, line_wavelength=LINE_WAVELENGTH)
        assert np.allclose(r2.spectrum.flux, 2 * r1.spectrum.flux, rtol=1e-9)


class TestSlitLosses:
    def test_narrower_slit_loses_flux(self):
        scene = compact_scene(inc=60, r_e=0.3, r_t=0.2)
        narrow = make_slit(slit_width=0.2 * u.arcsec).observe(
            scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH
        )
        wide = make_slit(slit_width=8.0 * u.arcsec).observe(
            scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH
        )
        assert narrow.spectrum.flux.sum() < wide.spectrum.flux.sum()

    def test_offset_slit_loses_flux(self):
        scene = compact_scene(inc=60, r_e=0.3, r_t=0.2)
        centred = make_slit(slit_width=0.7 * u.arcsec).observe(
            scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH
        )
        offset = make_slit(
            slit_width=0.7 * u.arcsec, slit_offset=(0.5 * u.arcsec, 0.0 * u.arcsec)
        ).observe(scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH)
        assert offset.spectrum.flux.sum() < centred.spectrum.flux.sum()

    def test_slit_aligned_with_disc_major_axis_captures_more_than_perpendicular(self):
        # an elongated, inclined disc through a narrow slit: aligning the
        # slit with the disc's own major axis should out-perform a
        # perpendicular placement of the same width.
        scene = compact_scene(inc=70, pa=30, r_e=0.4, r_t=0.3)
        aligned = make_slit(slit_width=0.3 * u.arcsec, slit_pa=30.0).observe(
            scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH
        )
        perpendicular = make_slit(slit_width=0.3 * u.arcsec, slit_pa=120.0).observe(
            scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH
        )
        assert aligned.spectrum.flux.sum() > perpendicular.spectrum.flux.sum()


class TestCompanionImage:
    def test_image_captures_at_least_as_much_flux_as_the_slit_spectrum(self):
        # the image is the un-truncated field; the slit spectrum is a strict
        # subset of that same light.
        scene = compact_scene(inc=60, r_e=0.3, r_t=0.2)
        result = make_slit(slit_width=0.5 * u.arcsec).observe(
            scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH
        )
        assert result.image.sum() >= result.spectrum.flux.sum()

    def test_image_shape_matches_padded_field(self):
        slit = make_slit(
            n_spatial=40, margin=15, slit_width=8.0 * u.arcsec, pixel_scale=0.05 * u.arcsec
        )
        result = slit.observe(compact_scene(), line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH)
        n_width_native = slit._n_width_native()
        assert result.image.shape == (
            n_width_native + 2 * slit.margin,
            slit.n_spatial + 2 * slit.margin,
        )


class TestNoiseModel:
    def test_snr_scales_as_sqrt_exposure_time_when_read_noise_free(self):
        scene = compact_scene()
        short = make_slit(
            exposure_time=600 * u.s, sky_electron_rate=1.0 * u.electron / u.s
        ).observe(scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH)
        long = make_slit(
            exposure_time=2400 * u.s, sky_electron_rate=1.0 * u.electron / u.s
        ).observe(scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH)
        snr_short = short.spectrum.flux.sum() / np.sqrt(short.spectrum.variance.sum())
        snr_long = long.spectrum.flux.sum() / np.sqrt(long.spectrum.variance.sum())
        assert np.isclose(snr_long / snr_short, np.sqrt(4.0), rtol=1e-6)

    def test_rng_draws_a_noisy_realisation(self):
        slit = make_slit(sky_electron_rate=5.0 * u.electron / u.s, read_noise=2.0 * u.electron)
        scene = compact_scene()
        expected = slit.observe(scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH)
        rng = np.random.default_rng(42)
        noisy = slit.observe(scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH, rng=rng)
        assert not np.array_equal(noisy.spectrum.flux, expected.spectrum.flux)
        assert not np.array_equal(noisy.image, expected.image)


class TestConfig:
    def test_from_config_loads_gmacs_fiducial(self):
        slit = SlitSpectrograph.from_config(_CONFIG_PATH)
        assert slit.slit_width > 0
        assert slit.n_spatial > 0
        assert slit.aperture_diameter == 24.5 * u.m
        assert slit.detector.read_noise > 0

    def test_from_config_observes_without_error(self):
        slit = SlitSpectrograph.from_config(_CONFIG_PATH)
        slit.n_spatial, slit.n_wave, slit.margin = 8, 6, 3  # keep the smoke test cheap
        scene = compact_scene()
        result = slit.observe(scene, line_flux=LINE_FLUX, line_wavelength=slit.band_center / 1.0)
        assert result.spectrum.shape == (6, 8, 1)
        assert np.all(np.isfinite(result.spectrum.flux))
        assert np.all(np.isfinite(result.image))


class TestValidation:
    def test_rejects_non_positive_n_spatial(self):
        with pytest.raises(ValueError):
            make_slit(n_spatial=0)

    def test_rejects_margin_below_one(self):
        with pytest.raises(ValueError):
            make_slit(margin=0)

    def test_rejects_non_positive_slit_width(self):
        with pytest.raises(ValueError):
            make_slit(slit_width=0 * u.arcsec)
