import astropy.units as u
import numpy as np
import pytest

from kino.galaxy import DiscGalaxy
from kino.instruments.base import LSF, PSF, Detector, Throughput, photon_energy
from kino.instruments.ifu import IFU
from kino.lensing import apply_shear
from kino.lensing.moments import measure_moments

_CONFIG_PATH = "kino/config/gmtifs.yaml"


def make_ifu(
    *,
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
    nx=24,
    ny=24,
    pixel_scale=0.05 * u.arcsec,
    aperture_diameter=10 * u.m,
    throughput_value=1.0,
) -> IFU:
    return IFU(
        pixel_scale=pixel_scale,
        nx=nx,
        ny=ny,
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
        n_exposures=n_exposures,
        sky_electron_rate=sky_electron_rate,
    )


def compact_face_on_scene(**overrides):
    params = dict(r_e=0.05, inc=30, pa=10, v_flat=150, flux=1.0, z=0.0, sigma0=0.0)
    params.update(overrides)
    return apply_shear(DiscGalaxy(**params), g1=0.0, g2=0.0)


LINE_FLUX = 1e-15 * u.erg / u.s / u.cm**2
LINE_WAVELENGTH = 1.1 * u.um  # rest wavelength; scene z=0 -> centred in the default test window


class TestFluxCalibration:
    def test_total_detected_signal_matches_analytic_photon_budget(self):
        """Compact, centred, zero-shear source; wide margins so ~no flux is
        lost to the field/window edges. Independently recomputes the
        expected photon count rather than re-deriving it from the code
        under test."""
        ifu = make_ifu()
        cube = ifu.observe(
            compact_face_on_scene(), line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH
        )

        e_photon = photon_energy(LINE_WAVELENGTH)
        expected = (LINE_FLUX / e_photon * ifu.effective_area * ifu.exposure_time).to_value(
            u.dimensionless_unscaled
        )
        assert np.isclose(cube.flux.sum(), expected, rtol=1e-2)

    def test_signal_scales_linearly_with_line_flux(self):
        ifu = make_ifu()
        scene = compact_face_on_scene()
        c1 = ifu.observe(scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH)
        c2 = ifu.observe(scene, line_flux=2 * LINE_FLUX, line_wavelength=LINE_WAVELENGTH)
        assert np.allclose(c2.flux, 2 * c1.flux, rtol=1e-9)

    def test_noiseless_observe_is_deterministic(self):
        ifu = make_ifu()
        scene = compact_face_on_scene()
        c1 = ifu.observe(scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH)
        c2 = ifu.observe(scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH)
        assert np.array_equal(c1.flux, c2.flux)


class TestPSFBroadening:
    def test_larger_seeing_broadens_the_rendered_image(self):
        scene = compact_face_on_scene()
        small = make_ifu(seeing=0.02 * u.arcsec).observe(
            scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH
        )
        large = make_ifu(seeing=0.4 * u.arcsec).observe(
            scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH
        )

        def image_second_moment(cube):
            image = cube.flux.sum(axis=0)
            ny, nx = image.shape
            y, x = np.mgrid[0:ny, 0:nx]
            _, _, qxx, qyy, _ = measure_moments(image, x, y)
            return qxx + qyy

        assert image_second_moment(large) > image_second_moment(small)


class TestLSFBroadening:
    def test_lower_resolving_power_broadens_the_spectrum(self):
        scene = compact_face_on_scene()
        broad = make_ifu(resolving_power=500.0).observe(
            scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH
        )
        narrow = make_ifu(resolving_power=5000.0).observe(
            scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH
        )

        def spectral_second_moment(cube):
            spectrum = cube.flux.sum(axis=(1, 2))
            wavelength = cube.spectral_axis.to_value(u.um)
            weight = spectrum / spectrum.sum()
            mean = (weight * wavelength).sum()
            return (weight * (wavelength - mean) ** 2).sum()

        assert spectral_second_moment(broad) > spectral_second_moment(narrow)


class TestNoiseModel:
    def test_snr_scales_as_sqrt_exposure_time_when_read_noise_free(self):
        # signal, sky, and dark electrons are all linear in exposure time, so
        # with no read noise S/N should scale exactly as sqrt(exposure_time).
        scene = compact_face_on_scene()
        short = make_ifu(exposure_time=600 * u.s, sky_electron_rate=1.0 * u.electron / u.s).observe(
            scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH
        )
        long = make_ifu(exposure_time=2400 * u.s, sky_electron_rate=1.0 * u.electron / u.s).observe(
            scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH
        )
        snr_short = short.flux.sum() / np.sqrt(short.variance.sum())
        snr_long = long.flux.sum() / np.sqrt(long.variance.sum())
        assert np.isclose(snr_long / snr_short, np.sqrt(4.0), rtol=1e-6)

    def test_read_noise_increases_variance_by_its_square_times_n_exposures(self):
        ifu0 = make_ifu(read_noise=0.0 * u.electron, n_exposures=3)
        ifu5 = make_ifu(read_noise=5.0 * u.electron, n_exposures=3)
        scene = compact_face_on_scene()
        c0 = ifu0.observe(scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH)
        c5 = ifu5.observe(scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH)
        assert np.allclose(c5.variance - c0.variance, 5.0**2 * 3, rtol=1e-9)

    def test_rng_draws_a_noisy_realisation_around_the_expected_value(self):
        ifu = make_ifu(sky_electron_rate=5.0 * u.electron / u.s, read_noise=2.0 * u.electron)
        scene = compact_face_on_scene()
        expected = ifu.observe(scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH)
        rng = np.random.default_rng(42)
        noisy = ifu.observe(scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH, rng=rng)
        assert not np.array_equal(noisy.flux, expected.flux)
        # the mock realisation's total should be close to the expected total
        # (many independent Poisson/Gaussian voxels -> tight relative scatter).
        assert np.isclose(noisy.flux.sum(), expected.flux.sum(), rtol=0.05)


class TestConfig:
    def test_from_config_loads_gmtifs_fiducial(self):
        ifu = IFU.from_config(_CONFIG_PATH)
        assert ifu.pixel_scale == 0.05 * u.arcsec
        assert (ifu.nx, ifu.ny) == (80, 40)
        assert ifu.aperture_diameter == 24.5 * u.m
        assert ifu.lsf.resolving_power == 8000
        assert ifu.psf.seeing_fwhm == 0.05 * u.arcsec
        assert ifu.detector.read_noise == 5 * u.electron
        assert ifu.n_exposures == 6

    def test_from_config_observes_without_error(self):
        ifu = IFU.from_config(_CONFIG_PATH)
        ifu.nx, ifu.ny, ifu.n_wave = 8, 8, 6  # keep the smoke test cheap
        scene = compact_face_on_scene(z=0.68)
        cube = ifu.observe(scene, line_flux=LINE_FLUX, line_wavelength=0.6563 * u.um)
        assert cube.shape == (6, 8, 8)
        assert np.all(np.isfinite(cube.flux))


class TestValidation:
    def test_rejects_non_positive_array_size(self):
        with pytest.raises(ValueError):
            make_ifu(nx=0)

    def test_rejects_inverted_wavelength_range(self):
        with pytest.raises(ValueError):
            make_ifu(wavelength_min=1.1 * u.um, wavelength_max=1.0 * u.um)

    def test_rejects_obstruction_fraction_out_of_range(self):
        with pytest.raises(ValueError):
            IFU(
                pixel_scale=0.05 * u.arcsec,
                nx=4,
                ny=4,
                wavelength_min=1.0 * u.um,
                wavelength_max=1.1 * u.um,
                n_wave=4,
                psf=PSF(seeing_fwhm=0.05 * u.arcsec),
                lsf=LSF(resolving_power=1000.0),
                throughput=Throughput(wavelength=[1.0, 1.2] * u.um, transmission=[1.0, 1.0]),
                aperture_diameter=10 * u.m,
                detector=Detector(read_noise=0 * u.electron),
                exposure_time=600 * u.s,
                obstruction_fraction=1.0,
            )
