import astropy.units as u
import numpy as np
import pytest
from astropy.constants import c as C_LIGHT

from kino.analysis.moments import compute_moment_maps
from kino.galaxy import DiscGalaxy
from kino.instruments.base import LSF, PSF, Detector, Throughput
from kino.instruments.ifu import IFU
from kino.io.base import Datacube
from kino.lensing import apply_shear


def _gaussian_line_cube(
    *,
    v0=50.0,
    sigma=30.0,
    amplitude=5000.0,
    background=0.0,
    variance_level=None,
    n_spec=121,
    v_range=300.0,
    ny=1,
    nx=1,
):
    """A synthetic velocity-kind cube whose per-channel value is the Gaussian
    line's flux *integrated over that channel's width* (matching how a real
    rendered cube -- e.g. kino.instruments.ifu.IFU's -- stores counts), not a
    density evaluated at a point."""
    velocity = np.linspace(-v_range, v_range, n_spec) * u.km / u.s
    dv = velocity.to_value(u.km / u.s)[1] - velocity.to_value(u.km / u.s)[0]
    profile = dv * amplitude * np.exp(-0.5 * ((velocity.value - v0) / sigma) ** 2) + background
    flux = np.broadcast_to(profile[:, np.newaxis, np.newaxis], (n_spec, ny, nx)).copy()
    variance = None
    if variance_level is not None:
        variance = np.full_like(flux, variance_level)
    return Datacube(
        flux=flux,
        spectral_axis=velocity,
        spectral_kind="velocity",
        pixel_scale=0.1 * u.arcsec,
        redshift=0.0,
        flux_unit=u.electron,
        variance=variance,
    )


class TestNoiselessRecovery:
    def test_recovers_known_vlos_and_dispersion(self):
        cube = _gaussian_line_cube(v0=50.0, sigma=30.0)
        m = compute_moment_maps(cube, snr_threshold=0, subtract_background=False)
        assert np.isclose(m.vlos[0, 0], 50.0, atol=1e-6)
        assert np.isclose(m.dispersion[0, 0], 30.0, rtol=1e-6)

    def test_flux_matches_analytic_gaussian_integral(self):
        amplitude, sigma = 5000.0, 30.0
        cube = _gaussian_line_cube(v0=0.0, sigma=sigma, amplitude=amplitude)
        m = compute_moment_maps(cube, snr_threshold=0, subtract_background=False)
        expected = amplitude * sigma * np.sqrt(2 * np.pi)
        assert np.isclose(m.flux[0, 0], expected, rtol=1e-3)

    def test_background_subtraction_removes_flat_pedestal(self):
        cube = _gaussian_line_cube(v0=0.0, sigma=30.0, amplitude=5000.0, background=1000.0)
        m_sub = compute_moment_maps(cube, snr_threshold=0, subtract_background=True)
        m_nosub = compute_moment_maps(cube, snr_threshold=0, subtract_background=False)
        # the un-subtracted pedestal (background * n_channels) inflates the
        # naive flux by roughly that fixed amount.
        n_channels = cube.shape[0]
        assert np.isclose(m_nosub.flux[0, 0] - m_sub.flux[0, 0], 1000.0 * n_channels, rtol=0.05)
        assert np.isclose(m_sub.vlos[0, 0], 0.0, atol=1.0)
        assert np.isclose(m_sub.dispersion[0, 0], 30.0, rtol=0.05)

    def test_lsf_correction_removes_instrumental_broadening_in_quadrature(self):
        lsf_sigma = 20.0 * u.km / u.s
        intrinsic_sigma = 30.0
        observed_sigma = np.hypot(intrinsic_sigma, lsf_sigma.value)
        cube = _gaussian_line_cube(v0=0.0, sigma=observed_sigma)
        m_corrected = compute_moment_maps(
            cube, snr_threshold=0, subtract_background=False, lsf_sigma=lsf_sigma
        )
        assert np.isclose(m_corrected.dispersion[0, 0], intrinsic_sigma, rtol=1e-2)

    def test_wavelength_kind_requires_line_wavelength(self):
        cube = _gaussian_line_cube(v0=0.0, sigma=30.0)
        wavelength_cube = Datacube(
            flux=cube.flux,
            spectral_axis=np.linspace(1.0, 1.2, cube.shape[0]) * u.um,
            spectral_kind="wavelength",
            pixel_scale=cube.pixel_scale,
            redshift=0.0,
            flux_unit=cube.flux_unit,
        )
        with pytest.raises(ValueError):
            compute_moment_maps(wavelength_cube)


class TestNoiseAndMasking:
    def test_snr_threshold_masks_low_significance_spaxels(self):
        cube = _gaussian_line_cube(v0=0.0, sigma=30.0, amplitude=5000.0, variance_level=1e8)
        m = compute_moment_maps(cube, snr_threshold=5.0, subtract_background=False)
        assert m.mask[0, 0]
        assert np.isnan(m.vlos[0, 0])
        assert np.isnan(m.dispersion[0, 0])
        # flux itself is always reported, masked or not
        assert np.isfinite(m.flux[0, 0])

    def test_bright_spaxel_is_not_masked(self):
        cube = _gaussian_line_cube(v0=0.0, sigma=30.0, amplitude=5000.0, variance_level=10.0)
        m = compute_moment_maps(cube, snr_threshold=5.0, subtract_background=False)
        assert not m.mask[0, 0]
        assert np.isfinite(m.vlos[0, 0])

    def test_no_variance_skips_errors_and_masking(self):
        cube = _gaussian_line_cube(v0=0.0, sigma=30.0)
        m = compute_moment_maps(cube, snr_threshold=1000.0, subtract_background=False)
        assert m.flux_err is None
        assert m.vlos_err is None
        assert m.dispersion_err is None
        assert m.snr is None
        assert m.mask is None
        assert np.isfinite(m.vlos[0, 0])  # not masked despite the absurd threshold

    def test_analytic_errors_match_monte_carlo_scatter(self):
        rng = np.random.default_rng(0)
        v0, sigma, amplitude, variance_level = 50.0, 30.0, 5000.0, 400.0
        n_draws = 300
        base = _gaussian_line_cube(
            v0=v0, sigma=sigma, amplitude=amplitude, variance_level=variance_level
        )
        clean_profile = base.flux[:, 0, 0]

        vlos_draws = np.empty(n_draws)
        dispersion_draws = np.empty(n_draws)
        analytic_vlos_err = analytic_dispersion_err = None
        for i in range(n_draws):
            noisy = clean_profile + rng.normal(
                0.0, np.sqrt(variance_level), size=clean_profile.shape
            )
            cube = Datacube(
                flux=noisy[:, np.newaxis, np.newaxis],
                spectral_axis=base.spectral_axis,
                spectral_kind="velocity",
                pixel_scale=base.pixel_scale,
                redshift=0.0,
                flux_unit=u.electron,
                variance=np.full((clean_profile.shape[0], 1, 1), variance_level),
            )
            m = compute_moment_maps(cube, snr_threshold=0, subtract_background=False)
            vlos_draws[i] = m.vlos[0, 0]
            dispersion_draws[i] = m.dispersion[0, 0]
            if analytic_vlos_err is None:
                analytic_vlos_err = m.vlos_err[0, 0]
                analytic_dispersion_err = m.dispersion_err[0, 0]

        assert np.isclose(np.std(vlos_draws), analytic_vlos_err, rtol=0.3)
        assert np.isclose(np.std(dispersion_draws), analytic_dispersion_err, rtol=0.3)


class TestIFURoundTrip:
    """Face-on (no velocity gradient, so no PSF-driven beam smearing to
    confound the comparison): the moment maps recovered from a real
    IFU.observe() cube should match the scene's own known Vlos/dispersion."""

    def test_recovers_known_vlos_and_dispersion_face_on(self):
        gal = DiscGalaxy(
            r_e=0.2, inc=0.0, pa=0.0, v_flat=180.0, flux=1.0, z=0.0, v_sys=40.0, sigma0=25.0
        )
        scene = apply_shear(gal, g1=0.0, g2=0.0)
        line_wavelength = 1.1 * u.um
        lsf = LSF(resolving_power=8000.0)

        ifu = IFU(
            pixel_scale=0.05 * u.arcsec,
            nx=16,
            ny=16,
            wavelength_min=1.099 * u.um,
            wavelength_max=1.101 * u.um,
            n_wave=80,
            psf=PSF(seeing_fwhm=0.02 * u.arcsec),
            lsf=lsf,
            throughput=Throughput(wavelength=[1.0, 1.2] * u.um, transmission=[1.0, 1.0]),
            aperture_diameter=10 * u.m,
            detector=Detector(read_noise=0 * u.electron, dark_current=0 * u.electron / u.s),
            exposure_time=600 * u.s,
        )
        cube = ifu.observe(
            scene, line_flux=5e-15 * u.erg / u.s / u.cm**2, line_wavelength=line_wavelength
        )

        lsf_sigma_kms = (C_LIGHT * lsf.sigma(line_wavelength) / line_wavelength).to(u.km / u.s)
        moments = compute_moment_maps(
            cube, line_wavelength=line_wavelength, snr_threshold=0, lsf_sigma=lsf_sigma_kms
        )

        assert np.allclose(moments.vlos, gal.v_sys, atol=0.5)
        assert np.allclose(moments.dispersion, gal.sigma0, rtol=0.1)
