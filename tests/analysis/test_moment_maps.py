import warnings

import astropy.units as u
import numpy as np
import pytest
from astropy.constants import c as C_LIGHT

from kino.analysis.moments import compute_moment_maps, crop_to_line
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


class TestCropToLine:
    def test_crops_to_requested_range(self):
        cube = _gaussian_line_cube(v0=0.0, sigma=30.0, n_spec=301, v_range=1000.0)
        cropped = crop_to_line(cube, center=0.0 * u.km / u.s, half_width=250.0 * u.km / u.s)
        v = cropped.spectral_axis.to_value(u.km / u.s)
        assert v.min() >= -250.0
        assert v.max() <= 250.0
        assert cropped.shape[0] < cube.shape[0]

    def test_slices_variance_and_flux_consistently(self):
        cube = _gaussian_line_cube(
            v0=0.0, sigma=30.0, n_spec=301, v_range=1000.0, variance_level=25.0
        )
        cropped = crop_to_line(cube, center=0.0 * u.km / u.s, half_width=250.0 * u.km / u.s)
        full_v = cube.spectral_axis.to_value(u.km / u.s)
        keep = (full_v >= -250.0) & (full_v <= 250.0)
        assert np.array_equal(cropped.flux, cube.flux[keep])
        assert np.array_equal(cropped.variance, cube.variance[keep])

    def test_preserves_other_cube_metadata(self):
        cube = _gaussian_line_cube(v0=0.0, sigma=30.0, n_spec=301, v_range=1000.0)
        cropped = crop_to_line(cube, center=0.0 * u.km / u.s, half_width=250.0 * u.km / u.s)
        assert cropped.pixel_scale == cube.pixel_scale
        assert cropped.redshift == cube.redshift
        assert cropped.flux_unit == cube.flux_unit
        assert cropped.spectral_kind == cube.spectral_kind

    def test_raises_if_window_misses_the_cube_entirely(self):
        cube = _gaussian_line_cube(v0=0.0, sigma=30.0, n_spec=301, v_range=1000.0)
        with pytest.raises(ValueError):
            crop_to_line(cube, center=5000.0 * u.km / u.s, half_width=10.0 * u.km / u.s)

    def test_cropping_a_too_wide_window_recovers_good_moments(self):
        # the same scenario the collapse-rate sweep in
        # notebooks/07_moment_window_width_caveat.ipynb maps out: crop first,
        # and the "too wide" failure disappears.
        wide = _gaussian_line_cube(
            v0=50.0, sigma=30.0, n_spec=3001, v_range=3000.0, variance_level=400.0
        )
        cropped = crop_to_line(wide, center=50.0 * u.km / u.s, half_width=240.0 * u.km / u.s)
        m = compute_moment_maps(cropped, snr_threshold=0, subtract_background=True)
        assert m.dispersion[0, 0] > 0
        assert np.isclose(m.dispersion[0, 0], 30.0, rtol=0.3)


class TestWindowWidthDiagnostic:
    def test_warns_on_a_too_wide_window(self):
        wide = _gaussian_line_cube(
            v0=50.0, sigma=30.0, n_spec=3001, v_range=3000.0, variance_level=400.0
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_moment_maps(wide, snr_threshold=0, subtract_background=True)
        assert any(issubclass(w.category, UserWarning) for w in caught)

    def test_does_not_warn_on_a_well_sized_window(self):
        good = _gaussian_line_cube(
            v0=50.0, sigma=30.0, n_spec=241, v_range=240.0, variance_level=400.0
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_moment_maps(good, snr_threshold=0, subtract_background=True)
        assert len(caught) == 0

    def test_check_window_false_silences_it(self):
        wide = _gaussian_line_cube(
            v0=50.0, sigma=30.0, n_spec=3001, v_range=3000.0, variance_level=400.0
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_moment_maps(wide, snr_threshold=0, subtract_background=True, check_window=False)
        assert len(caught) == 0


class TestKnownBackground:
    """The default median background is fragile when the line fills the window;
    a *known* baseline is not. (Regression tests for a measured 4.6x
    underestimate of ``dispersion_err``.)"""

    # a line whose expected counts fill about half of a 20-channel window (Poisson noise)
    EXPECTED = np.array(
        [0, 0, 0, 0.4, 4.1, 30, 217.4, 1508.9, 7368.1, 21609.9, 36661.4, 35714.2,
         19742.3, 6061.3, 1012.6, 92, 4.7, 0.2, 0, 0]
    )  # fmt: skip

    def _draws(self, n_draws, **kwargs):
        rng = np.random.default_rng(0)
        velocity = (np.arange(20) - 9.5) * 27.3 * u.km / u.s
        out, errs = [], []
        for _ in range(n_draws):
            counts = rng.poisson(self.EXPECTED).astype(float)[:, None, None]
            cube = Datacube(
                flux=counts,
                spectral_axis=velocity,
                spectral_kind="velocity",
                pixel_scale=0.1 * u.arcsec,
                redshift=0.0,
                flux_unit=u.electron,
                variance=self.EXPECTED[:, None, None].copy(),
            )
            m = compute_moment_maps(cube, snr_threshold=0, check_window=False, **kwargs)
            out.append(m.dispersion[0, 0])
            errs.append(m.dispersion_err[0, 0])
        return np.array(out), np.array(errs)

    def test_known_baseline_gives_calibrated_dispersion_errors_in_a_narrow_window(self):
        disp, err = self._draws(500, subtract_background=np.zeros((20, 1, 1)))
        assert np.std(disp, ddof=1) / np.mean(err) == pytest.approx(1.0, rel=0.15)

    def test_the_median_background_is_biased_and_its_errors_too_small_there(self):
        """Documents the failure the known-baseline option exists to avoid."""
        disp, err = self._draws(500)  # default: per-spaxel median
        known, _ = self._draws(500, subtract_background=np.zeros((20, 1, 1)))
        assert np.std(disp, ddof=1) / np.mean(err) > 2.0
        assert abs(np.mean(disp) - np.mean(known)) > 1.0  # km/s: the median distorts the wings

    def test_known_baseline_matches_the_median_for_a_flat_pedestal_in_a_wide_window(self):
        cube = _gaussian_line_cube(background=40.0, variance_level=100.0, n_spec=241, v_range=900.0)
        median = compute_moment_maps(cube, snr_threshold=0, check_window=False)
        known = compute_moment_maps(
            cube,
            snr_threshold=0,
            check_window=False,
            subtract_background=np.full((241, 1, 1), 40.0),
        )
        assert known.vlos[0, 0] == pytest.approx(median.vlos[0, 0], abs=0.05)
        assert known.dispersion[0, 0] == pytest.approx(median.dispersion[0, 0], abs=0.05)
        assert known.flux[0, 0] == pytest.approx(median.flux[0, 0], rel=1e-3)

    def test_false_means_the_cube_is_already_subtracted(self):
        cube = _gaussian_line_cube(background=40.0)
        no_sub = compute_moment_maps(cube, snr_threshold=0, subtract_background=False)
        zeros = compute_moment_maps(
            cube, snr_threshold=0, subtract_background=np.zeros((121, 1, 1))
        )
        assert no_sub.flux[0, 0] == pytest.approx(zeros.flux[0, 0])

    def test_baseline_with_a_wrong_shape_raises(self):
        cube = _gaussian_line_cube()
        with pytest.raises(ValueError, match="broadcast"):
            compute_moment_maps(cube, subtract_background=np.zeros((7, 1, 1)))


class TestIFUBaseline:
    def test_baseline_is_sky_plus_dark_over_the_whole_exposure(self):
        ifu = IFU(
            pixel_scale=0.1 * u.arcsec, nx=4, ny=4,
            wavelength_min=1.099 * u.um, wavelength_max=1.101 * u.um, n_wave=10,
            psf=PSF(seeing_fwhm=0.05 * u.arcsec), lsf=LSF(resolving_power=8000.0),
            throughput=Throughput(wavelength=[1.0, 1.2] * u.um, transmission=[1.0, 1.0]),
            aperture_diameter=10 * u.m,
            detector=Detector(read_noise=2 * u.electron, dark_current=0.01 * u.electron / u.s),
            exposure_time=100 * u.s, n_exposures=3, sky_electron_rate=0.5 * u.electron / u.s,
        )  # fmt: skip
        b = ifu.baseline_electrons()
        assert b.shape == (10, 1, 1)
        assert b == pytest.approx((0.5 + 0.01) * 300.0)

    def test_baseline_is_the_noiseless_floor_of_an_empty_scene(self):
        from kino.instruments.ifu import IFU as _IFU  # noqa: F401 -- readability

        ifu = IFU(
            pixel_scale=0.1 * u.arcsec, nx=4, ny=4,
            wavelength_min=1.099 * u.um, wavelength_max=1.101 * u.um, n_wave=10,
            psf=PSF(seeing_fwhm=0.05 * u.arcsec), lsf=LSF(resolving_power=8000.0),
            throughput=Throughput(wavelength=[1.0, 1.2] * u.um, transmission=[1.0, 1.0]),
            aperture_diameter=10 * u.m,
            detector=Detector(read_noise=0 * u.electron, dark_current=0.02 * u.electron / u.s),
            exposure_time=100 * u.s, sky_electron_rate=np.linspace(0.1, 0.5, 10) * u.electron / u.s,
        )  # fmt: skip
        gal = DiscGalaxy(r_e=0.1, inc=30.0, pa=0.0, v_flat=100.0, z=0.0, sigma0=20.0)
        cube = ifu.observe(gal, line_flux=1e-40 * u.erg / u.s / u.cm**2, line_wavelength=1.1 * u.um)
        floor = np.broadcast_to(ifu.baseline_electrons(), cube.flux.shape)
        assert cube.flux == pytest.approx(floor, rel=1e-6)
