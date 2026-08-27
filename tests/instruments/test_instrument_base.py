import astropy.units as u
import numpy as np
import pytest

from kino.instruments.base import LSF, PSF, Detector, Throughput, photon_energy


def test_photon_energy_scales_inversely_with_wavelength():
    e1 = photon_energy(500 * u.nm)
    e2 = photon_energy(1000 * u.nm)
    assert np.isclose(e2.to_value(u.erg), (e1 / 2).to_value(u.erg), rtol=1e-9)


def test_photon_energy_known_value():
    # E = hc/lambda; 500 nm -> ~3.9728e-12 erg (independent hand calculation).
    e = photon_energy(500 * u.nm)
    assert np.isclose(e.to_value(u.erg), 3.9728e-12, rtol=1e-3)


class TestPSF:
    def _kernel(self, seeing_fwhm, pixel_scale=0.01 * u.arcsec):
        psf = PSF(seeing_fwhm=seeing_fwhm)
        return psf.kernel(
            pixel_scale, wavelength=1.0 * u.um, aperture_diameter=10 * u.m, obscuration=0.1
        )

    def test_kernel_is_normalised_and_nonnegative(self):
        kernel = self._kernel(0.1 * u.arcsec)
        assert kernel.min() >= 0
        assert np.isclose(kernel.sum(), 1.0, rtol=1e-3)

    def test_zero_seeing_is_diffraction_limited_only(self):
        kernel = self._kernel(0.0 * u.arcsec)
        assert np.isclose(kernel.sum(), 1.0, rtol=1e-3)

    @staticmethod
    def _second_moment(kernel: np.ndarray) -> float:
        n = kernel.shape[0]
        c = (n - 1) / 2.0
        y, x = np.mgrid[0:n, 0:n]
        return float((kernel * ((x - c) ** 2 + (y - c) ** 2)).sum())

    def test_larger_seeing_gives_a_broader_kernel(self):
        small = self._kernel(0.03 * u.arcsec)
        large = self._kernel(0.3 * u.arcsec)
        assert self._second_moment(large) > self._second_moment(small)

    def test_negative_seeing_rejected(self):
        with pytest.raises(ValueError):
            PSF(seeing_fwhm=-0.1 * u.arcsec)

    def test_seeing_must_be_an_angle(self):
        with pytest.raises(u.UnitsError):
            PSF(seeing_fwhm=0.1 * u.m)


class TestLSF:
    def test_requires_exactly_one_of_resolving_power_or_fwhm(self):
        with pytest.raises(ValueError):
            LSF()
        with pytest.raises(ValueError):
            LSF(resolving_power=5000.0, fwhm=1 * u.nm)

    def test_sigma_from_resolving_power(self):
        lsf = LSF(resolving_power=1000.0)
        wl = 1.0 * u.um
        expected_fwhm = wl / 1000.0
        expected_sigma = expected_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        assert np.isclose(lsf.sigma(wl).to_value(u.um), expected_sigma.to_value(u.um))

    def test_sigma_from_resolving_power_scales_with_wavelength(self):
        lsf = LSF(resolving_power=1000.0)
        assert np.isclose(
            lsf.sigma(1000 * u.nm).to_value(u.nm), 2.0 * lsf.sigma(500 * u.nm).to_value(u.nm)
        )

    def test_sigma_from_fixed_fwhm_is_wavelength_independent(self):
        lsf = LSF(fwhm=0.1 * u.nm)
        assert np.isclose(
            lsf.sigma(500 * u.nm).to_value(u.nm), lsf.sigma(1000 * u.nm).to_value(u.nm)
        )


class TestThroughput:
    def test_interpolates_linearly_between_tabulated_points(self):
        t = Throughput(wavelength=[500, 600, 700] * u.nm, transmission=[0.2, 0.4, 0.2])
        assert np.isclose(t(550 * u.nm), 0.3)

    def test_zero_outside_tabulated_range(self):
        t = Throughput(wavelength=[500, 600] * u.nm, transmission=[0.5, 0.5])
        assert t(100 * u.nm) == 0.0
        assert t(10000 * u.nm) == 0.0

    def test_rejects_out_of_range_transmission(self):
        with pytest.raises(ValueError):
            Throughput(wavelength=[500, 600] * u.nm, transmission=[0.5, 1.5])

    def test_rejects_non_increasing_wavelength(self):
        with pytest.raises(ValueError):
            Throughput(wavelength=[600, 500] * u.nm, transmission=[0.5, 0.5])


class TestDetector:
    def test_rejects_negative_read_noise(self):
        with pytest.raises(ValueError):
            Detector(read_noise=-1 * u.electron)

    def test_rejects_negative_dark_current(self):
        with pytest.raises(ValueError):
            Detector(read_noise=1 * u.electron, dark_current=-1 * u.electron / u.s)
