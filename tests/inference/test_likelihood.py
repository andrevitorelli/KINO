import astropy.units as u
import numpy as np
import pytest

from kino.galaxy import DiscGalaxy
from kino.inference.likelihood import IFUMomentLikelihood
from kino.instruments.base import LSF, PSF, Detector, Throughput
from kino.instruments.ifu import IFU
from kino.lensing import apply_shear

LINE_WAVELENGTH = 1.1 * u.um
LINE_FLUX = 5e-15 * u.erg / u.s / u.cm**2
TRUTH = dict(r_e=0.2, inc=55.0, pa=30.0, v_flat=180.0, r_t=0.15, z=0.0, sigma0=25.0, flux=1.0)
G1_TRUE, G2_TRUE = 0.03, -0.02


def make_ifu(*, nx=12, ny=12, n_wave=40) -> IFU:
    return IFU(
        pixel_scale=0.07 * u.arcsec,
        nx=nx,
        ny=ny,
        wavelength_min=1.099 * u.um,
        wavelength_max=1.101 * u.um,
        n_wave=n_wave,
        psf=PSF(seeing_fwhm=0.02 * u.arcsec),
        lsf=LSF(resolving_power=8000.0),
        throughput=Throughput(wavelength=[1.0, 1.2] * u.um, transmission=[1.0, 1.0]),
        aperture_diameter=10 * u.m,
        detector=Detector(read_noise=0 * u.electron, dark_current=0 * u.electron / u.s),
        exposure_time=600 * u.s,
    )


def make_data_cube(ifu):
    scene = apply_shear(DiscGalaxy(**TRUTH), G1_TRUE, G2_TRUE)
    return ifu.observe(scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH, rng=None)


class TestIFUMomentLikelihood:
    def test_log_likelihood_is_zero_at_the_truth(self):
        ifu = make_ifu()
        data = make_data_cube(ifu)
        lik = IFUMomentLikelihood(
            ifu=ifu,
            line_wavelength=LINE_WAVELENGTH,
            data=data,
            fixed_galaxy_params={
                k: v for k, v in TRUTH.items() if k not in ("inc", "pa", "v_flat")
            },
            fixed_line_flux=LINE_FLUX,
        )
        ll = lik({"inc": 55.0, "pa": 30.0, "v_flat": 180.0, "g1": G1_TRUE, "g2": G2_TRUE})
        assert ll == pytest.approx(0.0, abs=1e-6)

    def test_log_likelihood_worsens_away_from_the_truth(self):
        ifu = make_ifu()
        data = make_data_cube(ifu)
        lik = IFUMomentLikelihood(
            ifu=ifu,
            line_wavelength=LINE_WAVELENGTH,
            data=data,
            fixed_galaxy_params=TRUTH,
            fixed_line_flux=LINE_FLUX,
        )
        ll_close = lik({"g1": G1_TRUE + 0.001, "g2": G2_TRUE})
        ll_far = lik({"g1": G1_TRUE + 0.05, "g2": G2_TRUE})
        assert ll_close > ll_far
        assert ll_far < 0

    def test_free_line_flux_at_truth_still_gives_zero(self):
        ifu = make_ifu()
        data = make_data_cube(ifu)
        lik = IFUMomentLikelihood(
            ifu=ifu,
            line_wavelength=LINE_WAVELENGTH,
            data=data,
            fixed_galaxy_params=TRUTH,
            line_flux_unit=LINE_FLUX.unit,
        )
        ll = lik({"g1": G1_TRUE, "g2": G2_TRUE, "line_flux": LINE_FLUX.to_value(LINE_FLUX.unit)})
        assert ll == pytest.approx(0.0, abs=1e-6)

    def test_missing_line_flux_raises(self):
        ifu = make_ifu()
        data = make_data_cube(ifu)
        lik = IFUMomentLikelihood(
            ifu=ifu, line_wavelength=LINE_WAVELENGTH, data=data, fixed_galaxy_params=TRUTH
        )
        with pytest.raises(ValueError):
            lik({"g1": G1_TRUE, "g2": G2_TRUE})

    def test_free_line_flux_without_a_unit_reference_raises(self):
        ifu = make_ifu()
        data = make_data_cube(ifu)
        lik = IFUMomentLikelihood(
            ifu=ifu, line_wavelength=LINE_WAVELENGTH, data=data, fixed_galaxy_params=TRUTH
        )
        with pytest.raises(ValueError):
            lik({"g1": G1_TRUE, "g2": G2_TRUE, "line_flux": LINE_FLUX.to_value(LINE_FLUX.unit)})

    def test_unphysical_shear_is_rejected_not_crashed(self):
        ifu = make_ifu()
        data = make_data_cube(ifu)
        lik = IFUMomentLikelihood(
            ifu=ifu,
            line_wavelength=LINE_WAVELENGTH,
            data=data,
            fixed_galaxy_params=TRUTH,
            fixed_line_flux=LINE_FLUX,
        )
        assert lik({"g1": 0.999, "g2": 0.999}) == -np.inf

    def test_no_valid_spaxels_raises_at_construction(self):
        ifu = make_ifu()
        data = make_data_cube(ifu)
        with pytest.raises(ValueError):
            IFUMomentLikelihood(
                ifu=ifu,
                line_wavelength=LINE_WAVELENGTH,
                data=data,
                fixed_galaxy_params=TRUTH,
                fixed_line_flux=LINE_FLUX,
                snr_threshold=1e6,
            )

    def test_fit_flux_includes_the_flux_moment(self):
        ifu = make_ifu()
        data = make_data_cube(ifu)
        lik = IFUMomentLikelihood(
            ifu=ifu,
            line_wavelength=LINE_WAVELENGTH,
            data=data,
            fixed_galaxy_params=TRUTH,
            fixed_line_flux=LINE_FLUX,
            fit_flux=True,
        )
        ll_at_truth = lik({"g1": G1_TRUE, "g2": G2_TRUE})
        assert ll_at_truth == pytest.approx(0.0, abs=1e-6)
        ll_wrong_flux = lik(
            {"g1": G1_TRUE, "g2": G2_TRUE, "line_flux": (2 * LINE_FLUX).to_value(LINE_FLUX.unit)}
        )
        assert ll_wrong_flux < ll_at_truth
