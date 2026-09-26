import dataclasses

import astropy.units as u
import numpy as np
import pytest

from kino.analysis.fisher import DEFAULT_CUBE_PRIORS, DEFAULT_PRIORS, shear_fisher
from kino.galaxy import DiscGalaxy
from kino.instruments.base import LSF, PSF, Detector, Throughput
from kino.instruments.ifu import IFU

LINE_WAVELENGTH = 1.1 * u.um
LINE_FLUX = 5e-15 * u.erg / u.s / u.cm**2
GAL = dict(r_e=0.2, inc=55.0, pa=30.0, v_flat=180.0, r_t=0.15, z=0.0, sigma0=25.0, flux=1.0)
G = (0.03, -0.02)


def make_ifu(**kw) -> IFU:
    base = dict(
        pixel_scale=0.1 * u.arcsec, nx=8, ny=8,
        wavelength_min=1.099 * u.um, wavelength_max=1.101 * u.um, n_wave=20,
        psf=PSF(seeing_fwhm=0.02 * u.arcsec), lsf=LSF(resolving_power=8000.0),
        throughput=Throughput(wavelength=[1.0, 1.2] * u.um, transmission=[1.0, 1.0]),
        aperture_diameter=10 * u.m,
        detector=Detector(read_noise=0 * u.electron, dark_current=0 * u.electron / u.s),
        exposure_time=600 * u.s,
    )  # fmt: skip
    base.update(kw)
    return IFU(**base)


def fisher(ifu=None, **kw):
    kw.setdefault("line_flux", LINE_FLUX)
    return shear_fisher(
        ifu or make_ifu(), DiscGalaxy(**GAL), line_wavelength=LINE_WAVELENGTH, g=G, **kw
    )


class TestShearFisher:
    def test_returns_positive_finite_uncertainties(self):
        r = fisher()
        for s in (*r.sigma_fixed, *r.sigma_marginal):
            assert 0 < s < 1
        assert r.n_spaxels > 10
        assert r.param_names[:2] == ["g1", "g2"]

    def test_marginalising_over_nuisance_parameters_never_helps(self):
        r = fisher()
        assert r.sigma_marginal[0] >= r.sigma_fixed[0]
        assert r.sigma_marginal[1] >= r.sigma_fixed[1]

    def test_covariance_is_symmetric_positive_definite(self):
        cov = fisher().covariance
        assert cov == pytest.approx(cov.T, rel=1e-6, abs=1e-12)
        assert np.all(np.linalg.eigvalsh(cov) > 0)

    def test_photon_limited_precision_scales_as_inverse_root_exposure(self):
        a = fisher(make_ifu(exposure_time=600 * u.s))
        b = fisher(make_ifu(exposure_time=2400 * u.s))
        assert b.sigma_fixed[0] == pytest.approx(a.sigma_fixed[0] / 2.0, rel=0.2)
        assert b.sigma_fixed[1] == pytest.approx(a.sigma_fixed[1] / 2.0, rel=0.2)

    def test_brighter_line_gives_tighter_shear(self):
        faint = fisher(line_flux=2e-15 * u.erg / u.s / u.cm**2)
        bright = fisher(line_flux=1e-14 * u.erg / u.s / u.cm**2)
        assert bright.sigma_marginal[0] < faint.sigma_marginal[0]

    def test_tighter_priors_shrink_the_marginal_uncertainty(self):
        loose = fisher(priors={"inc": 30.0, "pa": 20.0, "v_flat": 1.0})
        tight = fisher(priors={"inc": 1.0, "pa": 0.5, "v_flat": 0.02})
        assert tight.sigma_marginal[0] < loose.sigma_marginal[0]
        assert tight.sigma_marginal[1] < loose.sigma_marginal[1]

    def test_perfectly_known_nuisance_recovers_the_fixed_case(self):
        tiny = dict.fromkeys(DEFAULT_CUBE_PRIORS, 1e-4)
        r = fisher(priors=tiny)
        assert r.sigma_marginal[0] == pytest.approx(r.sigma_fixed[0], rel=0.05)
        assert r.sigma_marginal[1] == pytest.approx(r.sigma_fixed[1], rel=0.05)

    def test_moment_estimator_also_recovers_the_fixed_case(self):
        tiny = dict.fromkeys(DEFAULT_PRIORS, 1e-4)
        r = fisher(estimator="moments", priors=tiny)
        assert r.sigma_marginal[0] == pytest.approx(r.sigma_fixed[0], rel=0.05)

    def test_much_worse_seeing_degrades_the_shear(self):
        sharp = fisher(make_ifu(psf=PSF(seeing_fwhm=0.02 * u.arcsec)))
        blurred = fisher(make_ifu(psf=PSF(seeing_fwhm=0.15 * u.arcsec)))
        assert blurred.sigma_marginal[0] > sharp.sigma_marginal[0]

    def test_an_undetectable_target_raises(self):
        with pytest.raises(ValueError, match="undetectable"):
            fisher(line_flux=1e-20 * u.erg / u.s / u.cm**2)

    def test_shear_snr_definition(self):
        r = fisher()
        s_fixed = r.shear_snr((0.06, 0.0), marginal=False)
        assert s_fixed == pytest.approx(0.06 / r.sigma_fixed[0])
        assert r.shear_snr((0.0, 0.0)) == 0.0
        assert r.shear_snr((0.06, 0.0)) < s_fixed
        # direction matters: along g2 the (different) g2 uncertainty applies
        assert r.shear_snr((0.0, 0.06), marginal=False) == pytest.approx(0.06 / r.sigma_fixed[1])

    def test_does_not_mutate_the_inputs(self):
        ifu = make_ifu()
        before = dataclasses.replace(ifu)
        gal = DiscGalaxy(**GAL)
        shear_fisher(ifu, gal, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH, g=G)
        assert ifu == before
        assert gal == DiscGalaxy(**GAL)


class TestLowSnrFlag:
    def test_median_spaxel_snr_grows_with_the_line_flux(self):
        faint = fisher(line_flux=2e-15 * u.erg / u.s / u.cm**2)
        bright = fisher(line_flux=1e-14 * u.erg / u.s / u.cm**2)
        assert bright.median_spaxel_snr > faint.median_spaxel_snr > 0

    def test_peak_is_at_least_the_median(self):
        r = fisher()
        assert r.peak_spaxel_snr >= r.median_spaxel_snr >= 3.0  # every used spaxel passed the cut

    def test_the_bright_test_galaxy_is_in_the_trusted_regime(self):
        from kino.analysis.fisher import LOW_SNR_WARNING

        assert fisher().median_spaxel_snr > LOW_SNR_WARNING


class TestCubeEstimator:
    def test_is_the_default_and_labelled(self):
        assert fisher().meta["estimator"] == "cube"

    def test_uses_more_information_than_the_moment_maps(self):
        cube = fisher(estimator="cube")
        moments = fisher(estimator="moments")
        assert cube.sigma_fixed[0] < moments.sigma_fixed[0]
        assert cube.sigma_marginal[0] < moments.sigma_marginal[0]

    def test_unknown_estimator_raises(self):
        with pytest.raises(ValueError, match="estimator"):
            fisher(estimator="magic")

    def test_free_subset_and_validation(self):
        full = fisher()
        subset = fisher(free=["inc", "pa"])
        assert subset.param_names == ["g1", "g2", "inc", "pa"]
        assert subset.sigma_marginal[0] <= full.sigma_marginal[0]
        with pytest.raises(ValueError, match="not in the prior set"):
            fisher(free=["not_a_parameter"])

    def test_a_looser_line_flux_prior_cannot_help(self):
        tight = fisher(priors={"line_flux": 0.01})
        loose = fisher(priors={"line_flux": 1.0})
        assert loose.sigma_marginal[0] >= tight.sigma_marginal[0] * (1 - 1e-6)

    def test_covariance_is_positive_definite_and_includes_line_flux(self):
        r = fisher()
        assert "line_flux" in r.param_names
        assert np.all(np.linalg.eigvalsh(r.covariance) > 0)

    def test_undetectable_target_raises(self):
        with pytest.raises(ValueError, match="undetectable"):
            fisher(line_flux=1e-20 * u.erg / u.s / u.cm**2)


class TestPsfMismatchBias:
    def test_is_nan_unless_requested(self):
        assert np.all(np.isnan(fisher().psf_bias))

    def test_no_error_no_bias(self):
        r = fisher(psf_error=0.0, free=["inc", "pa", "v_flat"])
        assert r.psf_bias == pytest.approx((0.0, 0.0), abs=1e-6)

    def test_bias_grows_with_the_psf_error(self):
        small = fisher(psf_error=0.05, free=["inc", "pa", "v_flat"])
        large = fisher(psf_error=0.20, free=["inc", "pa", "v_flat"])
        assert np.hypot(*large.psf_bias) > np.hypot(*small.psf_bias) > 0

    def test_a_larger_psf_error_makes_the_opposite_sign_bias(self):
        plus = fisher(psf_error=0.1, free=["inc", "pa", "v_flat"])
        minus = fisher(psf_error=-0.1, free=["inc", "pa", "v_flat"])
        assert plus.psf_bias[0] * minus.psf_bias[0] < 0 or plus.psf_bias[1] * minus.psf_bias[1] < 0
