import dataclasses

import astropy.units as u
import numpy as np
import pytest

from kino.analysis.fisher import (
    DEFAULT_CUBE_PRIORS,
    DEFAULT_PRIORS,
    shear_fisher,
    shear_fisher_multi,
)
from kino.galaxy import DiscGalaxy
from kino.instruments.base import LSF, PSF, Detector, Throughput
from kino.instruments.ifu import IFU
from kino.instruments.slit import SlitSpectrograph

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


def make_slit(**kw) -> SlitSpectrograph:
    base = dict(
        slit_width=1.0 * u.arcsec, pixel_scale=0.15 * u.arcsec, n_spatial=20,
        wavelength_min=1.098 * u.um, wavelength_max=1.102 * u.um, n_wave=24,
        psf=PSF(seeing_fwhm=0.5 * u.arcsec), lsf=LSF(resolving_power=8000.0),
        throughput=Throughput(wavelength=[1.0, 1.2] * u.um, transmission=[1.0, 1.0]),
        aperture_diameter=10 * u.m,
        detector=Detector(read_noise=2 * u.electron, dark_current=0 * u.electron / u.s),
        exposure_time=600 * u.s, sky_electron_rate=0.5 * u.electron / u.s,
    )  # fmt: skip
    base.update(kw)
    return SlitSpectrograph(**base)


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


class TestSlitInstrument:
    """shear_fisher's cube estimator, given a SlitSpectrograph instead of an IFU."""

    @staticmethod
    def make_slit():
        from kino.instruments.slit import SlitSpectrograph

        return SlitSpectrograph(
            slit_width=1.0 * u.arcsec, pixel_scale=0.15 * u.arcsec, n_spatial=20,
            wavelength_min=1.098 * u.um, wavelength_max=1.102 * u.um, n_wave=24,
            psf=PSF(seeing_fwhm=0.5 * u.arcsec), lsf=LSF(resolving_power=8000.0),
            throughput=Throughput(wavelength=[1.0, 1.2] * u.um, transmission=[1.0, 1.0]),
            aperture_diameter=10 * u.m,
            detector=Detector(read_noise=2 * u.electron, dark_current=0 * u.electron / u.s),
            exposure_time=600 * u.s, sky_electron_rate=0.5 * u.electron / u.s,
        )  # fmt: skip

    def test_returns_a_sensible_result(self):
        r = shear_fisher(
            self.make_slit(), DiscGalaxy(**GAL), line_flux=LINE_FLUX,
            line_wavelength=LINE_WAVELENGTH, g=G,
        )  # fmt: skip
        assert 0 < r.sigma_fixed[0] < 1
        assert r.sigma_marginal[0] >= r.sigma_fixed[0]
        assert r.median_spaxel_snr > 0
        assert r.meta["estimator"] == "cube"

    def test_moment_estimator_is_ifu_only(self):
        with pytest.raises(AttributeError):
            shear_fisher(
                self.make_slit(), DiscGalaxy(**GAL), line_flux=LINE_FLUX,
                line_wavelength=LINE_WAVELENGTH, g=G, estimator="moments",
            )  # fmt: skip

    def test_psf_mismatch_bias_also_works_for_a_slit(self):
        r = shear_fisher(
            self.make_slit(), DiscGalaxy(**GAL), line_flux=LINE_FLUX,
            line_wavelength=LINE_WAVELENGTH, g=G, free=["inc", "pa", "v_flat"],
            priors={"inc": 5.0, "pa": 2.0, "v_flat": 0.25}, psf_error=0.1,
        )  # fmt: skip
        assert np.all(np.isfinite(r.psf_bias))
        assert np.hypot(*r.psf_bias) > 0


class TestShearFisherMulti:
    """Several independent exposures of the *same* galaxy (e.g. different slit position
    angles covering more of its velocity field), Fisher-summed before applying the
    nuisance priors once."""

    def multi(self, instruments, **kw):
        kw.setdefault("line_flux", LINE_FLUX)
        return shear_fisher_multi(
            instruments, DiscGalaxy(**GAL), line_wavelength=LINE_WAVELENGTH, g=G, **kw
        )

    def test_two_identical_half_exposures_match_one_full_exposure(self):
        # splitting a fixed time budget across two identical setups and summing their
        # Fisher information must reproduce one exposure with the full budget, up to the
        # tiny difference from _VARIANCE_FLOOR clipping a few voxels differently at half
        # the signal (not exact bit-for-bit, but agreement well under 1%)
        full = fisher(make_ifu(exposure_time=600 * u.s))
        halves = [make_ifu(exposure_time=300 * u.s), make_ifu(exposure_time=300 * u.s)]
        r = self.multi(halves)
        assert r.sigma_fixed[0] == pytest.approx(full.sigma_fixed[0], rel=1e-2)
        assert r.sigma_marginal[0] == pytest.approx(full.sigma_marginal[0], rel=1e-2)
        # the shear sub-block, which is what's actually reported/used; nuisance-parameter
        # cross-terms (e.g. sigma0) can differ by more in relative terms since the floor
        # clips a few voxels' variance differently at half the signal, without affecting
        # the shear precision itself
        assert r.covariance[:2, :2] == pytest.approx(full.covariance[:2, :2], rel=1e-2)

    def test_two_different_position_angles_help_the_marginal_precision(self):
        # a single misaligned slit vs. the same total time split across two different
        # position angles -- more independent information about (inc, pa, v_flat)
        # can only shrink (never grow) the marginal shear covariance
        one = [make_slit(slit_pa=0.0, exposure_time=600 * u.s)]
        two = [
            make_slit(slit_pa=0.0, exposure_time=300 * u.s),
            make_slit(slit_pa=90.0, exposure_time=300 * u.s),
        ]
        kw = dict(free=["inc", "pa", "v_flat"], priors={"inc": 20.0, "pa": 20.0, "v_flat": 0.5})
        r_one = self.multi(one, **kw)
        r_two = self.multi(two, **kw)
        assert r_two.sigma_marginal[0] <= r_one.sigma_marginal[0] + 1e-9
        assert r_two.sigma_marginal[1] <= r_one.sigma_marginal[1] + 1e-9

    def test_requires_at_least_one_instrument(self):
        with pytest.raises(ValueError, match="non-empty"):
            self.multi([])

    def test_meta_reports_the_exposure_count_and_estimator(self):
        r = self.multi([make_ifu(), make_ifu(), make_ifu()])
        assert r.meta == {"estimator": "cube", "n_exposures": 3}

    def test_psf_error_is_not_a_supported_argument(self):
        with pytest.raises(TypeError):
            self.multi([make_ifu()], psf_error=0.1)

    def test_works_for_a_slit_too(self):
        r = self.multi([make_slit(), make_slit(slit_pa=45.0)])
        assert 0 < r.sigma_fixed[0] < 1
        assert r.n_spaxels > 0

    def test_an_undetectable_target_raises(self):
        with pytest.raises(ValueError, match="undetectable"):
            self.multi([make_ifu(), make_ifu()], line_flux=1e-20 * u.erg / u.s / u.cm**2)

    def test_does_not_mutate_the_instruments(self):
        a, b = make_ifu(), make_ifu()
        before_a, before_b = dataclasses.replace(a), dataclasses.replace(b)
        self.multi([a, b])
        assert a == before_a
        assert b == before_b
