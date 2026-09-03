import astropy.units as u
import numpy as np
import pytest

from kino.galaxy import DiscGalaxy
from kino.inference.likelihood import IFUMomentLikelihood
from kino.inference.priors import build_prior
from kino.inference.sampler import Posterior, fit_shear
from kino.instruments.base import LSF, PSF, Detector, Throughput
from kino.instruments.ifu import IFU
from kino.lensing import apply_shear


class TestPosterior:
    def _make(self):
        rng = np.random.default_rng(0)
        samples = {
            "g1": rng.normal(0.03, 0.01, size=1000),
            "g2": rng.normal(-0.02, 0.02, size=1000),
        }
        return Posterior(samples=samples, log_z=-12.34, param_names=["g1", "g2"])

    def test_summary_medians_are_close_to_the_generating_values(self):
        post = self._make()
        summary = post.summary()
        assert summary["g1"][0] == pytest.approx(0.03, abs=0.01)
        assert summary["g2"][0] == pytest.approx(-0.02, abs=0.02)

    def test_summary_uncertainties_are_positive(self):
        post = self._make()
        for _median, minus, plus in post.summary().values():
            assert minus > 0
            assert plus > 0

    def test_n_samples(self):
        post = self._make()
        assert post.n_samples == 1000

    def test_repr_contains_log_z_and_every_parameter(self):
        post = self._make()
        text = repr(post)
        assert "log_z=-12.34" in text
        assert "g1 =" in text
        assert "g2 =" in text


class TestFitShearInjectionRecovery:
    def test_recovers_a_known_shear_from_a_noiseless_ifu_mock(self):
        """The core Phase 5 validation check (README.md/TEST_DATA.md): inject
        a known (g1, g2), observe noiselessly, and confirm nested sampling
        recovers it. Deliberately the smallest problem that still exercises
        the real nautilus sampler end to end (2 free parameters, a small
        IFU grid, low n_live/n_eff) -- this is the slowest test in the
        suite (tens of seconds) because nested sampling inherently needs
        many likelihood calls to converge, not because any single call is
        slow (see kino.inference.likelihood's module docstring)."""
        line_wavelength = 1.1 * u.um
        line_flux = 5e-15 * u.erg / u.s / u.cm**2
        truth = dict(
            r_e=0.2, inc=55.0, pa=30.0, v_flat=180.0, r_t=0.15, z=0.0, sigma0=25.0, flux=1.0
        )
        g1_true, g2_true = 0.03, -0.02

        ifu = IFU(
            pixel_scale=0.1 * u.arcsec,
            nx=8,
            ny=8,
            wavelength_min=1.099 * u.um,
            wavelength_max=1.101 * u.um,
            n_wave=20,
            psf=PSF(seeing_fwhm=0.02 * u.arcsec),
            lsf=LSF(resolving_power=8000.0),
            throughput=Throughput(wavelength=[1.0, 1.2] * u.um, transmission=[1.0, 1.0]),
            aperture_diameter=10 * u.m,
            detector=Detector(read_noise=0 * u.electron, dark_current=0 * u.electron / u.s),
            exposure_time=600 * u.s,
        )
        scene_true = apply_shear(DiscGalaxy(**truth), g1_true, g2_true)
        data_cube = ifu.observe(
            scene_true, line_flux=line_flux, line_wavelength=line_wavelength, rng=None
        )

        likelihood = IFUMomentLikelihood(
            ifu=ifu,
            line_wavelength=line_wavelength,
            data=data_cube,
            fixed_galaxy_params=truth,
            fixed_line_flux=line_flux,
        )
        prior = build_prior({"g1": (-0.15, 0.15), "g2": (-0.15, 0.15)})

        posterior = fit_shear(prior, likelihood, n_live=50, seed=0, n_eff=50)

        summary = posterior.summary()
        g1_med, g1_minus, g1_plus = summary["g1"]
        g2_med, g2_minus, g2_plus = summary["g2"]
        assert g1_med == pytest.approx(g1_true, abs=5 * max(g1_minus, g1_plus))
        assert g2_med == pytest.approx(g2_true, abs=5 * max(g2_minus, g2_plus))
        # and in absolute terms, since the tolerance above is only as good
        # as the (small-n_live) posterior's own self-reported width
        assert abs(g1_med - g1_true) < 0.01
        assert abs(g2_med - g2_true) < 0.01
