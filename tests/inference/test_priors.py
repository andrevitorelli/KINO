import numpy as np
import pytest

from kino.inference.priors import build_prior, tully_fisher_prior, tully_fisher_velocity


class TestTullyFisherVelocity:
    def test_matches_v0_at_the_reference_flux(self):
        assert tully_fisher_velocity(5.0, v0=200.0, flux0=5.0) == pytest.approx(200.0)

    def test_increases_with_flux(self):
        v_faint = tully_fisher_velocity(1.0, v0=200.0, flux0=5.0)
        v_bright = tully_fisher_velocity(20.0, v0=200.0, flux0=5.0)
        assert v_bright > v_faint

    def test_matches_notebook_03_alpha_default(self):
        # notebooks/03_tully_fisher_shear_signal.ipynb uses ALPHA = 4.0
        v = tully_fisher_velocity(20.0, v0=200.0, flux0=5.0)
        assert v == pytest.approx(200.0 * (20.0 / 5.0) ** 0.25)


class TestTullyFisherPrior:
    def test_median_matches_the_toy_relation(self):
        dist = tully_fisher_prior(5.0, v0=200.0, flux0=5.0, scatter_dex=0.1)
        assert dist.ppf(0.5) == pytest.approx(200.0, rel=1e-6)

    def test_scatter_is_symmetric_in_log_space(self):
        dist = tully_fisher_prior(5.0, v0=200.0, flux0=5.0, scatter_dex=0.1)
        lo, med, hi = dist.ppf([0.16, 0.5, 0.84])
        assert np.log10(hi / med) == pytest.approx(-np.log10(lo / med), rel=1e-6)

    def test_larger_scatter_widens_the_interval(self):
        tight = tully_fisher_prior(5.0, v0=200.0, flux0=5.0, scatter_dex=0.05)
        wide = tully_fisher_prior(5.0, v0=200.0, flux0=5.0, scatter_dex=0.3)
        assert (wide.ppf(0.84) - wide.ppf(0.16)) > (tight.ppf(0.84) - tight.ppf(0.16))

    def test_rvs_draws_positive_velocities(self):
        dist = tully_fisher_prior(5.0, v0=200.0, flux0=5.0, scatter_dex=0.2)
        samples = dist.rvs(size=200, random_state=np.random.default_rng(0))
        assert np.all(samples > 0)


class TestBuildPrior:
    def test_wires_uniform_bounds(self):
        prior = build_prior({"g1": (-0.5, 0.5), "g2": (-0.5, 0.5)})
        assert prior.dimensionality() == 2

    def test_accepts_a_scipy_style_distribution(self):
        dist = tully_fisher_prior(5.0, v0=200.0, flux0=5.0)
        prior = build_prior({"v_flat": dist, "g1": (-0.5, 0.5)})
        assert prior.dimensionality() == 2
