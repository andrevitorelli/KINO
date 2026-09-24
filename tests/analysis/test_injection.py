import astropy.units as u
import numpy as np
import pytest

from kino.analysis import injection
from kino.analysis.injection import (
    InjectionTrial,
    _trial_seeds,
    recover_shear,
    run_injection,
    summarize_trials,
)
from kino.galaxy import DiscGalaxy
from kino.instruments.base import LSF, PSF, Detector, Throughput
from kino.instruments.ifu import IFU

LINE_WAVELENGTH = 1.1 * u.um
LINE_FLUX = 5e-15 * u.erg / u.s / u.cm**2
TRUTH = dict(r_e=0.2, inc=55.0, pa=30.0, v_flat=180.0, r_t=0.15, z=0.0, sigma0=25.0, flux=1.0)


def make_trials(g1_values, *, m=0.0, c=0.0, sigma=0.01, n_per=200, seed=0):
    """Synthetic trials with known linear bias and Gaussian scatter that
    matches the reported per-trial error."""
    rng = np.random.default_rng(seed)
    trials = []
    for g in g1_values:
        for _ in range(n_per):
            rec1 = (1 + m) * g + c + rng.normal(0, sigma)
            rec2 = rng.normal(0, sigma)
            trials.append(InjectionTrial((g, 0.0), (rec1, rec2), (sigma, sigma)))
    return trials


class TestSummarizeTrials:
    def test_recovers_injected_multiplicative_and_additive_bias(self):
        trials = make_trials([-0.06, -0.03, 0.0, 0.03, 0.06], m=0.08, c=0.004)
        s = summarize_trials(trials).g1
        assert s.m == pytest.approx(0.08, abs=4 * s.m_err)
        assert s.c == pytest.approx(0.004, abs=4 * s.c_err)
        assert s.m_err > 0
        assert s.c_err > 0

    def test_unbiased_ensemble_is_consistent_with_m_c_zero(self):
        s = summarize_trials(make_trials([-0.05, 0.0, 0.05])).g1
        assert abs(s.m) < 4 * s.m_err
        assert abs(s.c) < 4 * s.c_err

    def test_sigma_eps_is_the_scatter_of_recovered_minus_true(self):
        s = summarize_trials(make_trials([0.0], sigma=0.02)).g1
        assert s.sigma_eps == pytest.approx(0.02, rel=0.15)

    def test_null_test_mean_bias_flags_an_additive_offset(self):
        clean = summarize_trials(make_trials([0.0], c=0.0)).g1
        offset = summarize_trials(make_trials([0.0], c=0.01)).g1
        assert abs(clean.mean_bias) < 4 * clean.mean_bias_err
        assert offset.mean_bias > 4 * offset.mean_bias_err

    def test_pull_std_is_one_when_posterior_widths_are_honest(self):
        s = summarize_trials(make_trials([0.0], sigma=0.01)).g1
        assert s.pull_std == pytest.approx(1.0, abs=0.15)
        assert abs(s.pull_mean) < 0.3

    def test_pull_std_exceeds_one_when_posteriors_are_overconfident(self):
        trials = [
            InjectionTrial(t.g_true, t.g_rec, (t.g_err[0] / 3, t.g_err[1] / 3))
            for t in make_trials([0.0])
        ]
        assert summarize_trials(trials).g1.pull_std > 2.0

    def test_m_and_c_are_nan_when_all_true_shears_are_identical(self):
        s = summarize_trials(make_trials([0.03], n_per=20)).g1
        assert np.isnan(s.m)
        assert np.isnan(s.c)
        assert np.isfinite(s.mean_bias)

    def test_components_are_summarised_independently(self):
        summary = summarize_trials(make_trials([-0.05, 0.05], m=0.1))
        assert summary.g1.m == pytest.approx(0.1, abs=4 * summary.g1.m_err)
        # g2 was generated with no dependence on the true g1
        assert summary.n_trials == 400

    def test_single_trial_does_not_crash(self):
        s = summarize_trials([InjectionTrial((0.0, 0.0), (0.001, 0.0), (0.01, 0.01))]).g1
        assert np.isnan(s.sigma_eps)
        assert s.mean_bias == pytest.approx(0.001)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            summarize_trials([])

    def test_repr_lists_both_components(self):
        text = repr(summarize_trials(make_trials([-0.05, 0.05], n_per=5)))
        assert "g1:" in text
        assert "g2:" in text


class TestRunInjectionBookkeeping:
    """Ordering, seeding and validation, with the (slow) fit stubbed out."""

    @pytest.fixture
    def calls(self, monkeypatch):
        recorded = []

        def fake_recover(galaxy, g1, g2, *, rng, sampler_seed, **kwargs):
            recorded.append((galaxy, (g1, g2), rng.integers(1 << 30), sampler_seed, kwargs))
            return InjectionTrial((g1, g2), (g1, g2), (0.01, 0.01), galaxy=galaxy)

        monkeypatch.setattr(injection, "recover_shear", fake_recover)
        return recorded

    def test_trials_are_ordered_shear_major_and_cover_every_pair(self, calls):
        galaxies = [DiscGalaxy(**{**TRUTH, "inc": i}) for i in (30.0, 50.0)]
        shears = [(0.0, 0.0), (0.05, 0.0), (0.0, 0.05)]
        trials = run_injection(galaxies, shears, seed=0, n_live=10)
        assert [t.g_true for t in trials] == [s for s in shears for _ in galaxies]
        assert [t.galaxy.inc for t in trials] == [30.0, 50.0] * 3

    def test_kwargs_are_forwarded(self, calls):
        run_injection([DiscGalaxy(**TRUTH)], [(0.0, 0.0)], seed=0, n_live=7)
        assert calls[0][4] == {"n_live": 7}

    def test_same_seed_reproduces_and_different_seed_differs(self, calls):
        gal = DiscGalaxy(**TRUTH)
        run_injection([gal], [(0.0, 0.0), (0.05, 0.0)], seed=1)
        run_injection([gal], [(0.0, 0.0), (0.05, 0.0)], seed=1)
        run_injection([gal], [(0.0, 0.0), (0.05, 0.0)], seed=2)
        first, second, third = calls[0:2], calls[2:4], calls[4:6]
        assert [c[2:4] for c in first] == [c[2:4] for c in second]
        assert [c[2:4] for c in first] != [c[2:4] for c in third]

    def test_trials_get_distinct_seeds(self, calls):
        run_injection([DiscGalaxy(**TRUTH)], [(0.0, 0.0), (0.05, 0.0), (0.1, 0.0)], seed=0)
        assert len({c[3] for c in calls}) == 3
        assert len({c[2] for c in calls}) == 3

    def test_trial_seed_does_not_depend_on_ensemble_size(self):
        small = _trial_seeds(5, 2)
        large = _trial_seeds(5, 6)
        assert small[1][1] == large[1][1]

    def test_reserved_kwargs_rejected(self, calls):
        gal = DiscGalaxy(**TRUTH)
        with pytest.raises(TypeError):
            run_injection([gal], [(0.0, 0.0)], seed=0, rng=np.random.default_rng(0))
        with pytest.raises(TypeError):
            run_injection([gal], [(0.0, 0.0)], seed=0, sampler_seed=1)

    def test_empty_inputs_rejected(self, calls):
        with pytest.raises(ValueError):
            run_injection([], [(0.0, 0.0)], seed=0)
        with pytest.raises(ValueError):
            run_injection([DiscGalaxy(**TRUTH)], [], seed=0)


def make_ifu() -> IFU:
    return IFU(
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


class TestRecoverShear:
    def test_rejects_free_priors_that_are_not_galaxy_fields(self):
        with pytest.raises(ValueError, match="DiscGalaxy fields"):
            recover_shear(
                DiscGalaxy(**TRUTH),
                0.0,
                0.0,
                ifu=make_ifu(),
                line_wavelength=LINE_WAVELENGTH,
                line_flux=LINE_FLUX,
                free_priors={"not_a_field": (0.0, 1.0)},
            )

    def test_recovers_a_noiseless_injected_shear_end_to_end(self):
        """The real pipeline: DiscGalaxy -> shear -> IFU -> moments ->
        nautilus. Slow (tens of seconds) for the same reason as
        ``tests/inference/test_sampler.py``'s injection test."""
        g1, g2 = 0.03, -0.02
        trial = recover_shear(
            DiscGalaxy(**TRUTH),
            g1,
            g2,
            ifu=make_ifu(),
            line_wavelength=LINE_WAVELENGTH,
            line_flux=LINE_FLUX,
            sampler_seed=0,
            n_live=30,
            n_eff=30,
        )
        assert trial.g_true == (g1, g2)
        assert trial.g_rec[0] == pytest.approx(g1, abs=0.01)
        assert trial.g_rec[1] == pytest.approx(g2, abs=0.01)
        assert all(e > 0 for e in trial.g_err)
        assert trial.posterior is None
