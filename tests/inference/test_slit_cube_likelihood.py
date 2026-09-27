import astropy.units as u
import numpy as np
import pytest
from scipy.optimize import minimize

from kino.galaxy import DiscGalaxy
from kino.inference import SlitCubeLikelihood
from kino.instruments.base import LSF, PSF, Detector, Throughput
from kino.instruments.slit import SlitSpectrograph
from kino.lensing import apply_shear

LINE_WAVELENGTH = 1.1 * u.um
LINE_FLUX = 5e-15 * u.erg / u.s / u.cm**2
TRUTH = dict(r_e=0.2, inc=55.0, pa=30.0, v_flat=180.0, r_t=0.15, z=0.0, sigma0=25.0, flux=1.0)
G = (0.03, -0.02)


def make_slit(read_noise=2.0, sky=0.5) -> SlitSpectrograph:
    return SlitSpectrograph(
        slit_width=1.0 * u.arcsec, pixel_scale=0.15 * u.arcsec, n_spatial=20,
        wavelength_min=1.098 * u.um, wavelength_max=1.102 * u.um, n_wave=24,
        psf=PSF(seeing_fwhm=0.5 * u.arcsec), lsf=LSF(resolving_power=8000.0),
        throughput=Throughput(wavelength=[1.0, 1.2] * u.um, transmission=[1.0, 1.0]),
        aperture_diameter=10 * u.m,
        detector=Detector(read_noise=read_noise * u.electron, dark_current=0 * u.electron / u.s),
        exposure_time=600 * u.s, sky_electron_rate=sky * u.electron / u.s,
    )  # fmt: skip


def data_spectrum(slit, rng=None):
    scene = apply_shear(DiscGalaxy(**TRUTH), *G)
    return slit.observe(
        scene, line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH, rng=rng
    ).spectrum


def make_lik(slit, data, **kw):
    kw.setdefault("fixed_galaxy_params", TRUTH)
    kw.setdefault("fixed_line_flux", LINE_FLUX)
    return SlitCubeLikelihood(slit=slit, line_wavelength=LINE_WAVELENGTH, data=data, **kw)


class TestSlitCubeLikelihood:
    def test_noiseless_data_peaks_at_the_truth(self):
        slit = make_slit()
        lik = make_lik(slit, data_spectrum(slit))
        at = lik({"g1": G[0], "g2": G[1]})
        assert at > lik({"g1": G[0] + 0.01, "g2": G[1]})
        assert at > lik({"g1": G[0], "g2": G[1] - 0.01})
        assert at > lik({"g1": G[0] + 0.1, "g2": G[1]})

    def test_worsens_monotonically_away_from_the_truth(self):
        slit = make_slit()
        lik = make_lik(slit, data_spectrum(slit))
        ll = [lik({"g1": G[0] + d, "g2": G[1]}) for d in (0.0, 0.01, 0.03, 0.1)]
        assert np.all(np.diff(ll) < 0)

    def test_noisy_data_are_recovered_unbiasedly_by_maximum_likelihood(self):
        slit = make_slit()
        rng = np.random.default_rng(4)
        est = []
        for _ in range(8):
            lik = make_lik(slit, data_spectrum(slit, rng))
            fit = minimize(
                lambda p, lik=lik: -lik({"g1": p[0], "g2": p[1]}),
                x0=[0.0, 0.0], method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-3),
            )  # fmt: skip
            est.append(fit.x)
        est = np.array(est)
        sem = est.std(0, ddof=1) / np.sqrt(len(est))
        assert np.all(np.abs(est.mean(0) - np.array(G)) < 4 * sem + 1e-4)

    def test_free_line_flux_with_unit(self):
        slit = make_slit()
        lik = make_lik(
            slit, data_spectrum(slit), fixed_line_flux=None, line_flux_unit=LINE_FLUX.unit
        )
        at = lik({"g1": G[0], "g2": G[1], "line_flux": LINE_FLUX.value})
        off = lik({"g1": G[0], "g2": G[1], "line_flux": 1.3 * LINE_FLUX.value})
        assert at > off

    def test_missing_line_flux_and_missing_unit_raise(self):
        slit = make_slit()
        data = data_spectrum(slit)
        with pytest.raises(ValueError, match="line_flux"):
            make_lik(slit, data, fixed_line_flux=None)({"g1": 0.0, "g2": 0.0})
        with pytest.raises(ValueError, match="line_flux_unit"):
            make_lik(slit, data, fixed_line_flux=None)({"g1": 0.0, "g2": 0.0, "line_flux": 1.0})

    def test_unphysical_shear_is_rejected_not_crashed(self):
        slit = make_slit()
        assert make_lik(slit, data_spectrum(slit))({"g1": 0.999, "g2": 0.999}) == -np.inf

    def test_shape_mismatch_raises(self):
        slit = make_slit()
        other = make_slit()
        other = type(other)(**{**other.__dict__, "n_spatial": 8})
        with pytest.raises(ValueError, match="shape"):
            make_lik(slit, data_spectrum(other))

    def test_free_galaxy_parameters_are_passed_through(self):
        slit = make_slit()
        lik = make_lik(
            slit,
            data_spectrum(slit),
            fixed_galaxy_params={k: v for k, v in TRUTH.items() if k != "inc"},
        )
        good = lik({"g1": G[0], "g2": G[1], "inc": 55.0})
        bad = lik({"g1": G[0], "g2": G[1], "inc": 65.0})
        assert good > bad

    def test_the_companion_image_is_not_used(self):
        """The likelihood only sees SlitObservation.spectrum -- confirm the data
        argument really is a bare Datacube, not the full SlitObservation."""
        slit = make_slit()
        obs = data_spectrum(slit)
        assert not hasattr(obs, "image")
        make_lik(slit, obs)  # constructs fine on the .spectrum alone
