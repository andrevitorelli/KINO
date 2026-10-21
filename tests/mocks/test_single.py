"""Tests for kino.mocks.single.simulate_observation (README.md Phase 4)."""

import astropy.units as u
import numpy as np

from kino.galaxy import DiscGalaxy
from kino.instruments.ifu import IFU
from kino.instruments.slit import SlitObservation, SlitSpectrograph
from kino.lensing import apply_shear
from kino.mocks.single import simulate_observation

GALAXY = DiscGalaxy(r_e=0.5, inc=55.0, pa=30.0, v_flat=180.0, z=0.1, sigma0=30.0)
LINE_FLUX = 1e-15 * u.erg / u.s / u.cm**2
LINE_WAVELENGTH = 0.65628 * u.um


class TestIFU:
    def test_matches_calling_apply_shear_and_observe_directly(self):
        ifu = IFU.from_config("kino/config/sifs.yaml")
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)

        direct = ifu.observe(
            apply_shear(GALAXY, 0.03, -0.01),
            line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH, rng=rng1,
        )  # fmt: skip
        via_single = simulate_observation(
            GALAXY, 0.03, -0.01, ifu,
            line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH, rng=rng2,
        )  # fmt: skip

        assert np.array_equal(direct.flux, via_single.flux)
        assert np.array_equal(direct.variance, via_single.variance)

    def test_provenance_is_attached(self):
        ifu = IFU.from_config("kino/config/sifs.yaml")
        result = simulate_observation(
            GALAXY, 0.03, -0.01, ifu,
            line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH, rng=None,
            meta={"seed": 42},
        )  # fmt: skip
        assert result.meta["r_e"] == GALAXY.r_e
        assert result.meta["inc"] == GALAXY.inc
        assert result.meta["g1"] == 0.03
        assert result.meta["g2"] == -0.01
        assert result.meta["seed"] == 42
        assert result.meta["line_flux"] == str(LINE_FLUX)


class TestSlit:
    def test_matches_calling_apply_shear_and_observe_directly(self):
        slit = SlitSpectrograph.from_config("kino/config/goodman.yaml")
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)

        direct = slit.observe(
            apply_shear(GALAXY, 0.03, -0.01),
            line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH, rng=rng1,
        )  # fmt: skip
        via_single = simulate_observation(
            GALAXY, 0.03, -0.01, slit,
            line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH, rng=rng2,
        )  # fmt: skip

        assert isinstance(via_single, SlitObservation)
        assert np.array_equal(direct.spectrum.flux, via_single.spectrum.flux)
        assert np.array_equal(direct.image, via_single.image)

    def test_provenance_is_attached_to_the_spectrum(self):
        slit = SlitSpectrograph.from_config("kino/config/goodman.yaml")
        result = simulate_observation(
            GALAXY, 0.03, -0.01, slit,
            line_flux=LINE_FLUX, line_wavelength=LINE_WAVELENGTH, rng=None,
        )  # fmt: skip
        assert result.spectrum.meta["r_e"] == GALAXY.r_e
        assert result.spectrum.meta["g1"] == 0.03
