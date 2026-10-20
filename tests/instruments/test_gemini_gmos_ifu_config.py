import astropy.units as u
import numpy as np
import pytest

from kino.galaxy import DiscGalaxy
from kino.instruments.ifu import IFU
from kino.lensing import apply_shear

CONFIG = "kino/config/gemini_gmos_ifu.yaml"


class TestGeminiGMOSIFUConfig:
    def test_geometry_matches_the_two_slit_mode(self):
        ifu = IFU.from_config(CONFIG)
        assert (ifu.nx, ifu.ny) == (35, 25)
        assert ifu.pixel_scale.to_value(u.arcsec) == pytest.approx(0.20)
        # Two-slit (full field) mode: 7" x 5" field (Gemini GMOS factsheet)
        assert ifu.nx * ifu.pixel_scale.to_value(u.arcsec) == pytest.approx(7.0)
        assert ifu.ny * ifu.pixel_scale.to_value(u.arcsec) == pytest.approx(5.0)

    def test_detector_and_aperture(self):
        ifu = IFU.from_config(CONFIG)
        assert ifu.detector.read_noise.to_value(u.electron) == pytest.approx(4.0)
        assert ifu.aperture_diameter.to_value(u.m) == pytest.approx(8.1)
        assert ifu.obstruction_fraction == pytest.approx(0.016)

    def test_lsf_matches_the_r831_lab_resolving_power(self):
        ifu = IFU.from_config(CONFIG)
        assert ifu.lsf.fwhm.to_value(u.angstrom) == pytest.approx(7570 / 4396, rel=1e-3)

    def test_throughput_covers_the_window(self):
        ifu = IFU.from_config(CONFIG)
        tp = ifu.throughput
        assert tp.wavelength.min() <= ifu.wavelength_min
        assert tp.wavelength.max() >= ifu.wavelength_max
        assert np.all((tp.transmission > 0) & (tp.transmission <= 0.20))  # fiducial, see header

    def test_placeholder_sky_and_dark_are_still_fiducial(self):
        """Pins the fact that sky is an unverified placeholder and dark current
        is a cross-instrument (GMOS-N, not GMOS-S) proxy: when either is
        replaced with a sourced value this test should be updated on purpose,
        not pass silently."""
        ifu = IFU.from_config(CONFIG)
        assert ifu.sky_electron_rate.to_value(u.electron / u.s) == 0.0
        assert ifu.detector.dark_current.to_value(u.electron / u.s) == pytest.approx(
            0.000833, rel=1e-3
        )

    def test_observes_halpha_at_moderate_redshift(self):
        ifu = IFU.from_config(CONFIG)
        gal = DiscGalaxy(r_e=1.0, inc=55.0, pa=30.0, v_flat=180.0, z=0.1535, sigma0=30.0)
        cube = ifu.observe(
            apply_shear(gal, 0.03, 0.0),
            line_flux=1e-15 * u.erg / u.s / u.cm**2,
            line_wavelength=0.65628 * u.um,
        )
        assert cube.flux.shape == (400, 25, 35)
        assert np.all(np.isfinite(cube.flux))
        assert cube.flux.max() > 0
