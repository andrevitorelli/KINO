import astropy.units as u
import numpy as np
import pytest

from kino.galaxy import DiscGalaxy
from kino.instruments.slit import SlitSpectrograph
from kino.lensing import apply_shear

CONFIG = "kino/config/goodman.yaml"


class TestGoodmanConfig:
    def test_geometry_matches_published_numbers(self):
        s = SlitSpectrograph.from_config(CONFIG)
        assert s.slit_width.to_value(u.arcsec) == pytest.approx(1.0)
        assert s.pixel_scale.to_value(u.arcsec) == pytest.approx(0.15)
        assert s.aperture_diameter.to_value(u.m) == pytest.approx(4.1)

    def test_detector_read_noise_is_the_sourced_red_camera_value(self):
        s = SlitSpectrograph.from_config(CONFIG)
        assert s.detector.read_noise.to_value(u.electron) == pytest.approx(3.89)

    def test_lsf_matches_the_930_line_grating_at_550nm(self):
        s = SlitSpectrograph.from_config(CONFIG)
        assert s.lsf.resolving_power == pytest.approx(4450)

    def test_channel_width_matches_the_published_dispersion(self):
        s = SlitSpectrograph.from_config(CONFIG)
        width = (s.wavelength_max - s.wavelength_min) / s.n_wave
        assert width.to_value(u.angstrom) == pytest.approx(0.42, rel=0.02)

    def test_throughput_is_the_documented_sifs_derived_fiducial(self):
        """Pins the derivation: SIFS's peak transmission (0.097, at 493 nm) divided
        by 0.275 (midpoint of NOIRLab's "SIFS is 25-30% of Goodman's efficiency"
        statement). If this ever changes, it should be because a *real* curve
        replaced the fiducial one, not silently."""
        s = SlitSpectrograph.from_config(CONFIG)
        peak = float(np.max(s.throughput.transmission))
        assert peak == pytest.approx(0.097 / 0.275, rel=1e-3)
        assert np.all((s.throughput.transmission > 0) & (s.throughput.transmission < 0.6))

    def test_placeholder_sky_and_dark_are_still_placeholders(self):
        """Pins the fact that sky/dark are unverified: update this test on purpose
        when they are replaced with sourced values."""
        s = SlitSpectrograph.from_config(CONFIG)
        assert s.sky_electron_rate.to_value(u.electron / u.s) == 0.0
        assert s.detector.dark_current.to_value(u.electron / u.s) == pytest.approx(0.001)

    def test_observes_oiii_at_low_redshift(self):
        s = SlitSpectrograph.from_config(CONFIG)
        gal = apply_shear(
            DiscGalaxy(r_e=1.0, inc=55.0, pa=30.0, v_flat=180.0, z=0.0, sigma0=25.0), 0.03, -0.02
        )
        obs = s.observe(gal, line_flux=1e-15 * u.erg / u.s / u.cm**2, line_wavelength=0.5007 * u.um)
        assert obs.spectrum.flux.shape == (s.n_wave, s.n_spatial, 1)
        assert np.all(np.isfinite(obs.spectrum.flux))
        assert obs.spectrum.flux.max() > 0
        assert obs.image.shape[0] > 0 and obs.image.shape[1] > 0
