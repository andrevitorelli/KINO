import astropy.units as u
import numpy as np
import pytest

from kino.galaxy import DiscGalaxy
from kino.instruments.ifu import IFU
from kino.lensing import apply_shear

CONFIG = "kino/config/sifs.yaml"


class TestSIFSConfig:
    def test_geometry_matches_the_published_instrument(self):
        ifu = IFU.from_config(CONFIG)
        assert (ifu.nx, ifu.ny) == (50, 26)
        assert ifu.pixel_scale.to_value(u.arcsec) == pytest.approx(0.30)
        # 1300 fibres, 15" x 7.8" field
        assert ifu.nx * ifu.ny == 1300
        assert ifu.nx * ifu.pixel_scale.to_value(u.arcsec) == pytest.approx(15.0)
        assert ifu.ny * ifu.pixel_scale.to_value(u.arcsec) == pytest.approx(7.8)

    def test_detector_and_aperture(self):
        ifu = IFU.from_config(CONFIG)
        assert ifu.detector.read_noise.to_value(u.electron) == pytest.approx(3.9)
        assert ifu.aperture_diameter.to_value(u.m) == pytest.approx(4.1)

    def test_channel_width_matches_the_published_dispersion(self):
        ifu = IFU.from_config(CONFIG)
        width = (ifu.wavelength_max - ifu.wavelength_min) / ifu.n_wave
        assert width.to_value(u.angstrom) == pytest.approx(0.69, rel=0.01)

    def test_throughput_covers_the_window_and_is_a_plausible_efficiency(self):
        ifu = IFU.from_config(CONFIG)
        tp = ifu.throughput
        assert tp.wavelength.min() <= ifu.wavelength_min
        assert tp.wavelength.max() >= ifu.wavelength_max
        assert np.all((tp.transmission > 0) & (tp.transmission < 0.15))  # ~10% peak, per the figure

    def test_lsf_is_a_constant_fwhm(self):
        ifu = IFU.from_config(CONFIG)
        assert ifu.lsf.fwhm.to_value(u.angstrom) == pytest.approx(5500 / 4200, rel=1e-3)

    def test_placeholder_sky_and_dark_are_still_zero(self):
        """Pins the fact that sky/dark are unverified placeholders: when they
        are replaced with sourced values this test should be updated on
        purpose, not pass silently."""
        ifu = IFU.from_config(CONFIG)
        assert ifu.sky_electron_rate.to_value(u.electron / u.s) == 0.0
        assert ifu.detector.dark_current.to_value(u.electron / u.s) == 0.0

    def test_observes_halpha_at_low_redshift(self):
        ifu = IFU.from_config(CONFIG)
        gal = DiscGalaxy(r_e=1.5, inc=55.0, pa=30.0, v_flat=180.0, z=0.05, sigma0=30.0)
        cube = ifu.observe(
            apply_shear(gal, 0.03, 0.0),
            line_flux=1e-15 * u.erg / u.s / u.cm**2,
            line_wavelength=0.65628 * u.um,
        )
        assert cube.flux.shape == (290, 26, 50)
        assert np.all(np.isfinite(cube.flux))
        assert cube.flux.max() > 0
