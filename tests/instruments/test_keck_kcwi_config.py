import astropy.units as u
import numpy as np
import pytest

from kino.galaxy import DiscGalaxy
from kino.instruments.ifu import IFU
from kino.lensing import apply_shear

CONFIG = "kino/config/keck_kcwi.yaml"


class TestKeckKCWIConfig:
    def test_geometry_matches_the_medium_slicer(self):
        ifu = IFU.from_config(CONFIG)
        assert (ifu.nx, ifu.ny) == (24, 30)
        assert ifu.pixel_scale.to_value(u.arcsec) == pytest.approx(0.69)
        # Medium slicer: 16.5" x 20.4" field (KOA modes page)
        assert ifu.nx * ifu.pixel_scale.to_value(u.arcsec) == pytest.approx(16.56, rel=0.01)
        assert ifu.ny * ifu.pixel_scale.to_value(u.arcsec) == pytest.approx(20.7, rel=0.01)

    def test_detector_and_aperture(self):
        ifu = IFU.from_config(CONFIG)
        assert ifu.detector.read_noise.to_value(u.electron) == pytest.approx(2.5)
        assert ifu.aperture_diameter.to_value(u.m) == pytest.approx(9.96)

    def test_channel_width_matches_the_bm_grating_dispersion(self):
        ifu = IFU.from_config(CONFIG)
        width = (ifu.wavelength_max - ifu.wavelength_min) / ifu.n_wave
        assert width.to_value(u.angstrom) == pytest.approx(0.28, rel=0.01)

    def test_throughput_peaks_at_the_published_floor(self):
        ifu = IFU.from_config(CONFIG)
        tp = ifu.throughput
        assert tp.wavelength.min() <= ifu.wavelength_min
        assert tp.wavelength.max() >= ifu.wavelength_max
        # "in excess of 45%" peak end-to-end efficiency, Morrissey et al. 2018
        assert np.max(tp.transmission) == pytest.approx(0.45)
        assert np.all((tp.transmission > 0) & (tp.transmission <= 0.45))

    def test_lsf_matches_the_bm_grating_resolving_power(self):
        ifu = IFU.from_config(CONFIG)
        assert ifu.lsf.fwhm.to_value(u.angstrom) == pytest.approx(5000 / 4000, rel=1e-3)

    def test_placeholder_sky_and_dark_are_still_fiducial(self):
        """Pins the fact that sky is an unverified placeholder and dark current
        is a generic (non-KCWI-specific) fiducial: when either is replaced with
        a sourced value this test should be updated on purpose, not pass
        silently."""
        ifu = IFU.from_config(CONFIG)
        assert ifu.sky_electron_rate.to_value(u.electron / u.s) == 0.0
        assert ifu.detector.dark_current.to_value(u.electron / u.s) == pytest.approx(0.0008)

    def test_observes_oii_at_low_redshift(self):
        ifu = IFU.from_config(CONFIG)
        gal = DiscGalaxy(r_e=1.2, inc=55.0, pa=30.0, v_flat=180.0, z=0.33, sigma0=30.0)
        cube = ifu.observe(
            apply_shear(gal, 0.03, 0.0),
            line_flux=1e-15 * u.erg / u.s / u.cm**2,
            line_wavelength=0.372709 * u.um,
        )
        assert cube.flux.shape == (714, 30, 24)
        assert np.all(np.isfinite(cube.flux))
        assert cube.flux.max() > 0
