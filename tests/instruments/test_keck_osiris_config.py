import astropy.units as u
import numpy as np
import pytest

from kino.galaxy import DiscGalaxy
from kino.instruments.ifu import IFU
from kino.lensing import apply_shear

CONFIG = "kino/config/keck_osiris.yaml"


class TestKeckOSIRISConfig:
    def test_geometry_matches_the_050_plate_scale_kn3_field(self):
        ifu = IFU.from_config(CONFIG)
        assert (ifu.nx, ifu.ny) == (48, 64)
        assert ifu.pixel_scale.to_value(u.arcsec) == pytest.approx(0.050)
        # Kn3 field at the 0.050" plate scale: ~2.4" x 3.2"
        assert ifu.nx * ifu.pixel_scale.to_value(u.arcsec) == pytest.approx(2.4)
        assert ifu.ny * ifu.pixel_scale.to_value(u.arcsec) == pytest.approx(3.2)

    def test_detector_and_aperture(self):
        ifu = IFU.from_config(CONFIG)
        assert ifu.detector.dark_current.to_value(u.electron / u.s) == pytest.approx(0.025)
        assert ifu.aperture_diameter.to_value(u.m) == pytest.approx(9.96)
        assert ifu.obstruction_fraction == pytest.approx(0.0)

    def test_lsf_matches_the_fine_plate_scale_resolving_power(self):
        ifu = IFU.from_config(CONFIG)
        assert ifu.lsf.resolving_power == pytest.approx(3800)

    def test_psf_uses_the_computed_diffraction_limit(self):
        ifu = IFU.from_config(CONFIG)
        # 1.22 * lambda / D at 2.2 um, Keck's 9.96 m effective aperture
        expected = (1.22 * 2.2e-6 / 9.96) * u.rad
        assert ifu.psf.seeing_fwhm.to_value(u.arcsec) == pytest.approx(
            expected.to_value(u.arcsec), rel=0.02
        )

    def test_placeholder_sky_read_noise_and_throughput_are_still_fiducial(self):
        """Pins the fact that sky, read noise and throughput are all
        unverified placeholders: when any is replaced with a sourced value
        this test should be updated on purpose, not pass silently."""
        ifu = IFU.from_config(CONFIG)
        assert ifu.sky_electron_rate.to_value(u.electron / u.s) == pytest.approx(3.0)
        assert ifu.detector.read_noise.to_value(u.electron) == pytest.approx(10.0)
        assert np.max(ifu.throughput.transmission) == pytest.approx(0.15)

    def test_observes_halpha_at_high_redshift(self):
        ifu = IFU.from_config(CONFIG)
        gal = DiscGalaxy(r_e=0.15, inc=55.0, pa=30.0, v_flat=180.0, z=2.352, sigma0=30.0)
        cube = ifu.observe(
            apply_shear(gal, 0.03, 0.0),
            line_flux=1e-15 * u.erg / u.s / u.cm**2,
            line_wavelength=0.65628 * u.um,
        )
        assert cube.flux.shape == (70, 64, 48)
        assert np.all(np.isfinite(cube.flux))
        assert cube.flux.max() > 0
