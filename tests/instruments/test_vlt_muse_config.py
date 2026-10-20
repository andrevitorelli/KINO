import astropy.units as u
import numpy as np
import pytest

from kino.galaxy import DiscGalaxy
from kino.instruments.ifu import IFU
from kino.lensing import apply_shear

CONFIG = "kino/config/vlt_muse.yaml"


class TestVLTMUSEConfig:
    def test_geometry_is_a_crop_of_the_real_wfm_field(self):
        ifu = IFU.from_config(CONFIG)
        assert (ifu.nx, ifu.ny) == (50, 50)
        assert ifu.pixel_scale.to_value(u.arcsec) == pytest.approx(0.20)
        # 10" x 10" crop; the real WFM field is 59.9" x 60.0"
        assert ifu.nx * ifu.pixel_scale.to_value(u.arcsec) == pytest.approx(10.0)
        assert ifu.ny * ifu.pixel_scale.to_value(u.arcsec) == pytest.approx(10.0)

    def test_detector_and_aperture(self):
        ifu = IFU.from_config(CONFIG)
        assert ifu.detector.read_noise.to_value(u.electron) == pytest.approx(3.0)
        assert ifu.aperture_diameter.to_value(u.m) == pytest.approx(8.2)
        assert ifu.obstruction_fraction == pytest.approx(0.013)

    def test_channel_width_matches_the_published_spectral_sampling(self):
        ifu = IFU.from_config(CONFIG)
        width = (ifu.wavelength_max - ifu.wavelength_min) / ifu.n_wave
        assert width.to_value(u.angstrom) == pytest.approx(1.25, rel=0.01)

    def test_lsf_matches_the_quoted_mean_resolving_power(self):
        ifu = IFU.from_config(CONFIG)
        assert ifu.lsf.fwhm.to_value(u.angstrom) == pytest.approx(7000 / 3000, rel=1e-3)

    def test_throughput_matches_the_eso_nutshell_figures(self):
        ifu = IFU.from_config(CONFIG)
        tp = ifu.throughput
        assert tp.wavelength.min() <= ifu.wavelength_min
        assert tp.wavelength.max() >= ifu.wavelength_max
        # ESO's own quoted WFM throughput: 14% (480nm), 35% (750nm), 14% (930nm)
        assert np.max(tp.transmission) == pytest.approx(0.35)
        assert np.min(tp.transmission) == pytest.approx(0.14)

    def test_placeholder_sky_and_detector_noise_are_still_fiducial(self):
        """Pins the fact that sky, read noise and dark current are all
        unverified placeholders: when any is replaced with a sourced value
        this test should be updated on purpose, not pass silently."""
        ifu = IFU.from_config(CONFIG)
        assert ifu.sky_electron_rate.to_value(u.electron / u.s) == 0.0
        assert ifu.detector.dark_current.to_value(u.electron / u.s) == pytest.approx(0.0005)

    def test_observes_halpha_at_low_redshift(self):
        ifu = IFU.from_config(CONFIG)
        gal = DiscGalaxy(r_e=1.0, inc=55.0, pa=30.0, v_flat=180.0, z=0.061, sigma0=30.0)
        cube = ifu.observe(
            apply_shear(gal, 0.03, 0.0),
            line_flux=1e-15 * u.erg / u.s / u.cm**2,
            line_wavelength=0.65628 * u.um,
        )
        assert cube.flux.shape == (80, 50, 50)
        assert np.all(np.isfinite(cube.flux))
        assert cube.flux.max() > 0
