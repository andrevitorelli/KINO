import astropy.units as u
import numpy as np
import pytest

from kino.galaxy import DiscGalaxy
from kino.instruments.ifu import IFU
from kino.lensing import apply_shear

CONFIG = "kino/config/gemini_nifs.yaml"


class TestGeminiNIFSConfig:
    def test_geometry_matches_the_real_slice_count(self):
        ifu = IFU.from_config(CONFIG)
        assert (ifu.nx, ifu.ny) == (29, 29)
        assert ifu.pixel_scale.to_value(u.arcsec) == pytest.approx(0.103)
        # real field is 3" x 3"; modelled as the real slice count at the
        # coarser cross-slice sampling in both directions (see header note)
        assert ifu.nx * ifu.pixel_scale.to_value(u.arcsec) == pytest.approx(3.0, rel=0.02)

    def test_detector_and_aperture(self):
        ifu = IFU.from_config(CONFIG)
        assert ifu.detector.read_noise.to_value(u.electron) == pytest.approx(4.7)
        assert ifu.aperture_diameter.to_value(u.m) == pytest.approx(8.1)
        assert ifu.obstruction_fraction == pytest.approx(0.016)

    def test_lsf_matches_the_hk_grating_resolving_power(self):
        ifu = IFU.from_config(CONFIG)
        assert ifu.lsf.resolving_power == pytest.approx(5290)

    def test_psf_uses_the_quoted_ao_corrected_fwhm(self):
        ifu = IFU.from_config(CONFIG)
        assert ifu.psf.seeing_fwhm.to_value(u.arcsec) == pytest.approx(0.1)

    def test_placeholder_sky_dark_and_throughput_are_still_fiducial(self):
        """Pins the fact that sky, dark current and throughput are all
        unverified placeholders: when any is replaced with a sourced value
        this test should be updated on purpose, not pass silently."""
        ifu = IFU.from_config(CONFIG)
        assert ifu.sky_electron_rate.to_value(u.electron / u.s) == pytest.approx(3.0)
        assert ifu.detector.dark_current.to_value(u.electron / u.s) == pytest.approx(0.01)
        assert np.max(ifu.throughput.transmission) == pytest.approx(0.10)

    def test_observes_a_redshifted_disc_in_k_band(self):
        ifu = IFU.from_config(CONFIG)
        gal = DiscGalaxy(r_e=0.3, inc=55.0, pa=30.0, v_flat=180.0, z=0.3, sigma0=30.0)
        cube = ifu.observe(
            apply_shear(gal, 0.03, 0.0),
            line_flux=1e-15 * u.erg / u.s / u.cm**2,
            line_wavelength=2.0 * u.um,
        )
        assert cube.flux.shape == (70, 29, 29)
        assert np.all(np.isfinite(cube.flux))
        assert cube.flux.max() > 0
