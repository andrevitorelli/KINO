import astropy.units as u
import numpy as np
import pytest

from kino.galaxy import DiscGalaxy
from kino.instruments.ifu import IFU
from kino.lensing import apply_shear

CONFIG = "kino/config/sifs_700r.yaml"
MEASURED_CONFIG = "kino/config/sifs.yaml"


class TestSIFS700RConfig:
    """`sifs_700r.yaml` is the 700R grating setting, needed for Halpha above z ~ 0.11.
    Its throughput redward of 730nm is fiducial, not measured -- see the config's own
    header."""

    def test_geometry_and_detector_match_sifs_yaml(self):
        """Same instrument, different grating tilt -- hardware numbers must agree."""
        r = IFU.from_config(CONFIG)
        m = IFU.from_config(MEASURED_CONFIG)
        assert (r.nx, r.ny) == (m.nx, m.ny)
        assert r.pixel_scale == m.pixel_scale
        assert r.aperture_diameter == m.aperture_diameter
        assert r.detector.read_noise == m.detector.read_noise

    def test_wavelength_window_targets_the_redder_range(self):
        ifu = IFU.from_config(CONFIG)
        assert ifu.wavelength_min.to_value(u.nm) == pytest.approx(740.0)
        assert ifu.wavelength_max.to_value(u.nm) == pytest.approx(760.0)

    def test_throughput_matches_the_measured_700m_curve_up_to_730nm(self):
        r = IFU.from_config(CONFIG)
        m = IFU.from_config(MEASURED_CONFIG)
        n_measured = len(m.throughput.wavelength)
        assert r.throughput.wavelength[:n_measured].to_value(u.um) == pytest.approx(
            m.throughput.wavelength.to_value(u.um)
        )
        assert r.throughput.transmission[:n_measured] == pytest.approx(m.throughput.transmission)

    def test_fiducial_extension_is_flat_at_the_last_measured_value(self):
        ifu = IFU.from_config(CONFIG)
        tp = ifu.throughput.transmission
        last_measured = tp[6]  # 0.73 um, 700M's last real point
        assert last_measured == pytest.approx(0.035, abs=0.001)
        assert tp[7] == pytest.approx(last_measured)

    def test_observes_halpha_at_the_redshift_this_config_is_for(self):
        ifu = IFU.from_config(CONFIG)
        gal = DiscGalaxy(r_e=1.0, inc=55.0, pa=30.0, v_flat=180.0, z=0.15, sigma0=30.0)
        cube = ifu.observe(
            apply_shear(gal, 0.03, 0.0),
            line_flux=1e-14 * u.erg / u.s / u.cm**2,
            line_wavelength=0.656461 * u.um,
        )
        assert np.all(np.isfinite(cube.flux))
        assert cube.flux.max() > 0
