import astropy.units as u
import numpy as np
import pytest

from kino.galaxy import DiscGalaxy
from kino.instruments.slit import SlitSpectrograph
from kino.lensing import apply_shear

CONFIG = "kino/config/goodman_red.yaml"
MEASURED_CONFIG = "kino/config/goodman.yaml"


class TestGoodmanRedConfig:
    """`goodman_red.yaml` extends goodman.yaml's real EG 21 throughput with a fiducial
    (not measured) continuation redward of 685nm -- see the config's own header for why
    and what's assumed."""

    def test_geometry_and_detector_match_goodman_yaml(self):
        """Same instrument, different grating setting -- hardware numbers must agree."""
        red = SlitSpectrograph.from_config(CONFIG)
        measured = SlitSpectrograph.from_config(MEASURED_CONFIG)
        assert red.slit_width == measured.slit_width
        assert red.pixel_scale == measured.pixel_scale
        assert red.aperture_diameter == measured.aperture_diameter
        assert red.detector.read_noise == measured.detector.read_noise
        assert red.lsf.resolving_power == measured.lsf.resolving_power

    def test_wavelength_window_targets_the_redder_range(self):
        s = SlitSpectrograph.from_config(CONFIG)
        assert s.wavelength_min.to_value(u.nm) == pytest.approx(740.0)
        assert s.wavelength_max.to_value(u.nm) == pytest.approx(760.0)

    def test_throughput_matches_the_real_curve_up_to_685nm(self):
        """The real, measured part of the curve must be untouched by the fiducial
        extension."""
        red = SlitSpectrograph.from_config(CONFIG)
        measured = SlitSpectrograph.from_config(MEASURED_CONFIG)
        n_measured = len(measured.throughput.wavelength)
        assert red.throughput.wavelength[:n_measured].to_value(u.um) == pytest.approx(
            measured.throughput.wavelength.to_value(u.um)
        )
        assert red.throughput.transmission[:n_measured] == pytest.approx(
            measured.throughput.transmission
        )

    def test_fiducial_extension_is_flat_at_the_last_measured_value(self):
        s = SlitSpectrograph.from_config(CONFIG)
        tp = s.throughput.transmission
        last_measured = tp[10]  # 0.685 um, the last real EG21 point
        assert last_measured == pytest.approx(0.160, abs=0.001)
        assert np.all(tp[10:] == pytest.approx(last_measured))

    def test_observes_halpha_at_the_redshift_this_config_is_for(self):
        s = SlitSpectrograph.from_config(CONFIG)
        gal = DiscGalaxy(r_e=1.0, inc=55.0, pa=30.0, v_flat=180.0, z=0.15, sigma0=30.0)
        obs = s.observe(
            apply_shear(gal, 0.03, 0.0),
            line_flux=1e-14 * u.erg / u.s / u.cm**2,
            line_wavelength=0.656461 * u.um,
        )
        assert np.all(np.isfinite(obs.spectrum.flux))
        assert obs.spectrum.flux.max() > 0
