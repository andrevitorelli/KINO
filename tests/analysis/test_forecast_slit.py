import dataclasses

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from kino.analysis.forecast import ForecastTarget, forecast_slit, narrowed_slit
from kino.galaxy import DiscGalaxy
from kino.instruments.base import LSF, PSF, Detector, Throughput
from kino.instruments.slit import SlitSpectrograph
from kino.observing import SOAR, SkyLibrary, night_window

Z = 0.05  # Halpha (vacuum 6564.61 A) at ~0.689 um


def flat_library(glow=20.0, max_nm=1300.0) -> SkyLibrary:
    lam = np.linspace(300.0, max_nm, int(max_nm - 300.0) + 1)
    elong, rho = np.array([20.0, 180.0]), np.array([10.0, 170.0])
    zm, x = np.array([0.0, 80.0]), np.array([1.0, 2.0])
    moon = {
        "lam_nm": lam, "elongation": elong, "target_sep": rho, "zenith_dist_moon": zm, "airmass": x,
        "radiance": np.full((2, 2, 2, 2, len(lam)), 5.0),
    }  # fmt: skip
    airglow = {
        "lam_nm": lam, "airmass": np.array([1.0, 2.5]), "solar_flux": np.array([70.0, 200.0]),
        "radiance": np.full((2, 2, len(lam)), glow),
    }  # fmt: skip
    zodi = {
        "lam_nm": lam, "ecl_lon": np.array([30.0, 180.0]), "ecl_lat": np.array([0.0, 90.0]),
        "radiance": np.full((2, 2, len(lam)), 2.0), "starlight": np.full(len(lam), 1.0),
    }  # fmt: skip
    return SkyLibrary(moon, airglow, zodi)


def make_slit(**kw) -> SlitSpectrograph:
    base = dict(
        slit_width=1.0 * u.arcsec, pixel_scale=0.15 * u.arcsec, n_spatial=20,
        wavelength_min=0.688 * u.um, wavelength_max=0.690 * u.um, n_wave=20,
        psf=PSF(seeing_fwhm=0.5 * u.arcsec), lsf=LSF(resolving_power=8000.0),
        throughput=Throughput(wavelength=[0.5, 0.9] * u.um, transmission=[0.3, 0.3]),
        aperture_diameter=10 * u.m,
        detector=Detector(read_noise=3 * u.electron, dark_current=0 * u.electron / u.s),
        exposure_time=600 * u.s,
    )  # fmt: skip
    base.update(kw)
    return SlitSpectrograph(**base)


def make_target(flux=2e-14, **kw):
    gal = DiscGalaxy(r_e=1.0, inc=55.0, pa=30.0, v_flat=180.0, r_t=0.5, z=Z, sigma0=25.0)
    return ForecastTarget(
        SkyCoord("02h00m00s", "-25d00m00s"), gal, flux * u.erg / u.s / u.cm**2, "Halpha", **kw
    )


@pytest.fixture(scope="module")
def night():
    return night_window(SOAR, "2026-10-03")


def run(night, **kw):
    slit = kw.pop("slit", make_slit())
    target = kw.pop("target", make_target())
    library = kw.pop("library", flat_library())
    args = dict(
        g=(0.05, 0.0), seeing_ref=[0.5, 0.9] * u.arcsec, solar_flux=110.0,
        time_step=180 * u.min, window_kms=500.0,
    )  # fmt: skip
    args.update(kw)
    return forecast_slit(slit, target, night, library, **args)


class TestNarrowedSlit:
    def test_keeps_the_native_dispersion_and_centres_on_the_line(self):
        slit = make_slit(wavelength_min=0.6 * u.um, wavelength_max=0.8 * u.um, n_wave=4000)
        native = (slit.wavelength_max - slit.wavelength_min) / slit.n_wave
        out = narrowed_slit(slit, 0.7 * u.um, 500.0)
        new = (out.wavelength_max - out.wavelength_min) / out.n_wave
        assert new.to_value(u.AA) == pytest.approx(native.to_value(u.AA), rel=1e-9)
        assert 0.5 * (out.wavelength_min + out.wavelength_max).to_value(u.um) == pytest.approx(0.7)
        assert out.n_wave < slit.n_wave
        assert out.slit_width == slit.slit_width  # untouched by re-windowing


class TestForecastSlit:
    def test_shapes_and_below_horizon_points_are_nan(self, night):
        fc = run(night)
        n_t, n_s = len(fc.times), 2
        assert fc.shear_snr.shape == (n_t, n_s)
        assert fc.sigma_g.shape == (n_t, n_s, 2)
        low = ~np.isfinite(fc.airmass) | (fc.airmass > 2.0)
        assert np.all(np.isnan(fc.shear_snr[low]))
        ok = ~low
        assert ok.any()

    def test_slit_losses_lower_the_snr_for_a_wide_galaxy(self, night):
        wide = run(night, target=make_target())
        narrow_slit = run(night, slit=make_slit(slit_width=5.0 * u.arcsec))
        assert np.nanmax(narrow_slit.line_snr) >= np.nanmax(wide.line_snr)

    def test_brighter_target_gives_higher_snr(self, night):
        faint = run(night, target=make_target(flux=2e-15))
        bright = run(night, target=make_target(flux=1e-13))
        assert np.nanmax(bright.shear_snr_stat) > np.nanmax(faint.shear_snr_stat)

    def test_moments_estimator_is_rejected(self, night):
        with pytest.raises(ValueError, match="cube"):
            run(night, estimator="moments")

    def test_line_outside_the_throughput_range_raises(self, night):
        slit = make_slit(
            throughput=Throughput(wavelength=[0.4, 0.5] * u.um, transmission=[0.3, 0.3])
        )
        with pytest.raises(ValueError, match="outside its tabulated coverage"):
            run(night, slit=slit)

    def test_does_not_mutate_the_instrument(self, night):
        slit = make_slit()
        before = dataclasses.replace(slit)
        run(night, slit=slit)
        assert slit == before

    def test_meta_reports_the_slit_instrument(self, night):
        fc = run(night)
        assert fc.meta["estimator"] == "cube"
        assert fc.meta["exposure_total"].to_value(u.s) == pytest.approx(600.0)
