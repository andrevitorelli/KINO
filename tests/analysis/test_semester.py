import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from kino.analysis.forecast import ForecastTarget
from kino.analysis.semester import forecast_over_nights, rank_candidates, semester_dates
from kino.galaxy import DiscGalaxy
from kino.instruments.base import LSF, PSF, Detector, Throughput
from kino.instruments.ifu import IFU
from kino.instruments.slit import SlitSpectrograph
from kino.observing import SkyLibrary

Z = 0.05  # Halpha (vacuum 6564.61 A) at ~0.689 um
DATES = ["2026-10-02", "2026-10-03", "2026-10-04"]  # within the NOAA F10.7 table


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


def make_ifu(**kw) -> IFU:
    base = dict(
        pixel_scale=0.1 * u.arcsec, nx=8, ny=8,
        wavelength_min=0.688 * u.um, wavelength_max=0.690 * u.um, n_wave=20,
        psf=PSF(seeing_fwhm=0.02 * u.arcsec), lsf=LSF(resolving_power=8000.0),
        throughput=Throughput(wavelength=[0.5, 0.9] * u.um, transmission=[0.3, 0.3]),
        aperture_diameter=10 * u.m,
        detector=Detector(read_noise=3 * u.electron, dark_current=0 * u.electron / u.s),
        exposure_time=600 * u.s,
    )  # fmt: skip
    base.update(kw)
    return IFU(**base)


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


def make_target(flux=2e-15, **kw):
    gal = DiscGalaxy(r_e=0.25, inc=55.0, pa=30.0, v_flat=180.0, r_t=0.15, z=Z, sigma0=25.0)
    return ForecastTarget(
        SkyCoord("02h00m00s", "-25d00m00s"), gal, flux * u.erg / u.s / u.cm**2, "Halpha", **kw
    )


class TestSemesterDates:
    def test_inclusive_weekly_stride(self):
        dates = semester_dates("2027-01-01", "2027-01-22", step_days=7)
        assert dates == ["2027-01-01", "2027-01-08", "2027-01-15", "2027-01-22"]

    def test_daily_stride(self):
        dates = semester_dates("2027-01-01", "2027-01-03", step_days=1)
        assert dates == ["2027-01-01", "2027-01-02", "2027-01-03"]

    def test_default_step_is_weekly(self):
        dates = semester_dates("2027-01-01", "2027-01-03")
        assert dates == ["2027-01-01"]

    def test_rejects_end_before_start(self):
        with pytest.raises(ValueError, match="before start"):
            semester_dates("2027-02-01", "2027-01-01")

    def test_rejects_non_positive_step(self):
        with pytest.raises(ValueError, match="step_days"):
            semester_dates("2027-01-01", "2027-01-10", step_days=0)


class TestForecastOverNights:
    def test_shapes(self):
        nf = forecast_over_nights(
            make_ifu(), make_target(), DATES, flat_library(),
            g=(0.05, 0.0), seeing_ref=[0.02] * u.arcsec, time_step=240 * u.min,
        )  # fmt: skip
        assert len(nf.dates) == len(DATES)
        assert nf.shear_snr.shape == (len(DATES),)
        assert nf.shear_snr_stat.shape == (len(DATES),)
        assert nf.seeing.shape == (len(DATES),)
        assert nf.airmass.shape == (len(DATES),)
        assert len(nf.time) == len(DATES)

    def test_bright_target_is_detectable_on_at_least_one_night(self):
        nf = forecast_over_nights(
            make_ifu(), make_target(flux=2e-13), DATES, flat_library(),
            g=(0.05, 0.0), seeing_ref=[0.02] * u.arcsec, time_step=240 * u.min,
        )  # fmt: skip
        assert nf.observable_mask.any()
        i = nf.best_night_index
        assert i is not None
        assert np.isfinite(nf.shear_snr[i])
        s = nf.summary()
        assert s["detectable"] is True
        assert s["date"] == nf.dates[i]

    def test_undetectably_faint_target_gives_no_best_night(self):
        nf = forecast_over_nights(
            make_ifu(), make_target(flux=1e-20), DATES, flat_library(),
            g=(0.05, 0.0), seeing_ref=[0.02] * u.arcsec, time_step=240 * u.min,
        )  # fmt: skip
        assert nf.best_night_index is None
        assert nf.summary() == {"detectable": False, "name": ""}
        assert np.all(np.isnan(nf.shear_snr))

    def test_dispatches_to_forecast_slit_for_a_slit_spectrograph(self):
        nf = forecast_over_nights(
            make_slit(), make_target(flux=2e-13), DATES, flat_library(),
            g=(0.05, 0.0), seeing_ref=[0.5] * u.arcsec, time_step=240 * u.min,
        )  # fmt: skip
        assert nf.observable_mask.any()

    def test_out_of_table_month_is_nan_with_one_warning(self):
        with pytest.warns(UserWarning, match="F10.7"):
            nf = forecast_over_nights(
                make_ifu(), make_target(flux=2e-13), ["2030-01-01", "2030-01-02"], flat_library(),
                g=(0.05, 0.0), seeing_ref=[0.02] * u.arcsec, time_step=240 * u.min,
            )  # fmt: skip
        assert np.all(np.isnan(nf.shear_snr))

    def test_custom_solar_flux_fn_covers_out_of_table_dates(self):
        nf = forecast_over_nights(
            make_ifu(), make_target(flux=2e-13), ["2030-01-01"], flat_library(),
            g=(0.05, 0.0), seeing_ref=[0.02] * u.arcsec, time_step=240 * u.min,
            solar_flux_fn=lambda d: 110.0,
        )  # fmt: skip
        assert nf.observable_mask.any()

    def test_structural_mismatch_raises_immediately(self):
        # the line falls outside this instrument's throughput coverage for every night
        bad_throughput = Throughput(wavelength=[0.3, 0.4] * u.um, transmission=[0.3, 0.3])
        bad_ifu = make_ifu(throughput=bad_throughput)
        with pytest.raises(ValueError):
            forecast_over_nights(
                bad_ifu, make_target(flux=2e-13), DATES, flat_library(),
                g=(0.05, 0.0), seeing_ref=[0.02] * u.arcsec, time_step=240 * u.min,
            )  # fmt: skip


class TestRankCandidates:
    def test_brighter_target_ranks_first(self):
        faint = make_target(flux=5e-15, name="faint")
        bright = make_target(flux=5e-13, name="bright")
        results = rank_candidates(
            make_ifu(), [faint, bright], DATES, flat_library(),
            g=(0.05, 0.0), seeing_ref=[0.02] * u.arcsec, time_step=240 * u.min,
        )  # fmt: skip
        assert [r.target.name for r in results] == ["bright", "faint"]

    def test_one_bad_target_does_not_abort_the_batch(self):
        good = make_target(flux=5e-13, name="good")
        bad = ForecastTarget(
            SkyCoord("02h00m00s", "-25d00m00s"),
            DiscGalaxy(r_e=0.25, inc=55.0, pa=30.0, v_flat=180.0, r_t=0.15, z=Z, sigma0=25.0),
            5e-13 * u.erg / u.s / u.cm**2,
            1.0 * u.um,  # way outside the test IFU's throughput coverage
            name="bad",
        )
        results = rank_candidates(
            make_ifu(), [bad, good], DATES, flat_library(),
            g=(0.05, 0.0), seeing_ref=[0.02] * u.arcsec, time_step=240 * u.min,
        )  # fmt: skip
        names = [r.target.name for r in results]
        assert names[0] == "good"
        assert names[-1] == "bad"
        bad_result = next(r for r in results if r.target.name == "bad")
        assert bad_result.summary()["detectable"] is False
