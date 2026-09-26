import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from kino.io.spectrum import LINES, Spectrum1D, measure_line_flux, read_spectrum

FLAM = u.erg / u.s / u.cm**2 / u.AA
Z = 0.35


def synthetic(flux=4e-16, sigma_kms=120.0, cont=2e-18, noise=1.5e-18, z=Z, dv=0.0, seed=0):
    """Halpha at redshift z: Gaussian of integrated `flux` erg/s/cm2 on a flat continuum."""
    rng = np.random.default_rng(seed)
    lam = np.arange(7000.0, 10500.0, 2.0)
    mu = LINES["Halpha"] * (1 + z) * (1 + dv / 299792.458)
    sig = mu * sigma_kms / 299792.458
    y = flux / (sig * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((lam - mu) / sig) ** 2) + cont
    y = y + rng.normal(0, noise, lam.shape)
    return lam, y, np.full_like(lam, noise)


class TestMeasureLineFlux:
    def _spec(self, **kw):
        lam, y, e = synthetic(**kw)
        return Spectrum1D(lam * u.AA, y * FLAM, e * FLAM)

    def test_recovers_the_injected_flux_within_its_error(self):
        m = measure_line_flux(self._spec(), Z)
        assert m.flux.value == pytest.approx(4e-16, abs=4 * m.flux_err.value)
        assert m.flux.unit == u.erg / u.s / u.cm**2
        assert m.snr > 5
        assert 0.01 < m.flux_err.value / m.flux.value < 0.1  # a few-percent error

    def test_recovers_the_width_and_velocity_offset(self):
        m = measure_line_flux(self._spec(dv=150.0), Z)
        assert m.velocity_offset.to_value(u.km / u.s) == pytest.approx(150.0, abs=30.0)
        assert m.sigma_velocity.to_value(u.km / u.s) == pytest.approx(120.0, rel=0.25)

    def test_unbiased_over_many_noise_realisations(self):
        fluxes = [measure_line_flux(self._spec(seed=s), Z).flux.value for s in range(40)]
        sem = np.std(fluxes) / np.sqrt(len(fluxes))
        assert np.mean(fluxes) == pytest.approx(4e-16, abs=4 * sem)

    def test_pull_of_the_reported_error_is_about_one(self):
        pulls = []
        for s in range(60):
            m = measure_line_flux(self._spec(seed=s), Z)
            pulls.append((m.flux.value - 4e-16) / m.flux_err.value)
        assert np.std(pulls) == pytest.approx(1.0, abs=0.25)

    def test_works_without_errors_using_the_residual_scatter(self):
        lam, y, _ = synthetic()
        m = measure_line_flux(Spectrum1D(lam * u.AA, y * FLAM), Z)
        assert m.flux.value == pytest.approx(4e-16, abs=5 * m.flux_err.value)

    def test_aperture_correction_scales_flux_and_error(self):
        base = measure_line_flux(self._spec(), Z)
        big = measure_line_flux(self._spec(), Z, aperture_correction=2.5)
        assert big.flux.value == pytest.approx(2.5 * base.flux.value)
        assert big.flux_err.value == pytest.approx(2.5 * base.flux_err.value)
        with pytest.raises(ValueError, match=">= 1"):
            measure_line_flux(self._spec(), Z, aperture_correction=0.5)

    def test_line_by_wavelength_and_name_agree(self):
        a = measure_line_flux(self._spec(), Z, "Halpha")
        b = measure_line_flux(self._spec(), Z, LINES["Halpha"] * u.AA)
        assert a.flux == b.flux

    def test_line_outside_coverage_raises(self):
        with pytest.raises(ValueError, match="coverage"):
            measure_line_flux(self._spec(), 1.5)

    def test_no_line_gives_a_low_snr_measurement(self):
        m = measure_line_flux(self._spec(flux=0.0), Z)
        assert m.snr < 4


class TestReadSpectrum:
    def _write_ascii(self, tmp_path, header=True, three=True):
        lam, y, e = synthetic()
        p = tmp_path / "spec.txt"
        rows = [
            f"{a:.3f} {b:.6e} {c:.6e}" if three else f"{a:.3f} {b:.6e}"
            for a, b, c in zip(lam, y, e, strict=True)
        ]
        p.write_text(("# wavelength flux error\n" if header else "") + "\n".join(rows) + "\n")
        return p

    def test_headerless_ascii_needs_explicit_units(self, tmp_path):
        p = self._write_ascii(tmp_path, header=False)
        with pytest.raises(ValueError, match="no unit"):
            read_spectrum(p)
        with pytest.raises(ValueError, match="flux has no unit"):
            read_spectrum(p, wavelength_unit="Angstrom")
        s = read_spectrum(p, wavelength_unit="Angstrom", flux_unit=FLAM)
        assert s.wavelength.unit == u.AA
        assert s.error is not None
        assert len(s.flux) == len(synthetic()[0])

    def test_two_column_ascii(self, tmp_path):
        p = self._write_ascii(tmp_path, header=False, three=False)
        s = read_spectrum(p, wavelength_unit="Angstrom", flux_unit=FLAM)
        assert s.error is None

    def test_sdss_style_unit_is_not_guessed(self, tmp_path):
        p = self._write_ascii(tmp_path)
        s = read_spectrum(p, wavelength_unit="nm", flux_unit=1e-17 * FLAM)
        assert s.flux.unit == u.Unit(1e-17 * FLAM)
        assert s.wavelength.unit == u.AA  # stored in Angstrom
        assert s.wavelength[0].value == pytest.approx(70000.0)  # 7000 "nm" -> A: units are honoured

    def test_fits_table_uses_units_from_the_file_and_refuses_conflicts(self, tmp_path):
        lam, y, e = synthetic()
        t = Table({"WAVELENGTH": lam * u.AA, "FLUX": y * FLAM, "ERR": e * FLAM})
        p = tmp_path / "spec.fits"
        t.write(p)
        s = read_spectrum(p)
        assert s.flux.unit == FLAM
        assert s.error.unit == FLAM
        with pytest.raises(ValueError, match="refusing to guess"):
            read_spectrum(p, flux_unit=1e-17 * FLAM)

    def test_ivar_becomes_an_error(self, tmp_path):
        lam, y, e = synthetic()
        t = Table({"wavelength": lam * u.AA, "flux": y * FLAM})
        t["ivar"] = 1.0 / e**2
        p = tmp_path / "ivar.fits"
        t.write(p)
        s = read_spectrum(p)
        assert s.error.value == pytest.approx(e)

    def test_loglam_is_converted(self, tmp_path):
        lam, y, _ = synthetic()
        t = Table({"loglam": np.log10(lam), "flux": y * FLAM})
        p = tmp_path / "sdss.fits"
        t.write(p)
        s = read_spectrum(p)
        assert s.wavelength.value == pytest.approx(lam)

    def test_nans_are_dropped_and_wavelengths_sorted(self, tmp_path):
        t = Table(
            {"wavelength": [3.0, 1.0, 2.0, 4.0] * u.AA, "flux": [1.0, np.nan, 2.0, 3.0] * FLAM}
        )
        p = tmp_path / "messy.fits"
        t.write(p)
        s = read_spectrum(p)
        assert list(s.wavelength.value) == [2.0, 3.0, 4.0]

    def test_unknown_columns_raise(self, tmp_path):
        t = Table({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        p = tmp_path / "odd.fits"
        t.write(p)
        with pytest.raises(ValueError, match="cannot identify"):
            read_spectrum(p)

    def test_fits_hdu_selection(self, tmp_path):
        lam, y, _ = synthetic()
        t = Table({"wavelength": lam * u.AA, "flux": y * FLAM})
        hdul = fits.HDUList(
            [fits.PrimaryHDU(), fits.BinTableHDU(Table({"x": [1]})), fits.table_to_hdu(t)]
        )
        p = tmp_path / "multi.fits"
        hdul.writeto(p)
        assert len(read_spectrum(p, hdu=2).flux) == len(lam)


class TestSpectrum1D:
    def test_validates_shapes_and_ordering(self):
        with pytest.raises(ValueError):
            Spectrum1D([1.0, 2.0] * u.AA, [1.0] * FLAM)
        with pytest.raises(ValueError, match="increasing"):
            Spectrum1D([2.0, 1.0] * u.AA, [1.0, 1.0] * FLAM)
