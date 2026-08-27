"""Tests for kino.io.manga, against a synthetic FITS fixture built to mimic
the *documented* MaNGIA cube structure (see kino/io/manga.py's module
docstring) -- not a real downloaded file, since none was available to this
session (no TNG-project API key). These tests pin down the reader's parsing
*logic*; they cannot confirm it matches a real MaNGIA file's actual layout.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits

from kino.io.manga import DEFAULT_FLUX_UNIT, DEFAULT_PIXEL_SCALE, _wavelength_axis, read_mangia_cube


def _write_cube(
    path,
    *,
    flux,
    error=None,
    mask=None,
    gas=None,
    wave=None,
    header_extra=None,
):
    primary = fits.PrimaryHDU(data=flux.astype(np.float32))
    if header_extra:
        for k, v in header_extra.items():
            primary.header[k] = v
    hdus = [primary]
    if error is not None:
        hdus.append(fits.ImageHDU(data=error.astype(np.float32), name="ERROR"))
    if mask is not None:
        hdus.append(fits.ImageHDU(data=mask.astype(np.int16), name="MASK"))
    if gas is not None:
        hdus.append(fits.ImageHDU(data=gas.astype(np.float32), name="GAS"))
    if wave is not None:
        hdus.append(fits.ImageHDU(data=wave.astype(np.float64), name="WAVE"))
    fits.HDUList(hdus).writeto(path, overwrite=True)


class TestFluxVarianceMask:
    def test_reads_flux_and_variance(self, tmp_path):
        n_spec, ny, nx = 5, 3, 4
        flux = np.arange(n_spec * ny * nx, dtype=float).reshape(n_spec, ny, nx)
        error = np.full((n_spec, ny, nx), 2.0)
        wave = np.linspace(4000.0, 5000.0, n_spec)
        path = tmp_path / "cube.fits"
        _write_cube(path, flux=flux, error=error, wave=wave)

        cube = read_mangia_cube(path)
        assert np.allclose(cube.flux, flux, atol=1e-3)
        assert np.allclose(cube.variance, error**2, atol=1e-3)

    def test_mask_polarity_is_inverted(self, tmp_path):
        # MASK is documented as a "valid" mask (True/1 = good pixel);
        # Datacube.mask is True = invalid, so the reader must invert it.
        n_spec, ny, nx = 4, 2, 2
        flux = np.ones((n_spec, ny, nx))
        valid = np.ones((n_spec, ny, nx), dtype=int)
        valid[:, 0, 0] = 0  # one bad spaxel
        wave = np.linspace(4000.0, 5000.0, n_spec)
        path = tmp_path / "cube.fits"
        _write_cube(path, flux=flux, mask=valid, wave=wave)

        cube = read_mangia_cube(path)
        assert cube.mask[:, 0, 0].all()  # invalid where file said valid=0
        assert not cube.mask[:, 1, 1].any()  # valid elsewhere

    def test_no_error_or_mask_extension_leaves_them_none(self, tmp_path):
        n_spec, ny, nx = 3, 2, 2
        flux = np.ones((n_spec, ny, nx))
        wave = np.linspace(4000.0, 5000.0, n_spec)
        path = tmp_path / "cube.fits"
        _write_cube(path, flux=flux, wave=wave)

        cube = read_mangia_cube(path)
        assert cube.variance is None
        assert cube.mask is None

    def test_rejects_non_3d_primary_hdu(self, tmp_path):
        path = tmp_path / "cube.fits"
        fits.HDUList([fits.PrimaryHDU(data=np.ones((3, 3)))]).writeto(path, overwrite=True)
        with pytest.raises(ValueError):
            read_mangia_cube(path)


class TestWavelengthAxis:
    def test_prefers_wave_extension(self, tmp_path):
        n_spec, ny, nx = 4, 2, 2
        flux = np.ones((n_spec, ny, nx))
        wave = np.array([4000.0, 4500.5, 5001.0, 5500.0])
        path = tmp_path / "cube.fits"
        # also set a *different* CRVAL3/CDELT3, to confirm WAVE wins
        _write_cube(
            path, flux=flux, wave=wave, header_extra={"CRVAL3": 1.0, "CDELT3": 1.0, "CRPIX3": 1.0}
        )

        cube = read_mangia_cube(path)
        assert np.allclose(cube.spectral_axis.to_value(u.AA), wave)

    def test_falls_back_to_linear_wcs(self, tmp_path):
        n_spec, ny, nx = 5, 2, 2
        flux = np.ones((n_spec, ny, nx))
        path = tmp_path / "cube.fits"
        _write_cube(path, flux=flux, header_extra={"CRVAL3": 4000.0, "CDELT3": 2.0, "CRPIX3": 1.0})

        cube = read_mangia_cube(path)
        expected = 4000.0 + np.arange(n_spec) * 2.0
        assert np.allclose(cube.spectral_axis.to_value(u.AA), expected)

    def test_no_wave_no_wcs_raises_when_length_mismatches_documented_grid(self, tmp_path):
        n_spec, ny, nx = 5, 2, 2  # documented default grid has 6603 elements, not 5
        flux = np.ones((n_spec, ny, nx))
        path = tmp_path / "cube.fits"
        _write_cube(path, flux=flux)
        with pytest.raises(ValueError):
            read_mangia_cube(path)

    def test_documented_default_grid_helper_matches_expected_length(self):
        # exercise the fallback grid itself without building a 6603-channel cube
        header = fits.Header()
        axis = _wavelength_axis(header, wave_hdu=None, n_spec=6603)
        assert axis.unit == u.AA
        assert np.isclose(axis[0].to_value(u.AA), 3749.0)
        assert np.isclose(axis[-1].to_value(u.AA), 10351.0)


class TestPixelScaleAndRedshift:
    def test_pixel_scale_from_header(self, tmp_path):
        n_spec, ny, nx = 3, 2, 2
        flux = np.ones((n_spec, ny, nx))
        wave = np.linspace(4000.0, 5000.0, n_spec)
        cdelt_deg = 0.5 / 3600.0  # 0.5 arcsec in degrees
        path = tmp_path / "cube.fits"
        _write_cube(
            path, flux=flux, wave=wave, header_extra={"CDELT1": -cdelt_deg, "CDELT2": cdelt_deg}
        )

        cube = read_mangia_cube(path)
        assert np.isclose(cube.pixel_scale.to_value(u.arcsec), 0.5, rtol=1e-6)

    def test_pixel_scale_falls_back_to_default(self, tmp_path):
        n_spec, ny, nx = 3, 2, 2
        flux = np.ones((n_spec, ny, nx))
        wave = np.linspace(4000.0, 5000.0, n_spec)
        path = tmp_path / "cube.fits"
        _write_cube(path, flux=flux, wave=wave)

        cube = read_mangia_cube(path)
        assert cube.pixel_scale == DEFAULT_PIXEL_SCALE

    def test_redshift_keyword_priority(self, tmp_path):
        n_spec, ny, nx = 3, 2, 2
        flux = np.ones((n_spec, ny, nx))
        wave = np.linspace(4000.0, 5000.0, n_spec)
        path = tmp_path / "cube.fits"
        _write_cube(path, flux=flux, wave=wave, header_extra={"REDSHIFT": 0.03, "Z": 0.05})

        cube = read_mangia_cube(path)
        assert cube.redshift == pytest.approx(0.03)

    def test_redshift_defaults_to_zero(self, tmp_path):
        n_spec, ny, nx = 3, 2, 2
        flux = np.ones((n_spec, ny, nx))
        wave = np.linspace(4000.0, 5000.0, n_spec)
        path = tmp_path / "cube.fits"
        _write_cube(path, flux=flux, wave=wave)

        cube = read_mangia_cube(path)
        assert cube.redshift == 0.0


class TestOverridesAndExtras:
    def test_explicit_overrides_win_over_header_and_defaults(self, tmp_path):
        n_spec, ny, nx = 3, 2, 2
        flux = np.ones((n_spec, ny, nx))
        wave = np.linspace(4000.0, 5000.0, n_spec)
        path = tmp_path / "cube.fits"
        _write_cube(path, flux=flux, wave=wave, header_extra={"REDSHIFT": 0.03})

        cube = read_mangia_cube(path, pixel_scale=1.2 * u.arcsec, flux_unit=u.Jy, redshift=0.9)
        assert cube.pixel_scale == 1.2 * u.arcsec
        assert cube.flux_unit == u.Jy
        assert cube.redshift == 0.9

    def test_default_flux_unit(self, tmp_path):
        n_spec, ny, nx = 3, 2, 2
        flux = np.ones((n_spec, ny, nx))
        wave = np.linspace(4000.0, 5000.0, n_spec)
        path = tmp_path / "cube.fits"
        _write_cube(path, flux=flux, wave=wave)

        cube = read_mangia_cube(path)
        assert cube.flux_unit == DEFAULT_FLUX_UNIT

    def test_include_gas_stashes_gas_flux_in_meta(self, tmp_path):
        n_spec, ny, nx = 3, 2, 2
        flux = np.ones((n_spec, ny, nx))
        gas = 2.0 * np.ones((n_spec, ny, nx))
        wave = np.linspace(4000.0, 5000.0, n_spec)
        path = tmp_path / "cube.fits"
        _write_cube(path, flux=flux, gas=gas, wave=wave)

        cube = read_mangia_cube(path, include_gas=True)
        assert np.allclose(cube.meta["gas_flux"], gas)

    def test_meta_flags_unverified_status(self, tmp_path):
        n_spec, ny, nx = 3, 2, 2
        flux = np.ones((n_spec, ny, nx))
        wave = np.linspace(4000.0, 5000.0, n_spec)
        path = tmp_path / "cube.fits"
        _write_cube(path, flux=flux, wave=wave)

        cube = read_mangia_cube(path)
        assert cube.meta["unverified"] is True
