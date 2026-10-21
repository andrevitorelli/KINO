"""Tests for kino.io.manga.

TestReadMangiaMaps/TestReadMangiaRSS use synthetic FITS fixtures built to
mirror the *real* structure confirmed by downloading real MaNGIA files
this session (see kino/io/manga.py's module docstring) -- fast, no
network/credentials needed, and pin the parsing logic.

TestRegridFibersToGrid checks the regridding math itself against simple,
exactly-predictable invariants (not against real data).

TestAgainstRealFiles is gated on the real, gitignored fixtures this
session cached under data/mangia/ (fetched via kino.io.download's
streaming archive extraction, see that module and TEST_DATA.md) --
picks whichever matched pair happens to be cached (see _find_real_pair),
and is skipped entirely if none is, so CI (which has no such cache) still
passes.
"""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits

from kino.io.manga import (
    MangiaMaps,
    MangiaRSS,
    read_mangia_maps,
    read_mangia_rss,
    regrid_fibers_to_grid,
    regrid_rss_to_cube,
)

DATA_DIR = Path("data/mangia")


def _find_real_pair() -> tuple[Path, Path] | None:
    """Any one matched (rss, maps) pair cached under DATA_DIR, or None --
    picks whichever is available rather than hardcoding one galaxy, since
    which galaxies are cached locally varies (fetched incrementally)."""
    if not DATA_DIR.is_dir():
        return None
    for rss_path in sorted(DATA_DIR.glob("*.cube_RSS.fits")):
        core = rss_path.name.removesuffix(".cube_RSS.fits")
        maps_path = DATA_DIR / f"{core}.cube_maps.fits"
        if maps_path.exists():
            return rss_path, maps_path
    return None


_REAL_PAIR = _find_real_pair()
REAL_RSS, REAL_MAPS = _REAL_PAIR if _REAL_PAIR is not None else (None, None)

_TYPE_NAMES = ["Av_T", "MASS", "VEL", "SFR", "DISP"]


def _write_maps_fixture(path, *, ny=5, nx=6, n_type=None, type_names=None):
    type_names = type_names if type_names is not None else _TYPE_NAMES
    n_type = n_type if n_type is not None else len(type_names)
    data = np.arange(n_type * ny * nx, dtype=float).reshape(n_type, ny, nx)
    primary = fits.PrimaryHDU(data=data)
    h = primary.header
    for i, name in enumerate(type_names):
        h[f"TYPE{i}"] = name
    h["CD1_1"] = -0.5 / 3600.0
    h["CD2_2"] = 0.5 / 3600.0
    h["REDSHIFT"] = 0.023
    h["PSF"] = 1.43
    h["FOV"] = 29.8
    fits.HDUList([primary]).writeto(path, overwrite=True)


def _write_rss_fixture(path, *, n_wave=10, n_fiber=7, with_gas=True):
    flux = np.arange(n_wave * n_fiber, dtype=float).reshape(n_wave, n_fiber)
    primary = fits.PrimaryHDU(data=flux)
    h = primary.header
    h["CRVAL2"] = 3749.0
    h["CDELT2"] = 1.0
    h["CRPIX2"] = 1
    h["PSF"] = 1.43
    h["REDSHIFT"] = 0.023
    hdus = [primary]
    if with_gas:
        hdus.append(fits.ImageHDU(data=flux * 0.1, name="Gas emission"))
    rv = np.arange(len(_TYPE_NAMES) * n_fiber, dtype=float).reshape(len(_TYPE_NAMES), n_fiber)
    rv_hdu = fits.ImageHDU(data=rv, name="Real values")
    for i, name in enumerate(_TYPE_NAMES):
        rv_hdu.header[f"TYPE{i}"] = name
    hdus.append(rv_hdu)
    rng = np.random.default_rng(0)
    x_ifu = rng.uniform(-10, 10, n_fiber)
    y_ifu = rng.uniform(-10, 10, n_fiber)
    hdus.append(fits.ImageHDU(data=x_ifu, name="x_ifu"))
    hdus.append(fits.ImageHDU(data=y_ifu, name="y_ifu"))
    fits.HDUList(hdus).writeto(path, overwrite=True)


class TestReadMangiaMaps:
    def test_parses_type_maps_and_metadata(self, tmp_path):
        path = tmp_path / "maps.fits"
        _write_maps_fixture(path, ny=5, nx=6)

        maps = read_mangia_maps(path)
        assert isinstance(maps, MangiaMaps)
        assert set(maps.maps) == set(_TYPE_NAMES)
        assert maps.maps["VEL"].shape == (5, 6)
        assert maps.redshift == pytest.approx(0.023)
        assert maps.psf_fwhm.to_value(u.arcsec) == pytest.approx(1.43)
        assert maps.fov.to_value(u.arcsec) == pytest.approx(29.8)
        assert maps.pixel_scale.to_value(u.arcsec) == pytest.approx(0.5, rel=1e-6)

    def test_type_values_match_the_right_slice(self, tmp_path):
        path = tmp_path / "maps.fits"
        _write_maps_fixture(path, ny=3, nx=3)
        maps = read_mangia_maps(path)
        idx = _TYPE_NAMES.index("DISP")
        expected = np.arange(len(_TYPE_NAMES) * 9, dtype=float).reshape(len(_TYPE_NAMES), 3, 3)[idx]
        assert np.array_equal(maps.maps["DISP"], expected)


class TestReadMangiaRSS:
    def test_parses_flux_wavelength_and_fiber_positions(self, tmp_path):
        path = tmp_path / "rss.fits"
        _write_rss_fixture(path, n_wave=8, n_fiber=6)

        rss = read_mangia_rss(path)
        assert isinstance(rss, MangiaRSS)
        assert rss.flux.shape == (8, 6)
        assert rss.flux_gas.shape == (8, 6)
        assert np.allclose(rss.wavelength.to_value(u.AA), 3749.0 + np.arange(8))
        assert rss.x_fiber.shape == (6,)
        assert rss.y_fiber.shape == (6,)
        assert set(rss.fiber_maps) == set(_TYPE_NAMES)
        assert rss.fiber_maps["VEL"].shape == (6,)
        assert rss.redshift == pytest.approx(0.023)
        assert rss.psf_fwhm.to_value(u.arcsec) == pytest.approx(1.43)

    def test_no_gas_extension_leaves_it_none(self, tmp_path):
        path = tmp_path / "rss.fits"
        _write_rss_fixture(path, with_gas=False)
        rss = read_mangia_rss(path)
        assert rss.flux_gas is None


class TestRegridFibersToGrid:
    def test_single_fiber_fills_the_whole_grid_with_its_own_value(self):
        x = np.array([0.0]) * u.arcsec
        y = np.array([0.0]) * u.arcsec
        values = np.array([42.0])
        out = regrid_fibers_to_grid(
            x, y, values, pixel_scale=1.0 * u.arcsec, nx=5, ny=5, kernel_fwhm=2.0 * u.arcsec
        )
        assert out.shape == (5, 5)
        assert np.allclose(out, 42.0)

    def test_two_fibers_with_equal_values_give_that_value_everywhere(self):
        x = np.array([-5.0, 5.0]) * u.arcsec
        y = np.array([0.0, 0.0]) * u.arcsec
        values = np.array([7.0, 7.0])
        out = regrid_fibers_to_grid(
            x, y, values, pixel_scale=1.0 * u.arcsec, nx=6, ny=6, kernel_fwhm=2.0 * u.arcsec
        )
        assert np.allclose(out, 7.0)

    def test_output_near_a_fiber_is_closer_to_its_value_than_a_distant_one(self):
        x = np.array([-5.0, 5.0]) * u.arcsec
        y = np.array([0.0, 0.0]) * u.arcsec
        values = np.array([0.0, 100.0])
        out = regrid_fibers_to_grid(
            x, y, values, pixel_scale=1.0 * u.arcsec, nx=11, ny=1, kernel_fwhm=2.0 * u.arcsec
        )
        # grid spans x in [-5, 5] at pixel_scale=1, nx=11 -> leftmost near fiber 0,
        # rightmost near fiber 1
        assert out[0, 0] < out[0, -1]
        assert out[0, 0] < 50.0
        assert out[0, -1] > 50.0

    def test_broadcasts_over_a_leading_spectral_axis(self):
        x = np.array([0.0, 1.0]) * u.arcsec
        y = np.array([0.0, 0.0]) * u.arcsec
        values = np.stack([np.array([1.0, 2.0]), np.array([10.0, 20.0])])  # (n_wave=2, n_fiber=2)
        out = regrid_fibers_to_grid(
            x, y, values, pixel_scale=0.5 * u.arcsec, nx=4, ny=4, kernel_fwhm=1.0 * u.arcsec
        )
        assert out.shape == (2, 4, 4)

    def test_mismatched_fiber_count_raises(self):
        x = np.array([0.0, 1.0]) * u.arcsec
        y = np.array([0.0, 1.0]) * u.arcsec
        values = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            regrid_fibers_to_grid(
                x, y, values, pixel_scale=1.0 * u.arcsec, nx=3, ny=3, kernel_fwhm=1.0 * u.arcsec
            )


class TestRegridRssToCube:
    def test_produces_a_datacube_with_the_requested_grid(self, tmp_path):
        path = tmp_path / "rss.fits"
        _write_rss_fixture(path, n_wave=6, n_fiber=5)
        rss = read_mangia_rss(path)

        cube = regrid_rss_to_cube(rss, nx=8, ny=9, pixel_scale=1.0 * u.arcsec)
        assert cube.shape == (6, 9, 8)
        assert cube.flux_unit == rss.flux_unit
        assert cube.redshift == rss.redshift
        assert np.all(np.isfinite(cube.flux))

    def test_default_kernel_fwhm_is_the_rss_psf(self, tmp_path):
        path = tmp_path / "rss.fits"
        _write_rss_fixture(path)
        rss = read_mangia_rss(path)
        cube = regrid_rss_to_cube(rss, nx=4, ny=4, pixel_scale=2.0 * u.arcsec)
        assert "1.43" in cube.meta["kernel_fwhm"]


@pytest.mark.skipif(
    _REAL_PAIR is None,
    reason="no real matched (rss, maps) MaNGIA pair cached locally under data/mangia/ -- "
    "fetch one with kino.io.download.fetch_mangia_members to enable this test",
)
class TestAgainstRealFiles:
    def test_real_maps_file_parses(self):
        maps = read_mangia_maps(REAL_MAPS)
        assert len(maps.maps) == 41
        assert "VEL" in maps.maps and "DISP" in maps.maps

    def test_real_rss_file_parses(self):
        rss = read_mangia_rss(REAL_RSS)
        assert rss.flux.shape[0] == rss.wavelength.shape[0]
        assert rss.x_fiber.shape == rss.y_fiber.shape

    def test_regridded_rss_matches_official_maps_vel_and_disp(self):
        """The real cross-check: regrid the RSS file's per-fibre ground-truth
        VEL/DISP the same way as flux, and compare against the matching
        official 'maps' file's own gridded VEL/DISP for the *same* galaxy --
        validates the regridding geometry independent of any spectral fitting.
        """
        maps = read_mangia_maps(REAL_MAPS)
        rss = read_mangia_rss(REAL_RSS)
        ny, nx = maps.maps["VEL"].shape
        pixel_scale = maps.pixel_scale

        for name in ("VEL", "DISP"):
            regridded = regrid_fibers_to_grid(
                rss.x_fiber, rss.y_fiber, rss.fiber_maps[name],
                pixel_scale=pixel_scale, nx=nx, ny=ny, kernel_fwhm=rss.psf_fwhm,
            )  # fmt: skip
            official = maps.maps[name]
            in_fov = official != 0.0  # outside-FOV spaxels are zero in both
            resid = regridded[in_fov] - official[in_fov]
            scale = np.std(official[in_fov])
            assert np.std(resid) < 0.5 * scale
