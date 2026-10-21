"""Round-trip tests for kino.io.hdf5 (README.md Phase 4) -- mirrors test_fitsio.py."""

import astropy.units as u
import numpy as np
import pytest

from kino.galaxy import DiscGalaxy
from kino.instruments.ifu import IFU
from kino.instruments.slit import SlitSpectrograph
from kino.io import hdf5
from kino.io.base import Datacube
from kino.lensing import apply_shear


def _make_cube(with_variance=True, with_mask=True, with_meta=True) -> Datacube:
    flux = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    variance = flux + 1.0 if with_variance else None
    mask = (flux > 5) if with_mask else None
    meta = {"a": 1, "b": "two", "q": 3.0 * u.um, "nested_tuple": (1.0 * u.arcsec, 2.0 * u.arcsec)}
    return Datacube(
        flux=flux,
        spectral_axis=np.array([1.0, 2.0]) * u.um,
        spectral_kind="wavelength",
        pixel_scale=0.5 * u.arcsec,
        redshift=0.2,
        flux_unit=u.electron,
        variance=variance,
        mask=mask,
        meta=meta if with_meta else {},
    )


class TestDatacubeRoundTrip:
    def test_round_trip_with_variance_mask_and_meta(self, tmp_path):
        cube = _make_cube()
        path = tmp_path / "cube.h5"
        hdf5.write_datacube(cube, path)
        back = hdf5.read_datacube(path)

        assert np.array_equal(back.flux, cube.flux)
        assert np.array_equal(back.variance, cube.variance)
        assert np.array_equal(back.mask, cube.mask)
        assert back.spectral_kind == cube.spectral_kind
        assert u.allclose(back.spectral_axis, cube.spectral_axis)
        assert back.pixel_scale == cube.pixel_scale
        assert back.redshift == cube.redshift
        assert back.flux_unit == cube.flux_unit
        assert back.meta["a"] == 1
        assert back.meta["b"] == "two"
        assert back.meta["q"] == str(3.0 * u.um)

    def test_round_trip_without_variance_or_mask(self, tmp_path):
        cube = _make_cube(with_variance=False, with_mask=False)
        path = tmp_path / "cube.h5"
        hdf5.write_datacube(cube, path)
        back = hdf5.read_datacube(path)
        assert back.variance is None
        assert back.mask is None

    def test_noiseless_cube_round_trips(self, tmp_path):
        ifu = IFU.from_config("kino/config/sifs.yaml")
        gal = DiscGalaxy(r_e=0.5, inc=55.0, pa=30.0, v_flat=180.0, z=0.1, sigma0=30.0)
        cube = ifu.observe(
            apply_shear(gal, 0.03, 0.0),
            line_flux=1e-15 * u.erg / u.s / u.cm**2,
            line_wavelength=0.65628 * u.um,
            rng=None,
        )
        path = tmp_path / "noiseless.h5"
        hdf5.write_datacube(cube, path)
        back = hdf5.read_datacube(path)
        assert np.array_equal(back.flux, cube.flux)
        assert np.array_equal(back.variance, cube.variance)


class TestSlitObservationRoundTrip:
    def test_round_trip(self, tmp_path):
        slit = SlitSpectrograph.from_config("kino/config/goodman.yaml")
        gal = DiscGalaxy(r_e=0.5, inc=55.0, pa=30.0, v_flat=180.0, z=0.1, sigma0=30.0)
        rng = np.random.default_rng(0)
        obs = slit.observe(
            apply_shear(gal, 0.03, -0.01),
            line_flux=1e-15 * u.erg / u.s / u.cm**2,
            line_wavelength=0.65628 * u.um,
            rng=rng,
        )
        path = tmp_path / "slit.h5"
        hdf5.write_slit_observation(obs, path)
        back = hdf5.read_slit_observation(path)

        assert np.array_equal(back.spectrum.flux, obs.spectrum.flux)
        assert np.array_equal(back.spectrum.variance, obs.spectrum.variance)
        assert np.array_equal(back.image, obs.image)
        assert back.image_pixel_scale == obs.image_pixel_scale
        assert back.slit_width == obs.slit_width
        assert back.slit_length == obs.slit_length
        assert back.slit_pa == pytest.approx(obs.slit_pa)
        assert back.slit_offset[0] == obs.slit_offset[0]
        assert back.slit_offset[1] == obs.slit_offset[1]
