"""Invariant tests for kino.io.base.Datacube (TEST_DATA.md Section 5)."""

import astropy.units as u
import numpy as np
import pytest

from kino.io.base import Datacube


def _make_cube(**overrides):
    n_spec, n_y, n_x = 4, 3, 2
    defaults = dict(
        flux=np.zeros((n_spec, n_y, n_x)),
        spectral_axis=np.linspace(-100, 100, n_spec) * u.km / u.s,
        spectral_kind="velocity",
        pixel_scale=0.5 * u.arcsec,
        redshift=0.1,
        flux_unit=u.Jy,
    )
    defaults.update(overrides)
    return Datacube(**defaults)


def test_valid_cube_roundtrips_shape():
    cube = _make_cube()
    assert cube.shape == (4, 3, 2)
    assert cube.flux.shape == cube.shape


def test_rejects_non_3d_flux():
    with pytest.raises(ValueError):
        _make_cube(flux=np.zeros((4, 3)))


def test_rejects_mismatched_spectral_axis_length():
    with pytest.raises(ValueError):
        _make_cube(spectral_axis=np.linspace(-100, 100, 5) * u.km / u.s)


def test_rejects_bad_spectral_kind():
    with pytest.raises(ValueError):
        _make_cube(spectral_kind="frequency")


def test_rejects_wavelength_axis_in_velocity_units():
    with pytest.raises(u.UnitsError):
        _make_cube(
            spectral_kind="wavelength",
            spectral_axis=np.linspace(6550, 6580, 4) * u.km / u.s,
        )


def test_wavelength_axis_accepts_length_units():
    cube = _make_cube(
        spectral_kind="wavelength",
        spectral_axis=np.linspace(6550, 6580, 4) * u.AA,
    )
    assert cube.spectral_kind == "wavelength"


def test_rejects_non_angle_pixel_scale():
    with pytest.raises(u.UnitsError):
        _make_cube(pixel_scale=0.5 * u.km)


def test_rejects_mismatched_variance_shape():
    with pytest.raises(ValueError):
        _make_cube(variance=np.zeros((4, 3, 3)))


def test_rejects_mismatched_mask_shape():
    with pytest.raises(ValueError):
        _make_cube(mask=np.zeros((4, 3, 3), dtype=bool))
