"""Invariant tests for kino.io.kinms (TEST_DATA.md Tier A / Section 5).

These pin down the two conventions that matter most before this generator is
trusted for injection-recovery tests: units/shape, and the position-angle
sign mapping documented in kino/io/kinms.py's module docstring.
"""

import astropy.units as u
import numpy as np
import pytest

pytest.importorskip("kinms")

from kino.io.kinms import generate_disc_cube  # noqa: E402

COMMON = dict(
    field_of_view=40 * u.arcsec,
    pixel_scale=2 * u.arcsec,
    velocity_range=600 * u.km / u.s,
    channel_width=20 * u.km / u.s,
    beam_size=2 * u.arcsec,
    sb_rad=[0, 15, 20] * u.arcsec,
    sb_prof=[1.0, 1.0, 0.0],
    vel_rad=[0, 5, 20] * u.arcsec,
    vel_prof=[0, 150, 150] * u.km / u.s,
    n_clouds=100_000,
)


def _flux_weighted_velocity(cube):
    v = cube.spectral_axis.to_value(u.km / u.s)[:, None, None]
    total = cube.flux.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_vel = (cube.flux * v).sum(axis=0) / total
    return mean_vel, total  # each (n_y, n_x)


def test_shape_and_units():
    rng = np.random.default_rng(0)
    cube = generate_disc_cube(inc=45, pa=0, gas_sigma=10 * u.km / u.s, rng=rng, **COMMON)

    n_chan = round((COMMON["velocity_range"] / COMMON["channel_width"]).to_value(u.one))
    n_pix = round((COMMON["field_of_view"] / COMMON["pixel_scale"]).to_value(u.one))
    assert cube.shape == (n_chan, n_pix, n_pix)
    assert cube.spectral_kind == "velocity"
    assert cube.spectral_axis.unit.is_equivalent(u.km / u.s)
    assert cube.pixel_scale == COMMON["pixel_scale"]


def test_face_on_disc_has_no_systematic_velocity_gradient():
    """inc=0 => sin(i)=0 => no rotational contribution to Vlos anywhere."""
    rng = np.random.default_rng(1)
    cube = generate_disc_cube(
        inc=0, pa=0, gas_sigma=0 * u.km / u.s, v_sys=0 * u.km / u.s, rng=rng, **COMMON
    )
    mean_vel, total = _flux_weighted_velocity(cube)
    signal = total > (0.05 * total.max())
    assert signal.any()
    assert np.nanmax(np.abs(mean_vel[signal])) < COMMON["channel_width"].to_value(u.km / u.s)


def test_position_angle_convention():
    """Kinematic major-axis LINE tracks KINO's East-of-North pa (see module
    docstring in kino/io/kinms.py) -- pa=0 puts the velocity gradient along
    the spatial y-axis (North), pa=90 puts it along the spatial x-axis
    (East). Only the axis line is asserted, not which end is redshifted
    (that sub-convention is not yet fixed anywhere in KINO)."""
    rng0 = np.random.default_rng(2)
    rng90 = np.random.default_rng(2)
    cube_pa0 = generate_disc_cube(inc=85, pa=0, gas_sigma=5 * u.km / u.s, rng=rng0, **COMMON)
    cube_pa90 = generate_disc_cube(inc=85, pa=90, gas_sigma=5 * u.km / u.s, rng=rng90, **COMMON)

    def axis_spreads(cube):
        mean_vel, total = _flux_weighted_velocity(cube)
        signal = total > (0.05 * total.max())
        ny, nx = mean_vel.shape
        cy, cx = ny // 2, nx // 2
        row = np.where(signal[cy, :], mean_vel[cy, :], np.nan)  # varies with x
        col = np.where(signal[:, cx], mean_vel[:, cx], np.nan)  # varies with y
        return np.nanstd(row), np.nanstd(col)

    x_spread_0, y_spread_0 = axis_spreads(cube_pa0)
    x_spread_90, y_spread_90 = axis_spreads(cube_pa90)

    # pa=0: gradient along y (North), flat along x.
    assert y_spread_0 > x_spread_0
    # pa=90: gradient along x (East), flat along y.
    assert x_spread_90 > y_spread_90
