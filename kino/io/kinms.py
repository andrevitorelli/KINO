"""Generate known-truth datacubes with KinMS, for injection-recovery tests.

Thin wrapper around the ``kinms`` package (Davis et al. 2013, MNRAS 429, 534;
https://github.com/TimothyADavis/KinMSpy) that maps KINO's documented
conventions (``CLAUDE.md``) onto KinMS's cube-generation API and returns a
:class:`kino.io.base.Datacube`. This is Tier A of the validation ladder in
``TEST_DATA.md``: cubes with an exactly known input, used for unit and
injection-recovery tests before touching any real or simulated survey data.

Position-angle convention
--------------------------
KinMS's ``posAng`` places its disc's redshifted lobe along **+y** at
``posAng=0`` and rotates it towards **-x** as ``posAng`` increases (verified
empirically — KinMS ships no explicit statement of rotation sense). KINO's
``pa`` is degrees **East of North** in the ``(x=East, y=North)`` plane, i.e.
it rotates from **+y towards +x** as ``pa`` increases. The two rotate in
opposite senses, so this wrapper passes ``posAng=-pa`` to KinMS to align the
kinematic major-**axis line** with KINO's convention.

The major-axis *line* is pinned by
``tests/io/test_kinms.py::test_position_angle_convention``. Which *end* of
that line is redshifted is confirmed consistent with ``kino.galaxy``'s
independently-derived ``Vlos`` convention (Phase 1) by
``tests/galaxy/test_disc.py::test_pa_zero_redshifted_side_matches_kinms_convention``
and, at the level of a full map rather than two points, by
``tests/galaxy/test_kinms_consistency.py`` (Vlos-map correlation > 0.99
against an independently generated KinMS cube).
"""

from __future__ import annotations

from collections.abc import Sequence

import astropy.units as u
import numpy as np
from kinms import KinMS

from kino.io.base import Datacube


def generate_disc_cube(
    *,
    field_of_view: u.Quantity,
    pixel_scale: u.Quantity,
    velocity_range: u.Quantity,
    channel_width: u.Quantity,
    beam_size: u.Quantity,
    inc: float,
    pa: float,
    sb_rad: u.Quantity,
    sb_prof: Sequence[float] | np.ndarray,
    vel_rad: u.Quantity,
    vel_prof: u.Quantity,
    gas_sigma: u.Quantity = 0 * u.km / u.s,
    v_sys: u.Quantity = 0 * u.km / u.s,
    redshift: float = 0.0,
    n_clouds: int = 500_000,
    rng: np.random.Generator | None = None,
) -> Datacube:
    """Build a KinMS rotating-disc cube in KINO's ``Datacube`` convention.

    Parameters
    ----------
    field_of_view : Quantity (angle)
        Square spatial extent of the output cube.
    pixel_scale : Quantity (angle)
        Spaxel size.
    velocity_range : Quantity (speed)
        Extent of the spectral (velocity) axis.
    channel_width : Quantity (speed)
        Channel width of the spectral axis.
    beam_size : Quantity (angle), scalar
        Circular PSF FWHM to convolve with. Only a circular beam is
        supported by this thin wrapper; pass an elliptical/angled beam
        directly to ``kinms.KinMS`` if needed.
    inc : float
        Inclination, degrees; 0 = face-on, 90 = edge-on (``CLAUDE.md``).
    pa : float
        Position angle, degrees East of North (``CLAUDE.md``); see the
        module docstring for how this maps onto KinMS's ``posAng``.
    sb_rad : Quantity (angle)
        Radius vector for the surface-brightness profile.
    sb_prof : array-like
        Surface-brightness profile (arbitrary scale) vs. ``sb_rad``.
    vel_rad : Quantity (angle)
        Radius vector for the circular-velocity profile.
    vel_prof : Quantity (speed)
        Circular velocity profile vs. ``vel_rad``.
    gas_sigma : Quantity (speed), optional
        Intrinsic (constant) velocity dispersion.
    v_sys : Quantity (speed), optional
        Systemic velocity, added to the spectral axis's zero point.
    redshift : float, optional
        Cosmological redshift recorded on the returned ``Datacube`` as
        metadata only — KinMS itself models in the local/rest frame.
    n_clouds : int, optional
        Number of point-cloud samples (KinMS ``nSamps``); higher reduces
        Monte Carlo noise in the cube at the cost of runtime.
    rng : numpy.random.Generator or None, optional
        Seeded RNG (``CLAUDE.md``: no global RNG use) used to draw a scalar
        seed for KinMS's ``seed`` argument (KinMS internally scales a single
        int into its 4 per-stream seeds -- do not pass a 4-element seed here,
        it silently overflows to an invalid seed for each stream).
        ``None`` uses KinMS's own default seed.

    Returns
    -------
    Datacube
        ``flux`` in arbitrary units (KinMS's default ``Jy/beam``-scaled
        output; no absolute flux calibration is applied). ``spectral_axis``
        is velocity (km/s), linearly spaced and centred on ``v_sys``.
    """
    seed = None
    if rng is not None:
        seed = int(rng.integers(1, 1_000_000))

    model = KinMS(
        xs=field_of_view.to_value(u.arcsec),
        ys=field_of_view.to_value(u.arcsec),
        vs=velocity_range.to_value(u.km / u.s),
        cellSize=pixel_scale.to_value(u.arcsec),
        dv=channel_width.to_value(u.km / u.s),
        beamSize=beam_size.to_value(u.arcsec),
        nSamps=n_clouds,
        seed=seed,
        fixSeed=seed is not None,
        verbose=False,
    )
    cube_xyv = model.model_cube(
        inc=inc,
        posAng=-pa,  # KinMS posAng rotates opposite to KINO's pa -- see module docstring
        gasSigma=gas_sigma.to_value(u.km / u.s),
        sbProf=np.asarray(sb_prof, dtype=float),
        sbRad=sb_rad.to_value(u.arcsec),
        velRad=vel_rad.to_value(u.arcsec),
        velProf=vel_prof.to_value(u.km / u.s),
        vSys=v_sys.to_value(u.km / u.s),
    )
    # KinMS returns (n_x, n_y, n_v); Datacube wants (n_spec, n_y, n_x).
    flux = np.transpose(cube_xyv, (2, 1, 0))

    n_chan = flux.shape[0]
    dv = channel_width.to_value(u.km / u.s)
    spectral_axis = ((np.arange(n_chan) - (n_chan - 1) / 2) * dv + v_sys.to_value(u.km / u.s)) * (
        u.km / u.s
    )

    return Datacube(
        flux=flux,
        spectral_axis=spectral_axis,
        spectral_kind="velocity",
        pixel_scale=pixel_scale.to(u.arcsec),
        redshift=redshift,
        flux_unit=u.dimensionless_unscaled,
        meta={
            "source": "KinMS",
            "citation": "Davis et al. 2013, MNRAS 429, 534",
            "generator_params": {
                "inc_deg": inc,
                "pa_deg": pa,
                "v_sys_kms": v_sys.to_value(u.km / u.s),
                "n_clouds": n_clouds,
            },
        },
    )
