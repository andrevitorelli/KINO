#!/usr/bin/env python
"""Regenerate the illustrative velocity-map gallery used at the top of README.md.

Renders one sheared, synthetic DiscGalaxy through each real/fiducial IFU
config's own observe() pipeline (its own pixel scale, field of view,
wavelength window and PSF), extracts Vlos via kino.analysis.moments, and
saves a minimal cover image (map + axes + an instrument-name title, no
colorbar) to assets/velocity_maps/. Purely illustrative -- the galaxy
geometry, flux and shear are chosen to look good, not to represent any real
target or a calibrated forecast; see notebooks 08-15 for real forecasts.

    python scripts/make_readme_velocity_maps.py
"""

from __future__ import annotations

import dataclasses

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from kino.analysis.moments import compute_moment_maps
from kino.galaxy import DiscGalaxy
from kino.instruments.ifu import IFU
from kino.lensing import apply_shear

HALPHA = 6564.61 * u.AA  # vacuum rest
HBETA = 4862.68 * u.AA  # vacuum rest

#: (config file, display name, rest line, flux [erg/s/cm2], r_e [arcsec], exposure override [s])
TARGETS = [
    ("sifs", "SOAR / SIFS", HALPHA, 3e-13, 1.5, None),
    ("gmtifs", "GMT / GMTIFS", HALPHA, 1e-13, 0.22, None),
    ("keck_kcwi", "Keck / KCWI", HBETA, 3e-13, 1.3, None),
    ("gemini_gmos_ifu", "Gemini South / GMOS-IFU", HALPHA, 3e-13, 1.0, None),
    ("vlt_muse", "VLT / MUSE", HALPHA, 3e-13, 1.4, None),
    ("gemini_nifs", "Gemini North / NIFS", HALPHA, 3e-12, 0.35, 1800.0),
    ("keck_osiris", "Keck / OSIRIS", HALPHA, 3e-12, 0.3, 1800.0),
]

G1, G2 = 0.06, -0.03


def _crop(mm, ifu, px):
    """(xlim, ylim) tightly bounding the detected disc, padded, clipped to the real FOV."""
    good = np.isfinite(mm.vlos)
    ys, xs = np.where(good)
    x_arc = (xs - ifu.nx / 2.0 + 0.5) * px
    y_arc = (ys - ifu.ny / 2.0 + 0.5) * px
    pad = 0.18 * max(x_arc.max() - x_arc.min(), y_arc.max() - y_arc.min(), 1e-3)
    fov_x, fov_y = ifu.nx * px, ifu.ny * px
    xlim = (max(x_arc.min() - pad, -fov_x / 2), min(x_arc.max() + pad, fov_x / 2))
    ylim = (max(y_arc.min() - pad, -fov_y / 2), min(y_arc.max() + pad, fov_y / 2))
    return xlim, ylim, int(good.sum())


def main() -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})
    for config, name, line, flux, r_e, exptime in TARGETS:
        ifu = IFU.from_config(f"kino/config/{config}.yaml")
        if exptime is not None:
            ifu = dataclasses.replace(ifu, exposure_time=exptime * u.s)
        z = (0.5 * (ifu.wavelength_min + ifu.wavelength_max) / line).decompose().value - 1.0
        gal = DiscGalaxy(r_e=r_e, inc=60.0, pa=25.0, v_flat=200.0, z=z, sigma0=25.0)
        scene = apply_shear(gal, G1, G2)
        cube = ifu.observe(
            scene, line_flux=flux * u.erg / u.s / u.cm**2, line_wavelength=line, rng=None
        )
        mm = compute_moment_maps(cube, line_wavelength=line, snr_threshold=3.0, check_window=False)

        px = ifu.pixel_scale.to_value(u.arcsec)
        fov_x, fov_y = ifu.nx * px, ifu.ny * px
        extent = [-fov_x / 2, fov_x / 2, -fov_y / 2, fov_y / 2]
        vmax = max(np.nanmax(np.abs(mm.vlos - gal.v_sys)), 10.0)
        xlim, ylim, n_good = _crop(mm, ifu, px)

        fig, ax = plt.subplots(figsize=(5.0, 4.2))
        ax.imshow(
            mm.vlos, origin="lower", extent=extent, cmap="RdBu_r",
            vmin=-vmax, vmax=vmax, aspect="equal",
        )  # fmt: skip
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("arcsec")
        ax.set_ylabel("arcsec")
        ax.set_title(name, fontsize=13, fontweight="bold")
        fig.tight_layout()
        out = f"assets/velocity_maps/{config}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f'{name:28s} z={z:.3f} fov={fov_x:.2f}x{fov_y:.2f}" good={n_good}/{mm.vlos.size}')
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
