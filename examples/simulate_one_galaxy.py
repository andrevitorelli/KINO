"""Simulate one sheared disc galaxy through an instrument backend.

Given an intrinsic galaxy, a reduced shear (g1, g2), and an instrument
config, produce that instrument's mock data product and write it to FITS:

- IFU backend         -> a FLUX (+ VARIANCE) datacube extension
- slit/grism backend  -> a SPECTRUM (+ VARIANCE) datacube extension, plus
                         the companion broadband IMAGE extension

Usage:
    python examples/simulate_one_galaxy.py --config kino/config/sifs.yaml \\
        --line-flux 1e-15 --line-wavelength 0.65628 --g1 0.03 --g2 -0.01 \\
        --out mock.fits

    python examples/simulate_one_galaxy.py --config kino/config/goodman.yaml \\
        --kind slit --line-flux 1e-15 --line-wavelength 0.65628 \\
        --out mock_slit.fits

See docs/quickstart.md and docs/manual/forecasting.md for the same forward
model called directly from Python rather than this CLI wrapper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits

from kino.galaxy import DiscGalaxy
from kino.instruments.ifu import IFU
from kino.instruments.slit import SlitSpectrograph
from kino.io.base import Datacube
from kino.lensing import apply_shear


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Instrument YAML config (see kino/config/)."
    )
    parser.add_argument(
        "--kind",
        choices=["ifu", "slit"],
        default="ifu",
        help="Instrument backend the config is for (default: ifu).",
    )
    parser.add_argument("--r_e", type=float, default=0.5, help="Sersic effective radius, arcsec.")
    parser.add_argument("--inc", type=float, default=60.0, help="Inclination, deg (0=face-on).")
    parser.add_argument("--pa", type=float, default=30.0, help="Position angle, deg East of North.")
    parser.add_argument("--v_flat", type=float, default=180.0, help="Asymptotic v_rot, km/s.")
    parser.add_argument("--z", type=float, default=0.1, help="Redshift.")
    parser.add_argument("--sigma0", type=float, default=30.0, help="Intrinsic dispersion, km/s.")
    parser.add_argument("--g1", type=float, default=0.0, help="Reduced shear component g1.")
    parser.add_argument("--g2", type=float, default=0.0, help="Reduced shear component g2.")
    parser.add_argument(
        "--line-flux",
        type=float,
        required=True,
        help="Intrinsic (unlensed) line flux, erg/s/cm^2.",
    )
    parser.add_argument(
        "--line-wavelength", type=float, required=True, help="Rest-frame line wavelength, micron."
    )
    parser.add_argument(
        "--out", type=Path, default=Path("mock.fits"), help="Output mock data path."
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the numpy.random.Generator.")
    return parser.parse_args()


def _datacube_hdu(cube: Datacube, name: str) -> fits.ImageHDU:
    """A Datacube's flux array as a FITS extension, with enough header
    metadata (``run_inference.py`` and :func:`kino.io` helpers) to
    reconstruct units/axes on read -- not a general-purpose FITS cube
    standard, just enough round-trip for KINO's own tools."""
    hdu = fits.ImageHDU(data=np.asarray(cube.flux, dtype=np.float64), name=name)
    hdu.header["BUNIT"] = str(cube.flux_unit)
    hdu.header["SPECKIND"] = cube.spectral_kind
    hdu.header["SPECUNIT"] = str(cube.spectral_axis.unit)
    hdu.header["SPECVAL0"] = cube.spectral_axis[0].value
    if len(cube.spectral_axis) > 1:
        hdu.header["SPECSTEP"] = (cube.spectral_axis[1] - cube.spectral_axis[0]).value
    hdu.header["PIXSCALE"] = cube.pixel_scale.to_value(u.arcsec)
    hdu.header["REDSHIFT"] = cube.redshift
    return hdu


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    gal = DiscGalaxy(
        r_e=args.r_e, inc=args.inc, pa=args.pa, v_flat=args.v_flat,
        z=args.z, sigma0=args.sigma0,
    )  # fmt: skip
    scene = apply_shear(gal, args.g1, args.g2)
    line_flux = args.line_flux * u.erg / u.s / u.cm**2
    line_wavelength = args.line_wavelength * u.um

    primary = fits.PrimaryHDU()
    primary.header["KIND"] = args.kind
    primary.header["CONFIG"] = str(args.config)
    primary.header["G1"] = args.g1
    primary.header["G2"] = args.g2
    primary.header["R_E"] = args.r_e
    primary.header["INC"] = args.inc
    primary.header["PA"] = args.pa
    primary.header["V_FLAT"] = args.v_flat
    primary.header["GAL_Z"] = args.z
    primary.header["SIGMA0"] = args.sigma0
    primary.header["LINEFLUX"] = args.line_flux
    primary.header["LINEWAVE"] = args.line_wavelength
    hdus: list[fits.hdu.base._BaseHDU] = [primary]

    if args.kind == "ifu":
        instrument = IFU.from_config(str(args.config))
        cube = instrument.observe(
            scene, line_flux=line_flux, line_wavelength=line_wavelength, rng=rng
        )
        hdus.append(_datacube_hdu(cube, "FLUX"))
        if cube.variance is not None:
            hdus.append(fits.ImageHDU(data=cube.variance, name="VARIANCE"))
    else:
        instrument = SlitSpectrograph.from_config(str(args.config))
        obs = instrument.observe(
            scene, line_flux=line_flux, line_wavelength=line_wavelength, rng=rng
        )
        hdus.append(_datacube_hdu(obs.spectrum, "SPECTRUM"))
        if obs.spectrum.variance is not None:
            hdus.append(fits.ImageHDU(data=obs.spectrum.variance, name="VARIANCE"))
        image_hdu = fits.ImageHDU(data=obs.image, name="IMAGE")
        image_hdu.header["PIXSCALE"] = obs.image_pixel_scale.to_value(u.arcsec)
        hdus.append(image_hdu)

    fits.HDUList(hdus).writeto(args.out, overwrite=True)
    print(f"wrote {args.out} ({args.kind}, g=({args.g1}, {args.g2}))")


if __name__ == "__main__":
    main()
