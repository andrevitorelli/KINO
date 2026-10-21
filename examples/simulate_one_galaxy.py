"""Simulate one sheared disc galaxy through an instrument backend.

Given an intrinsic galaxy, a reduced shear (g1, g2), and an instrument
config, produce that instrument's mock data product and write it to FITS
via :mod:`kino.mocks`/:mod:`kino.io.fitsio` (README.md Phase 4):

- IFU backend         -> a FLUX (+ VARIANCE) datacube extension
- slit/grism backend  -> a SPECTRUM (+ VARIANCE) datacube extension, plus
                         the companion broadband IMAGE extension

Every galaxy/shear/line parameter is recorded in the written file's own
``Datacube.meta`` (via :func:`kino.mocks.single.simulate_observation`'s
provenance), including ``kind``/``config`` -- ``run_inference.py`` reads
it all back from there rather than a bespoke header.

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

from kino.galaxy import DiscGalaxy
from kino.instruments.ifu import IFU
from kino.instruments.slit import SlitObservation, SlitSpectrograph
from kino.io import fitsio
from kino.mocks.single import simulate_observation


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


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    gal = DiscGalaxy(
        r_e=args.r_e, inc=args.inc, pa=args.pa, v_flat=args.v_flat,
        z=args.z, sigma0=args.sigma0,
    )  # fmt: skip
    line_flux = args.line_flux * u.erg / u.s / u.cm**2
    line_wavelength = args.line_wavelength * u.um

    if args.kind == "ifu":
        instrument = IFU.from_config(str(args.config))
    else:
        instrument = SlitSpectrograph.from_config(str(args.config))

    result = simulate_observation(
        gal, args.g1, args.g2, instrument,
        line_flux=line_flux, line_wavelength=line_wavelength, rng=rng,
        meta={"kind": args.kind, "config": str(args.config)},
    )  # fmt: skip

    if isinstance(result, SlitObservation):
        fitsio.write_slit_observation(result, args.out)
    else:
        fitsio.write_datacube(result, args.out)
    print(f"wrote {args.out} ({args.kind}, g=({args.g1}, {args.g2}))")


if __name__ == "__main__":
    main()
