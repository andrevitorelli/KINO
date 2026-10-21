"""Recover the (g1, g2) shear posterior for one mock galaxy observation.

Loads a mock data product written by simulate_one_galaxy.py and runs nested
sampling (kino.inference.fit_shear, backed by nautilus) to recover the
posterior on (g1, g2). Every other galaxy parameter is held fixed at the
value simulate_one_galaxy.py used to generate the mock -- read back from
the file's own ``Datacube.meta`` provenance (``kino.mocks.single.
simulate_observation``, README.md Phase 4), via ``kino.io.fitsio``, not a
bespoke FITS header. This script does not (yet) expose freeing
inc/v_flat/etc. as a CLI option; see docs/manual/inference.md for that by
hand.

Usage:
    python examples/simulate_one_galaxy.py --config kino/config/sifs.yaml \\
        --line-flux 1e-15 --line-wavelength 0.65628 --g1 0.03 --g2 -0.01 \\
        --out mock.fits
    python examples/run_inference.py --data mock.fits --config kino/config/sifs.yaml

**This is slow.** Each likelihood evaluation renders a full PSF-convolved
mock cube (roughly half a second on a laptop); nested sampling needs
thousands of evaluations even at this script's modest default n_live/n_eff.
Budget several minutes, not seconds, and raise --n-live/--n-eff for a
production-quality posterior rather than a quick check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits

from kino.inference.likelihood import GALAXY_FIELDS, IFUCubeLikelihood, SlitCubeLikelihood
from kino.inference.priors import build_prior
from kino.inference.sampler import fit_shear
from kino.instruments.ifu import IFU
from kino.instruments.slit import SlitSpectrograph
from kino.io import fitsio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data", type=Path, required=True, help="Mock data written by simulate_one_galaxy.py."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="The instrument config used to simulate it."
    )
    parser.add_argument(
        "--g-bound", type=float, default=0.3, help="Flat prior half-width on g1/g2 (default 0.3)."
    )
    parser.add_argument(
        "--n-live", type=int, default=200, help="nautilus live points (default 200)."
    )
    parser.add_argument(
        "--n-eff", type=float, default=500, help="nautilus effective sample size (default 500)."
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampler seed.")
    parser.add_argument(
        "--out", type=Path, default=Path("posterior.npz"), help="Output posterior samples path."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with fits.open(args.data) as hdul:
        kind = "ifu" if "FLUX" in hdul else "slit"

    if kind == "ifu":
        data = fitsio.read_datacube(args.data, extname="FLUX")
        instrument = IFU.from_config(str(args.config))
    else:
        data = fitsio.read_slit_observation(args.data).spectrum
        instrument = SlitSpectrograph.from_config(str(args.config))

    # Every galaxy/shear/line parameter was recorded in the mock's own
    # provenance by kino.mocks.single.simulate_observation; read it back
    # rather than assuming a bespoke header (README.md Phase 4).
    meta = data.meta
    line_flux = u.Quantity(meta["line_flux"])
    line_wavelength = u.Quantity(meta["line_wavelength"])
    truth_g1, truth_g2 = meta["g1"], meta["g2"]

    # g1/g2 are the only free parameters this CLI exposes, so every other
    # DiscGalaxy field -- including inc/v_flat -- must be fixed here.
    fixed_galaxy_params = {k: meta[k] for k in GALAXY_FIELDS if k in meta}

    if kind == "ifu":
        likelihood = IFUCubeLikelihood(
            ifu=instrument, line_wavelength=line_wavelength, data=data,
            fixed_galaxy_params=fixed_galaxy_params, fixed_line_flux=line_flux,
        )  # fmt: skip
    else:
        likelihood = SlitCubeLikelihood(
            slit=instrument, line_wavelength=line_wavelength, data=data,
            fixed_galaxy_params=fixed_galaxy_params, fixed_line_flux=line_flux,
        )  # fmt: skip

    g_bound = (-args.g_bound, args.g_bound)
    prior = build_prior({"g1": g_bound, "g2": g_bound})
    print(
        f"sampling {kind} likelihood, n_live={args.n_live}, "
        f"n_eff={args.n_eff} (this will take a while)..."
    )
    posterior = fit_shear(prior, likelihood, n_live=args.n_live, n_eff=args.n_eff, seed=args.seed)

    summary = posterior.summary()
    for name, truth in [("g1", truth_g1), ("g2", truth_g2)]:
        med, lo, hi = summary[name]
        print(f"{name} = {med:.4f} (+{hi:.4f}/-{lo:.4f})  (truth: {truth:.4f})")
    np.savez(args.out, log_z=posterior.log_z, **posterior.samples)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
