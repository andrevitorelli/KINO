"""Recover the (g1, g2) shear posterior for one mock galaxy observation.

Loads a mock data product written by simulate_one_galaxy.py and runs nested
sampling (kino.inference.fit_shear, backed by nautilus) to recover the
posterior on (g1, g2). Every other galaxy parameter is held fixed at the
value simulate_one_galaxy.py used to generate the mock (read back from the
FITS header) -- this script does not (yet) expose freeing inc/v_flat/etc.
as a CLI option; see docs/manual/inference.md for that by hand.

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

from kino.galaxy import DiscGalaxy
from kino.inference.likelihood import IFUCubeLikelihood, SlitCubeLikelihood
from kino.inference.priors import build_prior
from kino.inference.sampler import fit_shear
from kino.instruments.ifu import IFU
from kino.instruments.slit import SlitSpectrograph
from kino.io.base import Datacube


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


def _datacube_from_hdu(hdu: fits.ImageHDU, variance: np.ndarray | None = None) -> Datacube:
    """Inverse of simulate_one_galaxy.py's ``_datacube_hdu``."""
    h = hdu.header
    n_spec = hdu.data.shape[0]
    spec_val0 = h["SPECVAL0"]
    spec_step = h.get("SPECSTEP", 0.0)
    spectral_axis = (spec_val0 + spec_step * np.arange(n_spec)) * u.Unit(h["SPECUNIT"])
    return Datacube(
        flux=np.asarray(hdu.data, dtype=np.float64),
        spectral_axis=spectral_axis,
        spectral_kind=h["SPECKIND"],
        pixel_scale=h["PIXSCALE"] * u.arcsec,
        redshift=h["REDSHIFT"],
        flux_unit=u.electron,
        variance=variance,
    )


def main() -> None:
    args = parse_args()

    with fits.open(args.data) as hdul:
        primary = hdul["PRIMARY"].header
        kind = primary["KIND"]
        gal = DiscGalaxy(
            r_e=primary["R_E"], inc=primary["INC"], pa=primary["PA"],
            v_flat=primary["V_FLAT"], z=primary["GAL_Z"], sigma0=primary["SIGMA0"],
        )  # fmt: skip
        line_flux = primary["LINEFLUX"] * u.erg / u.s / u.cm**2
        line_wavelength = primary["LINEWAVE"] * u.um

        # g1/g2 are the only free parameters this CLI exposes, so every other
        # DiscGalaxy field -- including inc/v_flat -- must be fixed here.
        fixed_galaxy_params = {
            "r_e": gal.r_e, "inc": gal.inc, "pa": gal.pa, "v_flat": gal.v_flat,
            "z": gal.z, "sigma0": gal.sigma0, "flux": gal.flux,
        }  # fmt: skip
        variance = None
        if "VARIANCE" in hdul:
            variance = np.asarray(hdul["VARIANCE"].data, dtype=np.float64)
        if kind == "ifu":
            data = _datacube_from_hdu(hdul["FLUX"], variance)
            instrument = IFU.from_config(str(args.config))
            likelihood = IFUCubeLikelihood(
                ifu=instrument, line_wavelength=line_wavelength, data=data,
                fixed_galaxy_params=fixed_galaxy_params, fixed_line_flux=line_flux,
            )  # fmt: skip
        else:
            data = _datacube_from_hdu(hdul["SPECTRUM"], variance)
            instrument = SlitSpectrograph.from_config(str(args.config))
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
    for name, truth_key in [("g1", "G1"), ("g2", "G2")]:
        med, lo, hi = summary[name]
        print(f"{name} = {med:.4f} (+{hi:.4f}/-{lo:.4f})  (truth: {primary[truth_key]:.4f})")
    np.savez(args.out, log_z=posterior.log_z, **posterior.samples)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
