#!/usr/bin/env python
"""Fast check of the Fisher shear forecast against maximum-likelihood fits.

For a fixed galaxy and shear, draws many noise realisations of the mock IFU
cube, fits ``(g1, g2)`` (everything else known) by Nelder-Mead on the very
likelihood the inference uses (:class:`~kino.inference.likelihood.IFUMomentLikelihood`),
and compares the scatter and bias of the estimates with
:func:`kino.analysis.fisher.shear_fisher`'s ``sigma_fixed``. About 10 s per
realisation, so 24-30 realisations in a few minutes -- a stand-in for nested
sampling when only the estimator's scatter and bias are wanted.

    python scripts/validate_fisher_mle.py sifs --flux 1e-13 --n 24 --threshold 3
    python scripts/validate_fisher_mle.py gmtifs-nb04 --n 30 --threshold 3

Findings (recorded in ``notebooks/data/fisher_validation.json``): the scatter
matches the forecast at high per-spaxel S/N and exceeds it, with a bias, at low
S/N.
"""

from __future__ import annotations

import argparse
import dataclasses
import time

import astropy.units as u
import numpy as np
from scipy.optimize import minimize

from kino.analysis.fisher import shear_fisher
from kino.analysis.forecast import narrowed_ifu
from kino.galaxy import DiscGalaxy
from kino.inference.likelihood import GALAXY_FIELDS, IFUMomentLikelihood
from kino.instruments.base import PSF
from kino.instruments.ifu import IFU
from kino.lensing import apply_shear


def setups(name: str, flux: float):
    if name == "sifs":
        from kino.observing import SkyConditions, SkyLibrary, ifu_sky_electron_rate

        z = 0.06
        gal = DiscGalaxy(r_e=3.0, inc=55.0, pa=30.0, v_flat=180.0, z=z, sigma0=30.0)
        line = 6564.61 * u.AA
        ifu = narrowed_ifu(IFU.from_config("kino/config/sifs.yaml"), line * (1 + z), 500.0)
        lib = SkyLibrary.load("data/skycalc")
        sky = ifu_sky_electron_rate(ifu, SkyConditions(1.1, 110.0), lib)
        ifu = dataclasses.replace(ifu, psf=PSF(seeing_fwhm=1.0 * u.arcsec), sky_electron_rate=sky)
        return ifu, gal, line, flux * u.erg / u.s / u.cm**2, (0.05, 0.0)
    ifu0 = IFU.from_config("kino/config/gmtifs.yaml")
    centre, dlam = 1.1025, 1.1025 * 400 / 3e5
    ifu = dataclasses.replace(
        ifu0, nx=24, ny=24, n_wave=25,
        wavelength_min=(centre - dlam) * u.um, wavelength_max=(centre + dlam) * u.um,
    )  # fmt: skip
    gal = DiscGalaxy(r_e=0.3, inc=55.0, pa=30.0, v_flat=180.0, z=0.68, sigma0=20.0)
    return ifu, gal, 0.65628 * u.um, flux * u.erg / u.s / u.cm**2, (0.035, -0.02)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("setup", choices=["sifs", "gmtifs-nb04"])
    ap.add_argument("--flux", type=float, default=1e-13)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--threshold", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ifu, gal, line, flux, g_true = setups(a.setup, a.flux)
    fixed = {f: getattr(gal, f) for f in GALAXY_FIELDS}
    fis = shear_fisher(
        ifu, gal, line_flux=flux, line_wavelength=line, g=g_true, snr_threshold=a.threshold
    )
    rng = np.random.default_rng(a.seed)
    est = []
    t0 = time.time()
    for _ in range(a.n):
        data = ifu.observe(apply_shear(gal, *g_true), line_flux=flux, line_wavelength=line, rng=rng)
        lik = IFUMomentLikelihood(
            ifu=ifu, line_wavelength=line, data=data, fixed_galaxy_params=fixed,
            fixed_line_flux=flux, snr_threshold=a.threshold,
        )  # fmt: skip
        res = minimize(
            lambda p, lik=lik: -lik({"g1": p[0], "g2": p[1]}),
            x0=[0.0, 0.0], method="Nelder-Mead",
            options=dict(xatol=1e-4, fatol=1e-3, maxiter=300),
        )  # fmt: skip
        est.append(res.x)
    resid = np.array(est) - np.array(g_true)
    scatter = resid.std(0, ddof=1)
    print(f"{a.setup} flux={a.flux:.0e} thr={a.threshold}: n={a.n} in {time.time() - t0:.0f}s")
    print(f"  median spaxel S/N (forecast) {fis.median_spaxel_snr:.1f}, spaxels {fis.n_spaxels}")
    print(f"  bias    {resid.mean(0).round(4)} +/- {(scatter / np.sqrt(a.n)).round(4)}")
    print(f"  scatter {scatter.round(4)}  Fisher sigma_fixed {np.round(fis.sigma_fixed, 4)}")
    print(f"  scatter / Fisher {np.round(scatter / np.array(fis.sigma_fixed), 2)}")
