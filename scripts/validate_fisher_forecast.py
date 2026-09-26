#!/usr/bin/env python
"""Validate the Fisher shear forecast against a real injection-recovery ensemble.

Runs ``n`` noisy injection-recovery trials (cube likelihood by default; the Gaussian cube model
needs nonzero read noise, so use ``--read-noise 3 --sky 0.5``) (``kino.analysis.injection``) of one
fixed galaxy at one fixed shear on a small test IFU, fitting only ``(g1, g2)``
(all else known), and writes the recovered shears and reported posterior widths
to JSON. Compare the scatter of ``g_rec - g_true`` with
``kino.analysis.fisher.shear_fisher(...).sigma_fixed`` for the same setup:
if the forecast and the likelihood's errors are calibrated they agree, and the
pull ``(g_rec - g_true) / sigma_post`` has unit standard deviation.

Slow (about 2 minutes per trial): run several seeds in parallel processes.

    python scripts/validate_fisher_forecast.py --window narrow --seed 301 --n 6 --out /tmp/v301.json

``--window`` sets the spectral window of the test IFU: ``narrow`` is +/-273 km/s
(the line fills over half of it), ``wide`` +/-545 km/s. With the *known*
baseline subtracted (the default of ``IFUMomentLikelihood``) both should agree
with the forecast; with a per-spaxel median background the narrow window's
errors were 4.6x too small (see ``CLAUDE.md``, moment maps).
"""

from __future__ import annotations

import argparse
import json
import time

import astropy.units as u

from kino.analysis.injection import run_injection
from kino.galaxy import DiscGalaxy
from kino.instruments.base import LSF, PSF, Detector, Throughput
from kino.instruments.ifu import IFU

WINDOWS = {"narrow": (1.099, 1.101, 20), "wide": (1.098, 1.102, 40)}
GALAXY = dict(r_e=0.2, inc=55.0, pa=30.0, v_flat=180.0, r_t=0.15, z=0.0, sigma0=25.0, flux=1.0)
SHEAR = (0.03, -0.02)
LINE_FLUX = 5e-15 * u.erg / u.s / u.cm**2
LINE_WAVELENGTH = 1.1 * u.um


def make_ifu(window: str, read_noise: float = 0.0, sky: float = 0.0) -> IFU:
    wmin, wmax, n_wave = WINDOWS[window]
    return IFU(
        pixel_scale=0.1 * u.arcsec, nx=8, ny=8,
        wavelength_min=wmin * u.um, wavelength_max=wmax * u.um, n_wave=n_wave,
        psf=PSF(seeing_fwhm=0.02 * u.arcsec), lsf=LSF(resolving_power=8000.0),
        throughput=Throughput(wavelength=[1.0, 1.2] * u.um, transmission=[1.0, 1.0]),
        aperture_diameter=10 * u.m,
        detector=Detector(read_noise=read_noise * u.electron, dark_current=0 * u.electron / u.s),
        exposure_time=600 * u.s, sky_electron_rate=sky * u.electron / u.s,
    )  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--window", choices=list(WINDOWS), default="wide")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-live", type=int, default=100)
    ap.add_argument("--n-eff", type=int, default=200)
    ap.add_argument("--read-noise", type=float, default=0.0, help="electrons")
    ap.add_argument("--sky", type=float, default=0.0, help="electron/s per spaxel per channel")
    ap.add_argument("--likelihood", choices=["cube", "moments"], default="cube")
    a = ap.parse_args()

    t0 = time.time()
    galaxy = DiscGalaxy(**GALAXY)
    trials = run_injection(
        [galaxy] * a.n, [SHEAR], seed=a.seed, ifu=make_ifu(a.window, a.read_noise, a.sky),
        line_wavelength=LINE_WAVELENGTH, line_flux=LINE_FLUX, n_live=a.n_live, n_eff=a.n_eff,
        likelihood=a.likelihood,
    )  # fmt: skip
    out = [dict(g_true=t.g_true, g_rec=t.g_rec, g_err=t.g_err) for t in trials]
    with open(a.out, "w") as f:
        json.dump(
            {"window": a.window, "seed": a.seed, "likelihood": a.likelihood,
             "read_noise": a.read_noise, "sky": a.sky, "trials": out}, f
        )  # fmt: skip
    print(f"done seed={a.seed} n={a.n} in {time.time() - t0:.0f}s")
