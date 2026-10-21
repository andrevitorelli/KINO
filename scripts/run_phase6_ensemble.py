#!/usr/bin/env python
"""Phase 6 ensemble validation: bias vs S/N/size/inclination/shear, shape
noise, (m, c), null tests, a published-forecast benchmark, and an IFU vs.
slit comparison -- README.md Phase 6's remaining ensemble-suite bullets.

Uses the same fast maximum-likelihood-on-the-real-likelihood approach
`scripts/validate_fisher_mle.py` already validated as a stand-in for nested
sampling (Nelder-Mead on `IFUCubeLikelihood`/`SlitCubeLikelihood`, ~0.2-1.6s
per evaluation, ~15s per fit) -- full nested sampling at this ensemble size
(hundreds of fits) would take far too long. `kino.analysis.injection`
remains the nested-sampling, IFU-only machinery for a single run; this
script is the faster, broader ensemble characterization, on the same real
forward model and likelihoods, not a different estimator.

The fiducial IFU setup (SIFS, narrowed window, 1.0" seeing, read noise 3 e-,
sky 0.5 e-/s) matches `notebooks/data/fisher_validation.json`'s already
nested-sampling-validated ``nested_sampling_cube`` entry exactly, so these
MLE results are directly comparable to that validated reference point.

    python scripts/run_phase6_ensemble.py snr --n 15
    python scripts/run_phase6_ensemble.py size --n 15
    python scripts/run_phase6_ensemble.py inclination --n 15
    python scripts/run_phase6_ensemble.py shear --n 15
    python scripts/run_phase6_ensemble.py null --n 25
    python scripts/run_phase6_ensemble.py benchmark --n 30
    python scripts/run_phase6_ensemble.py instrument-compare --n 15

Each subcommand appends its results to ``notebooks/data/phase6_ensemble.json``
(read-modify-write, keyed by subcommand name) rather than overwriting it, so
sweeps can be run independently / resumed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import astropy.units as u
import numpy as np
from scipy.optimize import minimize

from kino.analysis.forecast import narrowed_ifu, narrowed_slit
from kino.galaxy import DiscGalaxy
from kino.inference.likelihood import GALAXY_FIELDS, IFUCubeLikelihood, SlitCubeLikelihood
from kino.instruments.base import PSF
from kino.instruments.ifu import IFU
from kino.instruments.slit import SlitSpectrograph
from kino.lensing import apply_shear

RESULTS_PATH = Path("notebooks/data/phase6_ensemble.json")

LINE = 6564.61 * u.AA  # Halpha, rest frame
Z = 0.06

# Fiducial IFU setup -- matches fisher_validation.json's validated
# "nested_sampling_cube" entry's own noise conditions exactly.
_READ_NOISE = 3.0 * u.electron
_SKY_IFU = 0.5 * u.electron / u.s


def fiducial_ifu() -> IFU:
    ifu0 = IFU.from_config("kino/config/sifs.yaml")
    ifu = narrowed_ifu(ifu0, LINE * (1 + Z), 500.0)
    return dataclasses.replace(
        ifu,
        psf=PSF(seeing_fwhm=1.0 * u.arcsec),
        detector=dataclasses.replace(ifu.detector, read_noise=_READ_NOISE),
        sky_electron_rate=_SKY_IFU,
    )


def fiducial_galaxy(**overrides) -> DiscGalaxy:
    kw = dict(r_e=3.0, inc=55.0, pa=30.0, v_flat=180.0, z=Z, sigma0=30.0)
    kw.update(overrides)
    return DiscGalaxy(**kw)


def mle_fit(lik, x0=(0.0, 0.0), maxiter=300) -> np.ndarray:
    res = minimize(
        lambda p: -lik({"g1": p[0], "g2": p[1]}),
        x0=list(x0),
        method="Nelder-Mead",
        options=dict(xatol=1e-4, fatol=1e-3, maxiter=maxiter),
    )
    return res.x


def run_ifu_ensemble(
    ifu: IFU, gal: DiscGalaxy, flux: u.Quantity, g_true, n: int, seed: int, line: u.Quantity = LINE
):
    fixed = {f: getattr(gal, f) for f in GALAXY_FIELDS}
    rng = np.random.default_rng(seed)
    est = []
    t0 = time.time()
    for _ in range(n):
        data = ifu.observe(apply_shear(gal, *g_true), line_flux=flux, line_wavelength=line, rng=rng)
        lik = IFUCubeLikelihood(
            ifu=ifu, line_wavelength=line, data=data,
            fixed_galaxy_params=fixed, fixed_line_flux=flux,
        )  # fmt: skip
        est.append(mle_fit(lik, x0=g_true))
    est = np.array(est)
    resid = est - np.array(g_true)
    elapsed = time.time() - t0
    return est, resid, elapsed


def run_slit_ensemble(
    slit: SlitSpectrograph,
    gal: DiscGalaxy,
    flux: u.Quantity,
    g_true,
    n: int,
    seed: int,
    line: u.Quantity = LINE,
):
    fixed = {f: getattr(gal, f) for f in GALAXY_FIELDS}
    rng = np.random.default_rng(seed)
    est = []
    t0 = time.time()
    for _ in range(n):
        obs = slit.observe(apply_shear(gal, *g_true), line_flux=flux, line_wavelength=line, rng=rng)
        lik = SlitCubeLikelihood(
            slit=slit, line_wavelength=line, data=obs.spectrum,
            fixed_galaxy_params=fixed, fixed_line_flux=flux,
        )  # fmt: skip
        est.append(mle_fit(lik, x0=g_true))
    est = np.array(est)
    resid = est - np.array(g_true)
    elapsed = time.time() - t0
    return est, resid, elapsed


def _summary(resid: np.ndarray, n: int) -> dict:
    scatter = resid.std(0, ddof=1) if n > 1 else [float("nan")] * 2
    bias = resid.mean(0)
    bias_err = scatter / np.sqrt(n) if n > 1 else [float("nan")] * 2
    return {
        "n": n,
        "bias": [float(b) for b in bias],
        "bias_err": [float(e) for e in bias_err],
        "scatter": [float(s) for s in scatter],
    }


def _load_results() -> dict:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def _save_results(results: dict) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=1)


def cmd_snr(args):
    ifu = fiducial_ifu()
    gal = fiducial_galaxy()
    g_true = (0.05, 0.0)
    fluxes = [3e-14, 1e-13, 3e-13, 1e-12]
    out = []
    for f in fluxes:
        flux = f * u.erg / u.s / u.cm**2
        est, resid, elapsed = run_ifu_ensemble(ifu, gal, flux, g_true, args.n, args.seed)
        row = {"flux": f, **_summary(resid, args.n), "elapsed_s": elapsed}
        out.append(row)
        print(
            f"flux={f:.0e}: n={args.n} in {elapsed:.0f}s  "
            f"bias={row['bias']}  scatter={row['scatter']}"
        )
    results = _load_results()
    results["snr"] = {"g_true": list(g_true), "rows": out}
    _save_results(results)


def cmd_size(args):
    ifu = fiducial_ifu()
    flux = 1e-13 * u.erg / u.s / u.cm**2
    g_true = (0.05, 0.0)
    sizes = [1.0, 2.0, 3.0, 5.0]
    out = []
    for r_e in sizes:
        gal = fiducial_galaxy(r_e=r_e)
        est, resid, elapsed = run_ifu_ensemble(ifu, gal, flux, g_true, args.n, args.seed)
        row = {"r_e": r_e, **_summary(resid, args.n), "elapsed_s": elapsed}
        out.append(row)
        print(
            f"r_e={r_e}: n={args.n} in {elapsed:.0f}s  bias={row['bias']}  scatter={row['scatter']}"
        )
    results = _load_results()
    results["size"] = {"g_true": list(g_true), "flux": 1e-13, "rows": out}
    _save_results(results)


def cmd_inclination(args):
    ifu = fiducial_ifu()
    flux = 1e-13 * u.erg / u.s / u.cm**2
    g_true = (0.05, 0.0)
    incs = [20.0, 40.0, 55.0, 70.0, 85.0]
    out = []
    for inc in incs:
        gal = fiducial_galaxy(inc=inc)
        est, resid, elapsed = run_ifu_ensemble(ifu, gal, flux, g_true, args.n, args.seed)
        row = {"inc": inc, **_summary(resid, args.n), "elapsed_s": elapsed}
        out.append(row)
        print(
            f"inc={inc}: n={args.n} in {elapsed:.0f}s  bias={row['bias']}  scatter={row['scatter']}"
        )
    results = _load_results()
    results["inclination"] = {"g_true": list(g_true), "flux": 1e-13, "rows": out}
    _save_results(results)


def cmd_shear(args):
    ifu = fiducial_ifu()
    gal = fiducial_galaxy()
    flux = 1e-13 * u.erg / u.s / u.cm**2
    g1s = [0.0, 0.02, 0.05, 0.08, 0.12]
    rows = []
    all_true, all_rec = [], []
    for g1 in g1s:
        g_true = (g1, 0.0)
        est, resid, elapsed = run_ifu_ensemble(ifu, gal, flux, g_true, args.n, args.seed)
        rows.append({"g_true": list(g_true), **_summary(resid, args.n), "elapsed_s": elapsed})
        all_true.extend([g_true] * args.n)
        all_rec.extend(est.tolist())
        print(f"g1={g1}: n={args.n} in {elapsed:.0f}s  bias={rows[-1]['bias']}")
    all_true = np.array(all_true)
    all_rec = np.array(all_rec)
    mc = {}
    for i, comp in enumerate(("g1", "g2")):
        # g2 is fixed at 0 throughout this sweep (only g1 varies) -- a linear
        # fit of rec vs. true is singular/undefined for a column with no
        # variation, so (m, c) is NaN there, matching
        # kino.analysis.injection.ComponentSummary's own convention for the
        # same "fewer than two distinct true shears" case.
        if len(np.unique(all_true[:, i])) >= 2:
            slope, c = np.polyfit(all_true[:, i], all_rec[:, i], 1)
            mc[comp] = {"m": float(slope - 1.0), "c": float(c)}
        else:
            mc[comp] = {"m": float("nan"), "c": float("nan")}
    print("m,c:", mc)
    results = _load_results()
    results["shear"] = {"flux": 1e-13, "rows": rows, "mc": mc}
    _save_results(results)


def cmd_null(args):
    ifu = fiducial_ifu()
    gal = fiducial_galaxy()
    flux = 1e-13 * u.erg / u.s / u.cm**2
    g_true = (0.0, 0.0)
    est, resid, elapsed = run_ifu_ensemble(ifu, gal, flux, g_true, args.n, args.seed)
    row = _summary(resid, args.n)
    print(f"null test: n={args.n} in {elapsed:.0f}s  bias={row['bias']}  scatter={row['scatter']}")
    results = _load_results()
    results["null"] = {"flux": 1e-13, **row, "elapsed_s": elapsed}
    _save_results(results)


#: Literature intrinsic shape-noise floor, Gurri, Taylor & Fluke 2021
#: (MNRAS 502, 252, doi:10.1093/mnras/stab423, arXiv:2012.09175), "an
#: effective shape noise of sigma_gamma = 0.024 +/- 0.007". KINO's own
#: idealised statistical bound has no such floor (CLAUDE.md: "real-galaxy
#: calibration of [an intrinsic floor] is out of scope for now") -- a
#: *moderately* bright target's pure-statistical scatter undershoots the
#: literature range by itself; this is expected, not a bug, and is why the
#: comparison below adds this floor in quadrature rather than comparing
#: the statistical number alone against the literature range.
_INTRINSIC_FLOOR = 0.024

#: Literature KL shape-noise range, Pranjal R. S. et al. 2022/2023, MNRAS
#: 524, 3977 (doi:10.1093/mnras/stad2014, arXiv:2209.11811), "Kinematic
#: Lensing Inference I": 0.022-0.041 depending on emission-line S/N.
_LITERATURE_RANGE = (0.022, 0.041)


def cmd_benchmark(args):
    """A moderately bright (not idealised-extreme) target, reporting both
    KINO's pure statistical scatter and that scatter combined in quadrature
    with the literature intrinsic floor (_INTRINSIC_FLOOR) -- the latter is
    the number to compare against the literature range (_LITERATURE_RANGE),
    per CLAUDE.md's own documented understanding that the statistical bound
    alone is not a real-galaxy-achievable number."""
    ifu = fiducial_ifu()
    gal = fiducial_galaxy()
    flux = 1e-13 * u.erg / u.s / u.cm**2
    g_true = (0.05, 0.0)
    est, resid, elapsed = run_ifu_ensemble(ifu, gal, flux, g_true, args.n, args.seed)
    row = _summary(resid, args.n)
    sigma_stat = 0.5 * (row["scatter"][0] + row["scatter"][1])
    sigma_total = float(np.hypot(sigma_stat, _INTRINSIC_FLOOR))
    lo, hi = _LITERATURE_RANGE
    in_range = lo <= sigma_total <= hi
    print(
        f"benchmark: n={args.n} in {elapsed:.0f}s  scatter={row['scatter']}\n"
        f"  sigma_stat={sigma_stat:.4f}  "
        f"sigma_stat+floor({_INTRINSIC_FLOOR})={sigma_total:.4f}  "
        f"literature range={_LITERATURE_RANGE}  in_range={in_range}"
    )
    results = _load_results()
    results["benchmark"] = {
        "flux": 1e-13, "g_true": list(g_true), **row,
        "sigma_eps_stat": sigma_stat, "intrinsic_floor": _INTRINSIC_FLOOR,
        "sigma_eps_stat_plus_floor": sigma_total, "literature_range": list(_LITERATURE_RANGE),
        "in_literature_range": in_range, "elapsed_s": elapsed,
    }  # fmt: skip
    _save_results(results)


def cmd_instrument_compare(args):
    # Goodman's real measured throughput (goodman.yaml) stops at 685nm -- a
    # lower redshift than the other sweeps' Z=0.06 keeps redshifted Halpha
    # within it (observed 675.97nm at Z=0.03), so this comparison rests on
    # real measured throughput for both instruments, not goodman_red.yaml's
    # fiducial extension. slit_pa matches the galaxy's own pa (real slit
    # losses are severe when misaligned -- see notebooks/11; a found
    # Nelder-Mead non-convergence at slit_pa=0/this pa=30 mismatch during
    # this script's own development is what surfaced this).
    cmp_z = 0.03
    cmp_line = 6562.8 * u.AA
    gal = fiducial_galaxy(r_e=1.5, z=cmp_z)
    g_true = (0.05, 0.0)
    flux = 3e-13 * u.erg / u.s / u.cm**2

    ifu0 = IFU.from_config("kino/config/sifs.yaml")
    ifu = narrowed_ifu(ifu0, cmp_line * (1 + cmp_z), 500.0)
    ifu = dataclasses.replace(
        ifu,
        psf=PSF(seeing_fwhm=1.0 * u.arcsec),
        detector=dataclasses.replace(ifu.detector, read_noise=_READ_NOISE),
        sky_electron_rate=_SKY_IFU,
    )
    est_i, resid_i, t_i = run_ifu_ensemble(ifu, gal, flux, g_true, args.n, args.seed, line=cmp_line)

    slit0 = SlitSpectrograph.from_config("kino/config/goodman.yaml")
    slit = narrowed_slit(slit0, cmp_line * (1 + cmp_z), 500.0)
    slit = dataclasses.replace(
        slit,
        detector=dataclasses.replace(slit.detector, read_noise=_READ_NOISE),
        sky_electron_rate=_SKY_IFU,
        slit_pa=gal.pa,
    )
    est_s, resid_s, t_s = run_slit_ensemble(
        slit, gal, flux, g_true, args.n, args.seed, line=cmp_line
    )

    row_i = _summary(resid_i, args.n)
    row_s = _summary(resid_s, args.n)
    print(f"IFU:  n={args.n} in {t_i:.0f}s  scatter={row_i['scatter']}")
    print(f"slit: n={args.n} in {t_s:.0f}s  scatter={row_s['scatter']}")
    results = _load_results()
    results["instrument_compare"] = {
        "g_true": list(g_true), "flux": 3e-13,
        "ifu": {**row_i, "elapsed_s": t_i},
        "slit": {**row_s, "elapsed_s": t_s},
    }  # fmt: skip
    _save_results(results)


COMMANDS = {
    "snr": cmd_snr,
    "size": cmd_size,
    "inclination": cmd_inclination,
    "shear": cmd_shear,
    "null": cmd_null,
    "benchmark": cmd_benchmark,
    "instrument-compare": cmd_instrument_compare,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=list(COMMANDS))
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    COMMANDS[a.command](a)
