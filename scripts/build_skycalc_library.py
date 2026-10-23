#!/usr/bin/env python
"""Build the local sky-model library from ESO's SkyCalc service.

``kino.observing.skymodel`` needs sky radiance spectra (moonlight, airglow,
zodiacal light, scattered starlight) for arbitrary moon/target/airmass
geometry. ESO's Cerro Paranal Advanced Sky Model (Noll et al. 2012, A&A 543,
A92; Jones et al. 2013, A&A 560, A91) provides them through the SkyCalc web
service, which is too slow (~5 s/call) and too network-dependent to sit in a
batch run and would break bit-for-bit reproducibility. This script queries it
once, on a grid, and stores the result in ``data/skycalc/`` (git-ignored) as
plain ``.npz`` files that ``SkyLibrary`` interpolates offline.

Requires ``pip install skycalc-cli`` and network access. Interrupted runs
resume: every server call is cached under ``data/skycalc/cache/``. Progress
prints as requests actually complete (not in submission order), and every
request carries a timeout (``_patch_requests_timeout``, since
``skycalc_cli``'s own HTTP calls set none) -- a stalled connection now
retries with backoff instead of hanging the whole run forever; both were
found from a real run that printed its request count and then nothing
further -- that turned out to be an unrelated, real connectivity problem
(as of 2026-10-23, ``etimecalret-002.eso.org`` and the whole
``134.171.46.0/24`` ESO "etc" subnet are unreachable from two separate,
unrelated networks at the TCP level, not a DNS/proxy/VPN/request-volume
issue on the client side -- reported to ESO; retry once resolved).

    python scripts/build_skycalc_library.py --workers 4
    python scripts/build_skycalc_library.py --wmax 2500  # extend into the near-infrared

Modelling choices baked into the library (see ``kino.observing.skymodel``):

* Site: ESO's "paranal" (2640 m), the closest offered altitude to SOAR
  (2738 m); Cerro Pachon's light pollution and water vapour differ.
* Vacuum wavelengths; airglow averaged over all seasons and the whole night;
  water-vapour 3.5 mm (irrelevant below ~1 micron).
* Moon geometry nodes whose (target-moon separation, zenith distances)
  violate the triangle inequality SkyCalc requires are moved to the nearest
  valid separation; queries always come from real, valid geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits

# skycalc_cli's own HTTP calls (requests.post/.get) pass no `timeout=`, so a
# single stalled connection hangs forever rather than raising -- this script's
# own retry-with-backoff in _query() never gets a chance to run. Patching a
# default timeout here (before any SkyModel is constructed) turns a hang into
# a normal, retried TimeoutError; found by a real run that looked "stuck"
# after printing its request count with nothing further.
_REQUEST_TIMEOUT_S = 30


def _patch_requests_timeout() -> None:
    for name in ("post", "get"):
        orig = getattr(requests, name)
        if getattr(orig, "_kino_timeout_patched", False):
            continue

        def wrapped(*args, _orig=orig, **kwargs):
            kwargs.setdefault("timeout", _REQUEST_TIMEOUT_S)
            return _orig(*args, **kwargs)

        wrapped._kino_timeout_patched = True
        setattr(requests, name, wrapped)


# --- grids -----------------------------------------------------------------
MOON_ELONGATION = [20.0, 45.0, 75.0, 105.0, 135.0, 160.0, 180.0]  # Sun-Moon sep; 180 = full
MOON_TARGET_SEP = [10.0, 30.0, 55.0, 80.0, 110.0, 140.0, 170.0]
MOON_ZENITH_DIST = [0.0, 25.0, 45.0, 65.0, 80.0]
AIRMASS_MOON = [1.0, 1.2, 1.5, 2.0]
AIRMASS_AIRGLOW = [1.0, 1.2, 1.5, 2.0, 2.5]
SOLAR_FLUX = [70.0, 100.0, 130.0, 160.0, 200.0]  # 10.7 cm flux, s.f.u.
ZODI_LON = [30.0, 60.0, 90.0, 120.0, 150.0, 180.0]  # |longitude relative to the Sun|
ZODI_LAT = [0.0, 15.0, 30.0, 45.0, 60.0, 90.0]

OFF = {
    "incl_moon": "N", "incl_starlight": "N", "incl_zodiacal": "N",
    "incl_loweratm": "N", "incl_upperatm": "N", "incl_airglow": "N", "incl_therm": "N",
}  # fmt: skip


def _params(wmin, wmax, wdelta, **kw):
    p = {
        "wmin": wmin, "wmax": wmax, "wdelta": wdelta, "vacair": "vac",
        "wgrid_mode": "fixed_wavelength_step", "lsf_type": "none",
        "observatory": "paranal", "pwv_mode": "pwv", "pwv": 3.5, "season": 0, "time": 0,
        "airmass": 1.0, "msolflux": 100.0, "temp_flag": 0,
    }  # fmt: skip
    p.update(OFF)
    p.update(kw)
    return p


def _key(params: dict) -> str:
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:20]


def _query(cache: Path, params: dict, retries: int = 4):
    from skycalc_cli.skycalc import SkyModel

    f = cache / f"{_key(params)}.npz"
    if f.exists():
        try:
            z = np.load(f)
            return z["lam"], z["flux"]
        except Exception:  # noqa: BLE001 -- truncated file from an interrupted run: refetch
            f.unlink(missing_ok=True)
    for attempt in range(retries):
        try:
            sm = SkyModel()
            sm.stop_on_errors_and_exceptions = False
            sm.callwith(params)
            d = fits.open(io.BytesIO(sm.data))[1].data
            lam, flux = np.asarray(d["lam"], "f8"), np.asarray(d["flux"], "f4")
            tmp = f.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, lam=lam, flux=flux)
            tmp.replace(f)  # atomic, so a concurrent reader never sees a partial file
            return lam, flux
        except Exception:  # noqa: BLE001 -- network flakiness, retry
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("unreachable")


def _valid_sep(rho, z, zm):
    return float(np.clip(rho, abs(z - zm), z + zm))


def build(out: Path, workers: int, wmin: float, wmax: float, dry_run: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    fine, coarse = 0.02, 0.1

    jobs = {}
    for el in MOON_ELONGATION:
        for rho in MOON_TARGET_SEP:
            for zm in MOON_ZENITH_DIST:
                for x in AIRMASS_MOON:
                    z = float(np.degrees(np.arccos(1.0 / x)))
                    rho_v = _valid_sep(rho, z, zm)
                    jobs[("moon", el, rho, zm, x)] = _params(
                        wmin, wmax, coarse, incl_moon="Y", moon_sun_sep=el,
                        moon_target_sep=rho_v, moon_alt=90.0 - zm, airmass=x,
                    )  # fmt: skip
    for x in AIRMASS_AIRGLOW:
        for sf in SOLAR_FLUX:
            jobs[("airglow", x, sf)] = _params(
                wmin, wmax, fine, incl_loweratm="Y", incl_upperatm="Y", incl_airglow="Y",
                airmass=x, msolflux=sf,
            )  # fmt: skip
    for lon in ZODI_LON:
        for lat in ZODI_LAT:
            jobs[("zodi", lon, lat)] = _params(
                wmin, wmax, coarse, incl_zodiacal="Y", ecl_lon=lon, ecl_lat=lat
            )
    jobs[("starlight",)] = _params(wmin, wmax, coarse, incl_starlight="Y")

    print(f"{len(jobs)} SkyCalc calls ({sum(1 for k in jobs if k[0] == 'moon')} moon)")
    if dry_run:
        return

    # several grid nodes share identical SkyCalc parameters (separations clamped to the
    # valid range): query each distinct request once
    unique = {}
    for p in jobs.values():
        unique.setdefault(_key(p), p)
    print(f"{len(unique)} distinct requests")

    fetched, t0 = {}, time.time()
    # as_completed (not executor.map) reports progress as requests actually
    # finish, not in submission order -- with executor.map, one slow/stuck
    # request among the first `workers` silently withholds every later
    # result's progress line even though the pool keeps working behind it,
    # which looks identical to a true hang. See _patch_requests_timeout()
    # above for why a request can actually hang rather than fail.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_query, cache, p): h for h, p in unique.items()}
        for i, fut in enumerate(as_completed(futures), 1):
            h = futures[fut]
            fetched[h] = fut.result()
            if i % 25 == 0 or i == len(unique):
                print(f"  {i}/{len(unique)}  {time.time() - t0:.0f}s", flush=True)
    results = {k: fetched[_key(p)] for k, p in jobs.items()}

    def stack(keys, ref):
        lam = results[ref][0]
        return lam, np.stack([results[k][1] for k in keys]).astype("f4")

    nel, nrho, nzm, nx = map(
        len, (MOON_ELONGATION, MOON_TARGET_SEP, MOON_ZENITH_DIST, AIRMASS_MOON)
    )
    keys = [
        ("moon", el, rho, zm, x)
        for el in MOON_ELONGATION for rho in MOON_TARGET_SEP
        for zm in MOON_ZENITH_DIST for x in AIRMASS_MOON
    ]  # fmt: skip
    lam, arr = stack(keys, keys[0])
    np.savez_compressed(
        out / "moon.npz", lam_nm=lam, elongation=MOON_ELONGATION, target_sep=MOON_TARGET_SEP,
        zenith_dist_moon=MOON_ZENITH_DIST, airmass=AIRMASS_MOON,
        radiance=arr.reshape(nel, nrho, nzm, nx, -1),
    )  # fmt: skip

    keys = [("airglow", x, sf) for x in AIRMASS_AIRGLOW for sf in SOLAR_FLUX]
    lam, arr = stack(keys, keys[0])
    np.savez_compressed(
        out / "airglow.npz", lam_nm=lam, airmass=AIRMASS_AIRGLOW, solar_flux=SOLAR_FLUX,
        radiance=arr.reshape(len(AIRMASS_AIRGLOW), len(SOLAR_FLUX), -1),
    )  # fmt: skip

    keys = [("zodi", lo, la) for lo in ZODI_LON for la in ZODI_LAT]
    lam, arr = stack(keys, keys[0])
    np.savez_compressed(
        out / "zodi.npz", lam_nm=lam, ecl_lon=ZODI_LON, ecl_lat=ZODI_LAT,
        radiance=arr.reshape(len(ZODI_LON), len(ZODI_LAT), -1),
        starlight=results[("starlight",)][1],
    )  # fmt: skip

    (out / "meta.json").write_text(
        json.dumps(
            {
                "source": "ESO SkyCalc (skycalc_cli), Cerro Paranal Advanced Sky Model",
                "references": ["Noll et al. 2012, A&A 543, A92", "Jones et al. 2013, A&A 560, A91"],
                "units": "radiance in ph/s/m2/micron/arcsec2; wavelengths in nm (vacuum)",
                "observatory": "paranal (2640 m)",
                "built": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "wmin_nm": wmin, "wmax_nm": wmax,
            },
            indent=2,
        )
    )  # fmt: skip
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("data/skycalc"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--wmin", type=float, default=300.0, help="nm")
    ap.add_argument("--wmax", type=float, default=1000.0, help="nm")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    _patch_requests_timeout()
    build(a.out, a.workers, a.wmin, a.wmax, a.dry_run)
