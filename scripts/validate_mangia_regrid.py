#!/usr/bin/env python
"""Validate kino.io.manga's RSS-to-cube regridding against real MaNGIA data.

For every matched (rss, maps) file pair found under a data directory (see
kino.io.download's module docstring for how to fetch a sample -- e.g.
fetch_mangia_members), regrids the rss file's per-fibre ground-truth VEL/DISP
the same way kino.io.manga.regrid_rss_to_cube regrids flux, onto the matching
maps file's own spatial grid, and compares against that maps file's own
official, pre-gridded VEL/DISP for the *same* galaxy.

This checks the regridding *geometry* in isolation (no spectral fitting
involved, since both sides start from the same ground-truth simulation
values) -- it is the empirical validation kino/io/manga.py's docstring
refers to.

    python scripts/validate_mangia_regrid.py data/mangia

Writes a summary to notebooks/data/mangia_regrid_validation.json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from kino.io.manga import read_mangia_maps, read_mangia_rss, regrid_fibers_to_grid

_CORE_RE = re.compile(r"(TNG50-\d+-\d+-\d+-\d+)\.cube_(RSS|maps)\.fits")


def find_pairs(data_dir: Path) -> list[tuple[str, Path, Path]]:
    """Matched (core_name, rss_path, maps_path) triples under ``data_dir``."""
    rss_paths, maps_paths = {}, {}
    for path in data_dir.glob("*.fits"):
        m = _CORE_RE.search(path.name)
        if m is None:
            continue
        core, kind = m.group(1), m.group(2)
        (rss_paths if kind == "RSS" else maps_paths)[core] = path
    return [
        (core, rss_paths[core], maps_paths[core])
        for core in sorted(rss_paths)
        if core in maps_paths
    ]


def validate_one(rss_path: Path, maps_path: Path) -> dict:
    maps = read_mangia_maps(maps_path)
    rss = read_mangia_rss(rss_path)
    ny, nx = maps.maps["VEL"].shape

    result: dict = {}
    for name in ("VEL", "DISP"):
        regridded = regrid_fibers_to_grid(
            rss.x_fiber, rss.y_fiber, rss.fiber_maps[name],
            pixel_scale=maps.pixel_scale, nx=nx, ny=ny, kernel_fwhm=rss.psf_fwhm,
        )  # fmt: skip
        official = maps.maps[name]
        in_fov = official != 0.0
        resid = regridded[in_fov] - official[in_fov]
        scale = float(np.std(official[in_fov]))
        result[name] = {
            "n_spaxels": int(in_fov.sum()),
            "official_std": scale,
            "resid_std": float(np.std(resid)),
            "resid_median": float(np.median(resid)),
            "resid_std_over_official_std": float(np.std(resid) / scale) if scale > 0 else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data_dir", type=Path, help="Directory with cached MaNGIA .fits files.")
    parser.add_argument(
        "--out", type=Path, default=Path("notebooks/data/mangia_regrid_validation.json")
    )
    args = parser.parse_args()

    pairs = find_pairs(args.data_dir)
    if not pairs:
        raise SystemExit(f"no matched (rss, maps) pairs found under {args.data_dir}")

    print(f"{len(pairs)} matched pair(s) found")
    all_results = {}
    for core, rss_path, maps_path in pairs:
        result = validate_one(rss_path, maps_path)
        all_results[core] = result
        for name, stats in result.items():
            print(
                f"{core:30s} {name:5s} resid_std/official_std = "
                f"{stats['resid_std_over_official_std']:.3f}  (n={stats['n_spaxels']})"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
