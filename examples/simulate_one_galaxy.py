"""Simulate one sheared disc galaxy through an instrument backend.

Given an intrinsic galaxy configuration, a reduced shear (g1, g2), and an
instrument config, produce that instrument's mock data products:

- IFU backend         -> simulated image + datacube
- slit/grism backend  -> simulated image + measured (dispersed) spectrum

Usage:
    python examples/simulate_one_galaxy.py --config config/gmtifs.yaml \\
        --g1 0.03 --g2 -0.01 --out mock.fits

Placeholder for the Phase 1-4 forward-model pipeline (see the roadmap in
README.md); raises NotImplementedError until kino.galaxy, kino.lensing,
kino.instruments, and kino.mocks are implemented.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="Instrument YAML config (see config/)."
    )
    parser.add_argument("--g1", type=float, default=0.0, help="Reduced shear component g1.")
    parser.add_argument("--g2", type=float, default=0.0, help="Reduced shear component g2.")
    parser.add_argument(
        "--out", type=Path, default=Path("mock.fits"), help="Output mock data path."
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the numpy.random.Generator.")
    return parser.parse_args()


def main() -> None:
    parse_args()

    # 1. intrinsic disc          -> kino.galaxy.DiscGalaxy
    # 2. apply shear (g1, g2)    -> kino.lensing.apply_shear
    # 3. instrument backend      -> kino.instruments.IFU / SlitSpectrograph, from --config
    # 4. assemble + write mock   -> kino.mocks, kino.io
    raise NotImplementedError(
        "The forward-model pipeline (kino.galaxy, kino.lensing, kino.instruments, "
        "kino.mocks) is not implemented yet -- see the Phase 1-4 roadmap in README.md."
    )


if __name__ == "__main__":
    main()
