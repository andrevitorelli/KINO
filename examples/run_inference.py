"""Recover the (g1, g2) shear posterior for one mock galaxy observation.

Loads a mock data product written by simulate_one_galaxy.py and runs the
shear inference (likelihood + priors + nautilus sampler) to recover the
posterior on (g1, g2).

Usage:
    python examples/run_inference.py --data mock.fits --config config/gmtifs.yaml

Placeholder for the Phase 5 inference engine (see the roadmap in README.md);
raises NotImplementedError until kino.inference is implemented.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True, help="Mock data product written by kino.io."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Instrument YAML config (see config/)."
    )
    parser.add_argument(
        "--sampler", default="nautilus", choices=["nautilus"], help="Posterior sampler."
    )
    parser.add_argument(
        "--out", type=Path, default=Path("posterior.h5"), help="Output posterior path."
    )
    return parser.parse_args()


def main() -> None:
    parse_args()

    # 1. load mock data           -> kino.io
    # 2. build likelihood + priors -> kino.inference
    # 3. sample posterior          -> kino.inference.fit_shear(data, sampler=args.sampler)
    # 4. save chains + summary     -> kino.io
    raise NotImplementedError(
        "The inference engine (kino.inference, nautilus wrapper) is not "
        "implemented yet -- see the Phase 5 roadmap in README.md."
    )


if __name__ == "__main__":
    main()
