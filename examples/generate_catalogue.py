"""Generate a batch of mock observations from a YAML galaxy catalogue.

Thin CLI around :func:`kino.mocks.catalogue.generate_batch` (README.md
Phase 4): one instrument, one global shear `(g1, g2)`, and either an
explicit list of galaxies or a sampled population -- see
`kino/config/catalogue_example.yaml` for a documented template and
`docs/manual/batch-mocks.md` for the full schema.

Usage:
    python examples/generate_catalogue.py \\
        --catalogue kino/config/catalogue_example.yaml --out mock_batch/

Writes one data file per galaxy (`galaxy_0000.fits`, ...) plus
`manifest.json` recording each galaxy's true parameters, into `--out`.
Reproducible: re-running with the same catalogue file regenerates
byte-identical output (the catalogue's own `seed` is the only source of
randomness).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kino.mocks.catalogue import generate_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--catalogue", type=Path, required=True, help="Catalogue YAML (see kino/config/)."
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output directory (created if missing)."
    )
    parser.add_argument(
        "--format", choices=["fits", "hdf5"], default="fits", help="Output format (default fits)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = generate_batch(args.catalogue, args.out, format=args.format)
    print(f"wrote {len(paths)} mock(s) + manifest.json to {args.out}")


if __name__ == "__main__":
    main()
