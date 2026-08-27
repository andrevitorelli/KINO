"""Fetch-on-demand for the curated test-data subset (TEST_DATA.md Section 4).

Downloads a small, named set of fixture cubes (a clean rotator, an inclined
rotator, and an interacting/tidal galaxy from MaNGIA or iMaNGA, plus one or
two KinMS known-truth cubes) into a git-ignored ``data/`` cache directory,
verified by checksum. Large data files are never committed to the repo — see
``.gitignore`` and TEST_DATA.md Section 1.

Not implemented yet: needs the actual curated fixture manifest (source URL,
object ID, checksum, download date for each entry) decided per TEST_DATA.md
Section 4 before there is anything meaningful to fetch.
"""

from __future__ import annotations

from pathlib import Path


def fetch(name: str, *, cache_dir: Path = Path("data")) -> Path:
    """Fetch a named curated fixture into ``cache_dir``, verifying its checksum.

    Returns the local path, downloading only if not already cached.
    """
    raise NotImplementedError(
        "kino.io.download is not implemented yet -- see TEST_DATA.md Section 4 "
        "(the curated fixture manifest doesn't exist yet)."
    )
