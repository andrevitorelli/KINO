"""Fetch-on-demand for the curated test-data subset (TEST_DATA.md Section 4).

:func:`fetch_file` is the generic, fully-tested core: download a URL to a
destination path, verifying a sha256 checksum, skipping the download if
already cached. It has no MaNGIA-specific knowledge and needs no
credentials to test (see ``tests/io/test_download.py``, which exercises it
against a local HTTP server).

:func:`fetch` is the MaNGIA-specific fixture layer on top of it, and is
**not fully usable yet**: MaNGIA's premade cubes are hosted on the
IllustrisTNG data portal (TNG50-1 supplementary data), gated behind free
user registration and a personal API key -- https://www.tng-project.org/users/register/.
This session had no such key, so it could not browse the actual TNG50-1
MaNGIA catalog to pick real galaxy IDs for TEST_DATA.md Section 4's curated
set (a clean rotator, an inclined rotator, an interacting/tidal galaxy) or
confirm the exact per-file download URL. ``MANGIA_FIXTURES`` below is a
ready-to-fill-in manifest -- populate each entry's ``url``/``sha256`` once
you have API access (or call :func:`fetch_file` directly with a URL/checksum
you already have). See ``CLAUDE.md``'s status banner.

Large data files are never committed to the repo -- see ``.gitignore`` and
TEST_DATA.md Section 1.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from astropy.utils.data import download_file

TNG_API_BASE = "https://www.tng-project.org/api"

#: TEST_DATA.md Section 4's curated MaNGIA fixture set. Each entry's ``url``
#: and ``sha256`` are placeholders (``None``) pending TNG-project API access
#: to confirm them -- see the module docstring. ``description`` documents the
#: *intent* (what TEST_DATA.md needs this fixture for), independent of which
#: real subhalo eventually fills that role.
MANGIA_FIXTURES: dict[str, dict] = {
    "mangia_regular_faceon": {
        "url": None,
        "sha256": None,
        "description": "Clean, regularly rotating disc, near face-on.",
    },
    "mangia_regular_inclined": {
        "url": None,
        "sha256": None,
        "description": "Clean, regularly rotating disc, inclined -- for PA/Vlos convention checks.",
    },
    "mangia_interacting": {
        "url": None,
        "sha256": None,
        "description": (
            "Interacting/tidally disturbed galaxy -- the model-comparison null test "
            "in TEST_DATA.md Section 3 (evidence should favour the tidal model, not KL)."
        ),
    },
}


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_file(
    url: str,
    dest: Path,
    *,
    sha256: str | None = None,
    headers: dict[str, str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Download ``url`` to ``dest``, verifying ``sha256`` if given.

    Skips the download (returns immediately) if ``dest`` already exists and
    (when ``sha256`` is given) matches it, unless ``overwrite=True``.

    Raises
    ------
    ValueError
        If the downloaded file's sha256 doesn't match ``sha256``.
    """
    dest = Path(dest)
    if dest.exists() and not overwrite and (sha256 is None or _sha256_of(dest) == sha256):
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    cached_path = download_file(url, cache=False, http_headers=headers, show_progress=False)

    if sha256 is not None:
        digest = _sha256_of(Path(cached_path))
        if digest != sha256:
            raise ValueError(f"checksum mismatch for {url}: expected {sha256}, got {digest}")

    shutil.copy(cached_path, dest)
    return dest


def fetch(
    name: str,
    *,
    cache_dir: Path = Path("data"),
    api_key: str | None = None,
) -> Path:
    """Fetch a named curated MaNGIA fixture (TEST_DATA.md Section 4) into ``cache_dir``.

    Parameters
    ----------
    name : str
        A key of ``MANGIA_FIXTURES``.
    cache_dir : Path, optional
        Git-ignored local cache directory.
    api_key : str or None, optional
        IllustrisTNG API key, sent as the ``api-key`` header
        (https://www.tng-project.org/data/docs/api/).

    Raises
    ------
    NotImplementedError
        If ``MANGIA_FIXTURES[name]["url"]`` hasn't been filled in yet (see
        the module docstring) -- this is expected until someone with TNG
        API access populates the manifest.
    """
    if name not in MANGIA_FIXTURES:
        raise ValueError(f"unknown fixture {name!r}; choose from {sorted(MANGIA_FIXTURES)}")
    entry = MANGIA_FIXTURES[name]
    if entry["url"] is None:
        raise NotImplementedError(
            f"fixture {name!r} has no confirmed download URL yet. MaNGIA cubes require a "
            "TNG-project API key to browse (https://www.tng-project.org/users/register/); "
            "populate MANGIA_FIXTURES[...]['url'] and ['sha256'] once you have access, or "
            "call fetch_file() directly with a URL/checksum you already have."
        )
    headers = {"api-key": api_key} if api_key else None
    dest = Path(cache_dir) / f"{name}.fits.gz"
    return fetch_file(entry["url"], dest, sha256=entry["sha256"], headers=headers)
