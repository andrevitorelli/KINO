"""Fetch-on-demand for the curated test-data subset (TEST_DATA.md Section 4).

:func:`fetch_file` is the generic, fully-tested core: download a URL to a
destination path, verifying a sha256 checksum, skipping the download if
already cached. It has no MaNGIA-specific knowledge and needs no
credentials to test (see ``tests/io/test_download.py``, which exercises it
against a local HTTP server).

MaNGIA access, confirmed against the real TNG-project portal
---------------------------------------------------------------
A free TNG-project API key (https://www.tng-project.org/users/register/)
unlocks the real access point, confirmed live this session: MaNGIA (Sarmiento
et al. 2022) is **not** served as one URL per galaxy. ``/api/TNG50-1/files/
mangia/`` lists exactly three bulk ``tar.gz`` archives (``MANGIA_ARCHIVES``
below): ``catalog`` (~480 KB, a summary table), ``maps`` (~37 GB, the
per-galaxy stellar-population/kinematic maps -- see ``kino.io.manga``), and
``rss`` (~345 GB, raw per-fiber row-stacked spectra, not pre-gridded into a
cube). There is no per-galaxy download URL or listing endpoint.

:func:`stream_tar_member_names`/:func:`stream_extract_tar_member` are the
generic mechanism this requires: both open the remote ``tar.gz`` as a
**streaming** file object (``urllib.request`` -> ``tarfile.open(...,
mode="r|gz")``) and stop reading as soon as they've found what they need,
so pulling one ~35 MB member out of a 37 GB archive never buffers the whole
archive to memory or disk. :func:`list_mangia_members`/
:func:`fetch_mangia_member` are the MaNGIA-specific convenience wrappers
around them. Finding one *specific* subhalo's member this way still means
streaming through the archive until it's reached -- there is no index, so a
member late in the archive costs real time/bandwidth regardless; this is a
property of the real data product, not something to fix here.

Large data files are never committed to the repo -- see ``.gitignore`` and
TEST_DATA.md Section 1.
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from astropy.utils.data import download_file

TNG_API_BASE = "https://www.tng-project.org/api"

#: The three bulk MaNGIA archives actually exposed by
#: ``/api/TNG50-1/files/mangia/`` -- see module docstring. Sizes as measured
#: this session (2026-10-21); no published checksum exists for any of them.
MANGIA_ARCHIVES: dict[str, str] = {
    "catalog": "mangia_catalog.tar.gz",
    "maps": "mangia_maps.tar.gz",
    "rss": "mangia_rss.tar.gz",
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


def stream_tar_member_names(
    url: str, *, headers: dict[str, str] | None = None, limit: int | None = None
) -> Iterator[str]:
    """Yield member names of the remote ``.tar.gz`` at ``url``, without
    downloading or extracting any file content.

    Stops reading the remote stream as soon as ``limit`` names have been
    yielded (``None``: read to the end of the archive).
    """
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tf:
        for i, member in enumerate(tf):
            if limit is not None and i >= limit:
                return
            yield member.name


def stream_extract_tar_members(
    url: str,
    members: dict[str, Path],
    *,
    headers: dict[str, str] | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Extract several named members from the remote ``.tar.gz`` at ``url``
    in **one streaming pass**, writing each to its own destination in
    ``members`` (``{member name: dest path}``).

    Stops reading the remote stream as soon as every requested member has
    been found, so asking for several members scattered through a large
    archive costs one scan, not one per member (the naive per-member
    approach :func:`stream_extract_tar_member` uses, fine for one member
    but wasteful for many -- see module docstring).

    Members whose ``dest`` already exists are skipped (not re-extracted)
    unless ``overwrite=True``, and are not searched for in the stream.

    Returns
    -------
    dict[str, Path]
        ``{member name: dest path}`` for every member actually extracted
        (or already cached) this call -- members not found in the archive
        are simply absent, not an error (callers can diff against
        ``members`` to see what's missing).
    """
    pending = {
        name: Path(dest) for name, dest in members.items() if overwrite or not Path(dest).exists()
    }
    found: dict[str, Path] = {
        name: Path(dest) for name, dest in members.items() if name not in pending
    }
    if not pending:
        return found

    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tf:
        for entry in tf:
            if entry.name not in pending:
                continue
            extracted = tf.extractfile(entry)
            if extracted is None:
                continue  # not a regular file (e.g. a directory entry)
            dest = pending.pop(entry.name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as out:
                shutil.copyfileobj(extracted, out)
            found[entry.name] = dest
            if not pending:
                break
    return found


def stream_extract_tar_member(
    url: str,
    member: str,
    dest: Path,
    *,
    headers: dict[str, str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Extract exactly ``member`` from the remote ``.tar.gz`` at ``url`` to
    ``dest`` (see :func:`stream_extract_tar_members` -- fetching several
    members from the same archive is far cheaper as one call to that
    function than one call each to this one).

    Raises
    ------
    KeyError
        If ``member`` is not found anywhere in the archive (the whole
        archive is streamed through in this case).
    """
    found = stream_extract_tar_members(
        url, {member: Path(dest)}, headers=headers, overwrite=overwrite
    )
    if member not in found:
        raise KeyError(f"{member!r} not found in {url}")
    return found[member]


def mangia_archive_url(which: str = "maps") -> str:
    """URL of one of the three bulk MaNGIA archives (see module docstring)."""
    if which not in MANGIA_ARCHIVES:
        raise ValueError(f"which must be one of {sorted(MANGIA_ARCHIVES)}, got {which!r}")
    return f"{TNG_API_BASE}/TNG50-1/files/{MANGIA_ARCHIVES[which]}"


def list_mangia_members(which: str = "maps", *, api_key: str, limit: int | None = 20) -> list[str]:
    """List member names in one of the bulk MaNGIA archives.

    Parameters
    ----------
    which : {"catalog", "maps", "rss"}, optional
    api_key : str
        TNG-project API key (https://www.tng-project.org/users/register/),
        sent as the ``api-key`` header.
    limit : int or None, optional
        Stop after this many names (default 20, since the full ``maps``
        archive alone has tens of thousands of members -- pass ``None`` for
        every member, which means streaming the whole archive).
    """
    url = mangia_archive_url(which)
    return list(stream_tar_member_names(url, headers={"api-key": api_key}, limit=limit))


#: Each archive's real per-galaxy member-name pattern (confirmed against
#: live downloads this session): directory prefix + filename suffix,
#: sandwiching ``TNG50-{snapshot}-{subhalo_id}-{view}-{n_fibers}``.
_MANGIA_MEMBER_PATTERN = {
    "maps": ("maps/", ".cube_maps.fits"),
    "rss": ("RSS/", ".cube_RSS.fits"),
}


def mangia_member_name(
    snapshot: int, subhalo_id: int, view: int, n_fibers: int, which: str = "maps"
) -> str:
    """One galaxy's member name within the ``which`` MaNGIA archive --
    ``"maps"``: ``maps/TNG50-{snapshot}-{subhalo_id}-{view}-{n_fibers}.cube_maps.fits``;
    ``"rss"``: ``RSS/TNG50-{snapshot}-{subhalo_id}-{view}-{n_fibers}.cube_RSS.fits``.
    """
    if which not in _MANGIA_MEMBER_PATTERN:
        raise ValueError(f"which must be one of {sorted(_MANGIA_MEMBER_PATTERN)}, got {which!r}")
    prefix, suffix = _MANGIA_MEMBER_PATTERN[which]
    return f"{prefix}TNG50-{snapshot}-{subhalo_id}-{view}-{n_fibers}{suffix}"


def fetch_mangia_member(
    member: str,
    dest: Path,
    *,
    api_key: str,
    which: str = "maps",
    overwrite: bool = False,
) -> Path:
    """Fetch exactly one named member out of a bulk MaNGIA archive (see
    module docstring) -- the practical way to get one real galaxy's file
    without downloading the full archive. ``member`` is typically built
    with :func:`mangia_member_name`. Fetching **several** members from the
    same ``which`` archive is far cheaper via :func:`fetch_mangia_members`.
    """
    url = mangia_archive_url(which)
    return stream_extract_tar_member(
        url, member, Path(dest), headers={"api-key": api_key}, overwrite=overwrite
    )


def fetch_mangia_members(
    members: dict[str, Path],
    *,
    api_key: str,
    which: str = "maps",
    overwrite: bool = False,
) -> dict[str, Path]:
    """Fetch several named members out of one bulk MaNGIA archive in a
    single streaming pass (see :func:`stream_extract_tar_members`) --
    the efficient way to pull a sample of galaxies, rather than calling
    :func:`fetch_mangia_member` once per galaxy (one archive scan each).

    Parameters
    ----------
    members : dict[str, Path]
        ``{member name: dest path}``, e.g. built with
        :func:`mangia_member_name`.
    api_key : str
    which : {"catalog", "maps", "rss"}, optional
    overwrite : bool, optional

    Returns
    -------
    dict[str, Path]
        As :func:`stream_extract_tar_members`: only the members actually
        found/extracted (or already cached).
    """
    url = mangia_archive_url(which)
    return stream_extract_tar_members(
        url, members, headers={"api-key": api_key}, overwrite=overwrite
    )


def fetch_mangia_archive(
    cache_dir: Path, *, api_key: str, which: str = "maps", overwrite: bool = False
) -> Path:
    """Download a **whole** bulk MaNGIA archive to ``cache_dir`` -- genuinely
    large (``rss`` is ~345 GB, ``maps`` ~37 GB; only ``catalog`` at ~480 KB
    is small) and slow. Prefer :func:`fetch_mangia_member` for one galaxy;
    use this only if you actually want the full archive on disk. No
    published checksum exists for any of these archives.
    """
    url = mangia_archive_url(which)
    dest = Path(cache_dir) / MANGIA_ARCHIVES[which]
    return fetch_file(url, dest, headers={"api-key": api_key}, overwrite=overwrite)
