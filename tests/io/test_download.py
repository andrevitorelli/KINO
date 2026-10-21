"""Tests for kino.io.download.

fetch_file() and the generic streaming-tar functions (stream_tar_member_names,
stream_extract_tar_member(s)) have no MaNGIA/TNG-specific knowledge and are
tested here against a local HTTP server serving a small test .tar.gz -- no
network access or credentials needed. The MaNGIA-specific wrappers
(fetch_mangia_member(s), list_mangia_members) are thin enough (URL + header
construction around the already-tested generic functions) that they aren't
separately covered here; mangia_archive_url/mangia_member_name (pure string
construction) are.
"""

from __future__ import annotations

import hashlib
import http.server
import io
import tarfile
import threading

import pytest

from kino.io.download import (
    fetch_file,
    mangia_archive_url,
    mangia_member_name,
    stream_extract_tar_member,
    stream_extract_tar_members,
    stream_tar_member_names,
)


@pytest.fixture
def http_server(tmp_path):
    directory = tmp_path / "served"
    directory.mkdir()

    def handler(*args, **kwargs):
        return http.server.SimpleHTTPRequestHandler(*args, directory=str(directory), **kwargs)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, directory
    finally:
        server.shutdown()
        thread.join()


def _url(server, name: str) -> str:
    host, port = server.server_address[:2]
    return f"http://{host}:{port}/{name}"


def _write_test_tarball(path, contents: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name, data in contents.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


class TestFetchFile:
    def test_downloads_and_caches(self, http_server, tmp_path):
        server, served_dir = http_server
        content = b"hello kino"
        (served_dir / "data.bin").write_bytes(content)

        dest = tmp_path / "out" / "data.bin"
        result = fetch_file(_url(server, "data.bin"), dest)
        assert result == dest
        assert dest.read_bytes() == content

    def test_verifies_checksum(self, http_server, tmp_path):
        server, served_dir = http_server
        content = b"checksum me"
        (served_dir / "data.bin").write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()

        dest = tmp_path / "data.bin"
        result = fetch_file(_url(server, "data.bin"), dest, sha256=digest)
        assert result.read_bytes() == content

    def test_raises_on_checksum_mismatch(self, http_server, tmp_path):
        server, served_dir = http_server
        (served_dir / "data.bin").write_bytes(b"wrong content")

        dest = tmp_path / "data.bin"
        with pytest.raises(ValueError):
            fetch_file(_url(server, "data.bin"), dest, sha256="0" * 64)
        assert not dest.exists()

    def test_skips_download_if_already_cached_with_matching_checksum(self, http_server, tmp_path):
        server, served_dir = http_server
        content = b"cached content"
        digest = hashlib.sha256(content).hexdigest()
        (served_dir / "data.bin").write_bytes(content)

        dest = tmp_path / "data.bin"
        dest.write_bytes(content)
        mtime_before = dest.stat().st_mtime_ns

        # remove the served file entirely -- if fetch_file tried to
        # re-download, this would fail
        (served_dir / "data.bin").unlink()
        result = fetch_file(_url(server, "data.bin"), dest, sha256=digest)
        assert result.stat().st_mtime_ns == mtime_before

    def test_redownloads_if_cached_checksum_does_not_match(self, http_server, tmp_path):
        server, served_dir = http_server
        new_content = b"the real content"
        (served_dir / "data.bin").write_bytes(new_content)
        digest = hashlib.sha256(new_content).hexdigest()

        dest = tmp_path / "data.bin"
        dest.write_bytes(b"stale content")

        result = fetch_file(_url(server, "data.bin"), dest, sha256=digest)
        assert result.read_bytes() == new_content

    def test_overwrite_forces_redownload(self, http_server, tmp_path):
        server, served_dir = http_server
        new_content = b"fresh"
        (served_dir / "data.bin").write_bytes(new_content)

        dest = tmp_path / "data.bin"
        dest.write_bytes(b"old, no checksum given so would normally be kept")

        result = fetch_file(_url(server, "data.bin"), dest, overwrite=True)
        assert result.read_bytes() == new_content


class TestStreamTarMemberNames:
    def test_lists_members(self, http_server, tmp_path):
        server, served_dir = http_server
        _write_test_tarball(served_dir / "archive.tar.gz", {"a.txt": b"1", "b.txt": b"2"})
        names = list(stream_tar_member_names(_url(server, "archive.tar.gz")))
        assert names == ["a.txt", "b.txt"]

    def test_respects_limit(self, http_server, tmp_path):
        server, served_dir = http_server
        _write_test_tarball(
            served_dir / "archive.tar.gz", {"a.txt": b"1", "b.txt": b"2", "c.txt": b"3"}
        )
        names = list(stream_tar_member_names(_url(server, "archive.tar.gz"), limit=2))
        assert names == ["a.txt", "b.txt"]


class TestStreamExtractTarMembers:
    def test_extracts_requested_members_only(self, http_server, tmp_path):
        server, served_dir = http_server
        _write_test_tarball(
            served_dir / "archive.tar.gz",
            {"a.txt": b"content-a", "b.txt": b"content-b", "c.txt": b"content-c"},
        )
        targets = {"a.txt": tmp_path / "a_out.txt", "c.txt": tmp_path / "c_out.txt"}
        found = stream_extract_tar_members(_url(server, "archive.tar.gz"), targets)
        assert set(found) == {"a.txt", "c.txt"}
        assert targets["a.txt"].read_bytes() == b"content-a"
        assert targets["c.txt"].read_bytes() == b"content-c"
        assert not (tmp_path / "b_out.txt").exists()

    def test_missing_member_is_absent_from_result_not_an_error(self, http_server, tmp_path):
        server, served_dir = http_server
        _write_test_tarball(served_dir / "archive.tar.gz", {"a.txt": b"content-a"})
        targets = {"a.txt": tmp_path / "a_out.txt", "nope.txt": tmp_path / "nope_out.txt"}
        found = stream_extract_tar_members(_url(server, "archive.tar.gz"), targets)
        assert set(found) == {"a.txt"}

    def test_skips_already_cached_dest_unless_overwrite(self, http_server, tmp_path):
        server, served_dir = http_server
        _write_test_tarball(served_dir / "archive.tar.gz", {"a.txt": b"fresh-content"})
        dest = tmp_path / "a_out.txt"
        dest.write_bytes(b"stale")

        found = stream_extract_tar_members(_url(server, "archive.tar.gz"), {"a.txt": dest})
        assert found["a.txt"].read_bytes() == b"stale"

        found = stream_extract_tar_members(
            _url(server, "archive.tar.gz"), {"a.txt": dest}, overwrite=True
        )
        assert found["a.txt"].read_bytes() == b"fresh-content"


class TestStreamExtractTarMember:
    def test_extracts_one_member(self, http_server, tmp_path):
        server, served_dir = http_server
        _write_test_tarball(served_dir / "archive.tar.gz", {"a.txt": b"content-a"})
        dest = tmp_path / "out.txt"
        result = stream_extract_tar_member(_url(server, "archive.tar.gz"), "a.txt", dest)
        assert result == dest
        assert dest.read_bytes() == b"content-a"

    def test_raises_keyerror_if_not_found(self, http_server, tmp_path):
        server, served_dir = http_server
        _write_test_tarball(served_dir / "archive.tar.gz", {"a.txt": b"content-a"})
        with pytest.raises(KeyError):
            dest = tmp_path / "out.txt"
            stream_extract_tar_member(_url(server, "archive.tar.gz"), "nope.txt", dest)


class TestMangiaUrlHelpers:
    def test_mangia_archive_url_builds_expected_url(self):
        assert mangia_archive_url("maps") == (
            "https://www.tng-project.org/api/TNG50-1/files/mangia_maps.tar.gz"
        )

    def test_mangia_archive_url_rejects_unknown_which(self):
        with pytest.raises(ValueError):
            mangia_archive_url("not_a_real_archive")

    def test_mangia_member_name_maps(self):
        assert (
            mangia_member_name(97, 664052, 0, 127, which="maps")
            == "maps/TNG50-97-664052-0-127.cube_maps.fits"
        )

    def test_mangia_member_name_rss(self):
        assert (
            mangia_member_name(97, 664052, 0, 127, which="rss")
            == "RSS/TNG50-97-664052-0-127.cube_RSS.fits"
        )

    def test_mangia_member_name_rejects_unknown_which(self):
        with pytest.raises(ValueError):
            mangia_member_name(97, 664052, 0, 127, which="not_a_real_kind")
