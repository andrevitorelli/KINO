"""Tests for kino.io.download.

fetch_file() is generic (no MaNGIA/TNG-specific knowledge) and is tested
here against a local HTTP server -- no network access or credentials
needed. fetch() itself (the MaNGIA-fixture layer) can't be exercised beyond
its NotImplementedError path: this session had no TNG-project API key to
confirm real download URLs (see kino/io/download.py's module docstring).
"""

from __future__ import annotations

import hashlib
import http.server
import threading

import pytest

from kino.io.download import MANGIA_FIXTURES, fetch, fetch_file


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


class TestFetchMangiaFixture:
    def test_unknown_fixture_name_raises(self, tmp_path):
        with pytest.raises(ValueError):
            fetch("not_a_real_fixture", cache_dir=tmp_path)

    @pytest.mark.parametrize("name", sorted(MANGIA_FIXTURES))
    def test_unpopulated_fixtures_raise_not_implemented(self, name, tmp_path):
        # every fixture manifest entry has url=None until someone with TNG
        # API access fills it in -- see kino/io/download.py's module docstring.
        with pytest.raises(NotImplementedError):
            fetch(name, cache_dir=tmp_path)
