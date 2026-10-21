"""Tests for kino.mocks.catalogue.generate_batch (README.md Phase 4)."""

import hashlib
import json

import pytest
import yaml

from kino.inference.likelihood import GALAXY_FIELDS
from kino.mocks.catalogue import _OBSERVATION_FIELDS, _allowed_fields, generate_batch

EXPLICIT_CATALOGUE = {
    "instrument": {"kind": "ifu", "config": "kino/config/sifs.yaml"},
    "shear": {"g1": 0.03, "g2": -0.01},
    "seed": 42,
    "galaxies": [
        {
            "r_e": 0.5, "inc": 60, "pa": 30, "v_flat": 180, "z": 0.1, "sigma0": 30,
            "line_flux": "1.0e-15 erg / (s cm2)", "line_wavelength": "0.65628 um",
        },
        {
            "r_e": 0.8, "inc": 45, "pa": 10, "v_flat": 220, "z": 0.15, "sigma0": 35,
            "line_flux": "5.0e-15 erg / (s cm2)", "line_wavelength": "0.65628 um",
        },
    ],
}  # fmt: skip

POPULATION_CATALOGUE = {
    "instrument": {"kind": "ifu", "config": "kino/config/sifs.yaml"},
    "shear": {"g1": 0.02, "g2": 0.01},
    "seed": 7,
    "population": {
        "n": 3,
        "r_e": {"dist": "uniform", "low": 0.3, "high": 1.0},
        "inc": {"dist": "uniform", "low": 20, "high": 80},
        "pa": {"dist": "uniform", "low": 0, "high": 180},
        "v_flat": {"dist": "lognormal", "median": 180, "scatter_dex": 0.1},
        "z": {"dist": "uniform", "low": 0.05, "high": 0.5},
        "sigma0": {"dist": "uniform", "low": 20, "high": 40},
        "line_flux": {
            "dist": "loguniform", "low": 1.0e-16, "high": 1.0e-14, "unit": "erg / (s cm2)",
        },
        "line_wavelength": "0.65628 um",
    },
}  # fmt: skip


def _write_yaml(tmp_path, data, name="catalogue.yaml"):
    path = tmp_path / name
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    return path


def _md5(path) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


class TestAllowedFields:
    def test_matches_galaxy_fields_plus_observation_fields(self):
        """Pins that kino.mocks.catalogue's own field list, built from
        dataclasses.fields(DiscGalaxy) without importing kino.inference,
        agrees with kino.inference.likelihood.GALAXY_FIELDS -- the two
        can't silently diverge."""
        assert _allowed_fields() == set(GALAXY_FIELDS) | set(_OBSERVATION_FIELDS)


class TestExplicitGalaxyList:
    def test_writes_one_file_per_galaxy_and_a_manifest(self, tmp_path):
        catalogue_path = _write_yaml(tmp_path, EXPLICIT_CATALOGUE)
        out_dir = tmp_path / "out"
        paths = generate_batch(catalogue_path, out_dir, format="fits")

        assert len(paths) == 2
        assert all(p.exists() for p in paths)
        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert len(manifest) == 2
        assert manifest[0]["r_e"] == 0.5
        assert manifest[0]["g1"] == 0.03
        assert manifest[1]["r_e"] == 0.8

    def test_unknown_field_raises(self, tmp_path):
        bad = json.loads(json.dumps(EXPLICIT_CATALOGUE))  # deep copy
        bad["galaxies"][0]["not_a_real_field"] = 1.0
        catalogue_path = _write_yaml(tmp_path, bad)
        with pytest.raises(ValueError, match="unknown catalogue field"):
            generate_batch(catalogue_path, tmp_path / "out", format="fits")

    def test_missing_line_flux_raises(self, tmp_path):
        bad = json.loads(json.dumps(EXPLICIT_CATALOGUE))
        del bad["galaxies"][0]["line_flux"]
        catalogue_path = _write_yaml(tmp_path, bad)
        with pytest.raises(ValueError, match="missing required field"):
            generate_batch(catalogue_path, tmp_path / "out", format="fits")

    def test_both_galaxies_and_population_set_raises(self, tmp_path):
        bad = json.loads(json.dumps(EXPLICIT_CATALOGUE))
        bad["population"] = {"n": 1, "r_e": 0.5}
        catalogue_path = _write_yaml(tmp_path, bad)
        with pytest.raises(ValueError, match="exactly one"):
            generate_batch(catalogue_path, tmp_path / "out", format="fits")


class TestPopulationSampling:
    def test_same_seed_is_byte_reproducible(self, tmp_path):
        catalogue_path = _write_yaml(tmp_path, POPULATION_CATALOGUE)
        paths_a = generate_batch(catalogue_path, tmp_path / "a", format="fits")
        paths_b = generate_batch(catalogue_path, tmp_path / "b", format="fits")

        assert len(paths_a) == len(paths_b) == 3
        for a, b in zip(paths_a, paths_b, strict=True):
            assert _md5(a) == _md5(b)
        manifest_a = (tmp_path / "a" / "manifest.json").read_text()
        manifest_b = (tmp_path / "b" / "manifest.json").read_text()
        assert manifest_a == manifest_b

    def test_sampled_values_are_within_bounds(self, tmp_path):
        catalogue_path = _write_yaml(tmp_path, POPULATION_CATALOGUE)
        out_dir = tmp_path / "out"
        generate_batch(catalogue_path, out_dir, format="fits")
        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert len(manifest) == 3
        for row in manifest:
            assert 0.3 <= row["r_e"] <= 1.0
            assert 20 <= row["inc"] <= 80
            assert row["line_wavelength"] == "0.65628 um"
