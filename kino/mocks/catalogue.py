"""Batch mock generation from a YAML catalogue (README.md Phase 4).

A catalogue config names one instrument, **one global reduced shear
``(g1, g2)`` applied to every galaxy** (matching every existing KINO
forecast/notebook's treatment of shear -- a spatially-varying shear field
is out of scope here, see ``CLAUDE.md``/the Phase 4 plan), a top-level
``seed``, and either:

* ``galaxies``: an explicit list of per-galaxy parameter rows, or
* ``population``: ``n`` galaxies sampled from per-field distributions.

Each row/spec may set any :class:`~kino.galaxy.DiscGalaxy` constructor
field plus the two observation-level extras ``line_flux``/
``line_wavelength`` (both parsed with
:func:`kino.instruments.config.parse_quantity`, so they take the same
``"<value> <unit>"`` string form every instrument config already uses).
An unknown key raises :class:`ValueError` naming the allowed set.

Supported ``population`` distributions (deliberately a small, fixed set,
not a general DSL -- extend by adding a branch in :func:`_sample_value`):

* ``{dist: uniform, low, high}``
* ``{dist: loguniform, low, high}`` (``low`` and ``high`` must be positive)
* ``{dist: lognormal, median, scatter_dex}`` -- ``log10`` scatter, matching
  :func:`kino.inference.priors.tully_fisher_prior`'s own parameterization.
* a bare scalar (or unit string) -- fixed for every sampled galaxy.

Reproducibility
----------------
One top-level ``seed`` is spawned into ``n + 1`` independent
``numpy.random.SeedSequence`` children (the same pattern
:func:`kino.analysis.injection._trial_seeds` uses): child 0 seeds the
population sampler (unused for an explicit ``galaxies`` list), children
``1..n`` each seed one galaxy's noise realisation. Re-running
:func:`generate_batch` with the same catalogue file therefore regenerates
byte-identical output files and an identical ``manifest.json``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import yaml

from kino.galaxy import DiscGalaxy
from kino.instruments.config import parse_quantity
from kino.instruments.ifu import IFU
from kino.instruments.slit import SlitObservation, SlitSpectrograph
from kino.io import fitsio, hdf5
from kino.mocks.single import simulate_observation

_OBSERVATION_FIELDS = ("line_flux", "line_wavelength")
_WRITERS = {
    "fits": (fitsio.write_datacube, fitsio.write_slit_observation, "fits"),
    "hdf5": (hdf5.write_datacube, hdf5.write_slit_observation, "h5"),
}


def _allowed_fields() -> frozenset[str]:
    galaxy_fields = {f.name for f in dataclasses.fields(DiscGalaxy)}
    return frozenset(galaxy_fields | set(_OBSERVATION_FIELDS))


def _validate_row(row: dict, allowed: frozenset[str]) -> None:
    bad = set(row) - allowed
    if bad:
        raise ValueError(
            f"unknown catalogue field(s) {sorted(bad)}; allowed fields: {sorted(allowed)}"
        )
    missing = set(_OBSERVATION_FIELDS) - set(row)
    if missing:
        raise ValueError(f"catalogue row is missing required field(s) {sorted(missing)}")


def _sample_value(spec, rng: np.random.Generator):
    if isinstance(spec, dict) and "dist" in spec:
        dist = spec["dist"]
        if dist == "uniform":
            value = float(rng.uniform(spec["low"], spec["high"]))
        elif dist == "loguniform":
            value = float(np.exp(rng.uniform(np.log(spec["low"]), np.log(spec["high"]))))
        elif dist == "lognormal":
            value = float(10.0 ** rng.normal(np.log10(spec["median"]), spec["scatter_dex"]))
        else:
            raise ValueError(
                f"unknown population dist {dist!r}; supported: uniform, loguniform, lognormal"
            )
        unit = spec.get("unit")
        return f"{value} {unit}" if unit is not None else value
    return spec


def _sample_population(population_cfg: dict, rng: np.random.Generator) -> list[dict]:
    n = int(population_cfg["n"])
    specs = {k: v for k, v in population_cfg.items() if k != "n"}
    return [{key: _sample_value(spec, rng) for key, spec in specs.items()} for _ in range(n)]


def _build_instrument(instrument_cfg: dict):
    kind = instrument_cfg["kind"]
    config_path = instrument_cfg["config"]
    if kind == "ifu":
        return IFU.from_config(config_path)
    if kind == "slit":
        return SlitSpectrograph.from_config(config_path)
    raise ValueError(f"instrument.kind must be 'ifu' or 'slit', got {kind!r}")


def generate_batch(
    catalogue_path: str | Path, out_dir: str | Path, *, format: str = "fits"
) -> list[Path]:
    """Generate one mock observation per catalogue galaxy, plus a manifest.

    Parameters
    ----------
    catalogue_path : str or Path
        A catalogue YAML (see module docstring; ``kino/config/
        catalogue_example.yaml`` is a documented template).
    out_dir : str or Path
        Directory for the per-galaxy data files and ``manifest.json``
        (created if missing).
    format : {"fits", "hdf5"}, optional
        Output file format, written via :mod:`kino.io.fitsio`/
        :mod:`kino.io.hdf5`.

    Returns
    -------
    list[Path]
        The written per-galaxy data file paths, in catalogue order.
    """
    if format not in _WRITERS:
        raise ValueError(f"format must be one of {sorted(_WRITERS)}, got {format!r}")
    write_cube, write_slit, ext = _WRITERS[format]

    with open(catalogue_path) as f:
        cfg = yaml.safe_load(f)

    instrument = _build_instrument(cfg["instrument"])
    g1 = float(cfg["shear"]["g1"])
    g2 = float(cfg["shear"]["g2"])
    seed = int(cfg["seed"])

    galaxies_cfg = cfg.get("galaxies")
    population_cfg = cfg.get("population")
    if (galaxies_cfg is None) == (population_cfg is None):
        raise ValueError("catalogue must set exactly one of 'galaxies' or 'population'")

    n = len(galaxies_cfg) if galaxies_cfg is not None else int(population_cfg["n"])
    children = np.random.SeedSequence(seed).spawn(n + 1)
    population_rng = np.random.default_rng(children[0])
    noise_seeds = children[1:]

    rows = (
        [dict(row) for row in galaxies_cfg]
        if galaxies_cfg is not None
        else _sample_population(population_cfg, population_rng)
    )

    allowed = _allowed_fields()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    width = max(4, len(str(n - 1)))
    paths: list[Path] = []
    manifest: list[dict] = []
    for i, (row, child_seed) in enumerate(zip(rows, noise_seeds, strict=True)):
        _validate_row(row, allowed)
        galaxy_kwargs = {k: v for k, v in row.items() if k not in _OBSERVATION_FIELDS}
        galaxy = DiscGalaxy(**galaxy_kwargs)
        line_flux = parse_quantity(row["line_flux"])
        line_wavelength = parse_quantity(row["line_wavelength"])
        rng = np.random.default_rng(child_seed)
        int_seed = int(child_seed.generate_state(1)[0] & 0x7FFFFFFF)

        result = simulate_observation(
            galaxy, g1, g2, instrument,
            line_flux=line_flux, line_wavelength=line_wavelength, rng=rng,
            meta={"catalogue": str(catalogue_path), "seed": int_seed},
        )  # fmt: skip

        path = out_dir / f"galaxy_{i:0{width}d}.{ext}"
        if isinstance(result, SlitObservation):
            write_slit(result, path)
        else:
            write_cube(result, path)
        paths.append(path)

        manifest.append(
            {
                "index": i,
                "file": path.name,
                "seed": int_seed,
                "g1": g1,
                "g2": g2,
                "line_flux": str(line_flux),
                "line_wavelength": str(line_wavelength),
                **dataclasses.asdict(galaxy),
            }
        )

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return paths
