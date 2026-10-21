# Batch mock generation

A single mock (`docs/quickstart.md`) is a galaxy, a shear, and an
instrument. A **catalogue** generalizes this to many galaxies at once,
sharing one instrument and **one global reduced shear** -- KINO treats
shear this way everywhere (every forecast/notebook assumes a single
`(g1, g2)` per target); a spatially-varying shear field is out of scope.

## A minimal catalogue

```yaml
instrument:
  kind: ifu                        # "ifu" or "slit"
  config: kino/config/sifs.yaml

shear:
  g1: 0.03
  g2: -0.01

seed: 42

galaxies:
  - r_e: 0.5
    inc: 60
    pa: 30
    v_flat: 180
    z: 0.10
    sigma0: 30
    line_flux: "1.0e-15 erg / (s cm2)"
    line_wavelength: "0.65628 um"
  - r_e: 0.8
    inc: 45
    pa: 10
    v_flat: 220
    z: 0.15
    sigma0: 35
    line_flux: "5.0e-15 erg / (s cm2)"
    line_wavelength: "0.65628 um"
```

```python
from kino.mocks.catalogue import generate_batch

paths = generate_batch("catalogue.yaml", "mock_batch/", format="fits")
```

or from the shell:

```bash
python examples/generate_catalogue.py --catalogue catalogue.yaml --out mock_batch/
```

This writes `mock_batch/galaxy_0000.fits`, `galaxy_0001.fits`, ... plus
`mock_batch/manifest.json` recording each galaxy's true parameters, its
output filename, and the per-galaxy noise seed used.

Every key in a galaxy row must be a `kino.galaxy.DiscGalaxy` constructor
field (`r_e`, `inc`, `pa`, `v_flat` are required; everything else defaults
as `DiscGalaxy` itself documents) plus the two observation-level extras
`line_flux`/`line_wavelength` -- both parsed the same way every instrument
config parses a physical value (`"<value> <unit>"`). An unknown key raises
`ValueError` naming the allowed set, so a typo is caught immediately
rather than silently ignored.

## Sampling a population instead of listing galaxies

Replace `galaxies:` with `population:` to draw `n` galaxies from per-field
distributions instead of listing them:

```yaml
population:
  n: 50
  r_e: {dist: uniform, low: 0.3, high: 1.0}
  inc: {dist: uniform, low: 20, high: 80}
  pa: {dist: uniform, low: 0, high: 180}
  v_flat: {dist: lognormal, median: 180, scatter_dex: 0.1}
  z: {dist: uniform, low: 0.05, high: 0.5}
  sigma0: {dist: uniform, low: 20, high: 40}
  line_flux: {dist: loguniform, low: 1.0e-16, high: 1.0e-14, unit: "erg / (s cm2)"}
  line_wavelength: "0.65628 um"       # a bare scalar is fixed for every galaxy
```

Three distributions are supported -- deliberately a small, fixed set, not
a general DSL:

- `uniform`: `low`, `high`.
- `loguniform`: `low`, `high` (both must be positive).
- `lognormal`: `median`, `scatter_dex` (`log10` scatter -- the same
  convention `kino.inference.priors.tully_fisher_prior` uses).

A `DiscGalaxy` field (`r_e`, `inc`, ...) samples a plain number directly.
`line_flux`/`line_wavelength` are `astropy.units.Quantity`-valued, so a
sampled value for either needs an explicit `unit:` key (appended to the
drawn number before parsing) -- see `line_flux` above.

## Reproducibility

Exactly one of `galaxies`/`population` must be set, and the catalogue's
one top-level `seed` is spawned into `n + 1` independent
`numpy.random.SeedSequence` children: one seeds the population sampler
(unused for an explicit list), and one seeds each galaxy's noise
realisation -- the same pattern `kino.analysis.injection` uses internally.
Running `generate_batch` twice on the same catalogue file therefore
regenerates **byte-identical** output files and an identical
`manifest.json`, whether the galaxies are listed or sampled.

## Reading a batch back

Each file is a normal `kino.io.fitsio`/`kino.io.hdf5` product (pick the
format with `generate_batch(..., format="fits" | "hdf5")`), with every
galaxy/shear/line parameter recorded in its own `Datacube.meta` by
`kino.mocks.single.simulate_observation` -- the same mechanism
`examples/simulate_one_galaxy.py` + `run_inference.py` use to round-trip
truth values without a bespoke header:

```python
from kino.io import fitsio

cube = fitsio.read_datacube("mock_batch/galaxy_0000.fits")
print(cube.meta["r_e"], cube.meta["g1"], cube.meta["line_flux"])
```

`manifest.json` holds the same information for every galaxy in one file,
so a downstream script (an ensemble inference run, say) doesn't need to
open every data file just to check the truth it's recovering.
