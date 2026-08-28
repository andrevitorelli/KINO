# CLAUDE.md — working guidance for KINO

This file orients Claude Code (and any AI assistant) working in this repository.
Read it before making changes. Keep it up to date as the project evolves.

> **Status: Phase 0 done; Phases 1-3 done; Phase 6 io-layer and one
> `kino.analysis` module started early.**
> `pyproject.toml` and the `kino/` package skeleton exist and `pip install
> -e ".[dev]"` / `ruff` / `pytest` work. Repo is git-initialised with an MIT
> `LICENSE`, GitHub Actions CI (lint + test matrix), `pre-commit` config,
> `CONTRIBUTING.md`, issue/PR templates, and `CHANGELOG.md`. Still missing:
> `docs/` (deferred to Phase 8). `kino.galaxy`
> (Phase 1: `DiscGalaxy`, Sersic profile, rotation curves, Vlos, dispersion,
> ellipticity), `kino.lensing` (Phase 2: `apply_shear`/`LensedScene`, the
> shear matrix + Mobius ellipticity composition + `magnification` in
> `kino.lensing.shear`, and a flux-moment shape measurement in
> `kino.lensing.moments`), `kino.instruments`
> (Phase 3: `Instrument` base API, `PSF`/`LSF`/`Throughput`/`Detector` and the
> shared rendering helpers (`gaussian_channel_fractions`,
> `photon_flux_to_electrons`, `apply_shot_and_read_noise`) in
> `kino.instruments.base`; `kino.instruments.ifu.IFU` -- spaxel-grid
> rendering, GalSim optical+atmospheric PSF, Gaussian LSF, and a
> photon/sky/dark/read noise model; `kino.instruments.slit.SlitSpectrograph`
> -- slit geometry/losses/misalignment and a companion broadband image,
> reusing the same PSF/LSF/noise machinery; both config-loadable via
> `IFU.from_config`/`SlitSpectrograph.from_config` and
> `kino/config/gmtifs.yaml`/`kino/config/gmacs.yaml`), `kino.io`
> (`Datacube` + KinMS generator, plus a MaNGIA reader/downloader started
> early -- see below -- ahead of the rest of Phase 6 per `TEST_DATA.md`),
> and `kino.analysis.moments` (flux/Vlos/dispersion moment maps with
> delta-method error propagation from any `Datacube`, also ahead of the rest
> of Phase 6) are implemented and tested. **`kino.io.manga.read_mangia_cube`
> and `kino.io.download`'s MaNGIA fixture manifest are unverified against
> real data** -- this session had no IllustrisTNG API key (needed to browse
> the MaNGIA catalogue; free registration at
> https://www.tng-project.org/users/register/), so both were written from
> documented file structure and tested against a synthetic stand-in cube
> instead (`notebooks/06_mangia_datacube_demo.ipynb`); see both modules'
> docstrings before trusting them against a real file. Phases 4-5
> (mocks/inference) have not started, so `examples/`,
> `notebooks/04_one_galaxy_inference.ipynb`, and the rest of `kino/analysis/`
> are still placeholders that raise `NotImplementedError`.

## What KINO is

KINO simulates and tests the inference of **kinematic lensing (KL)**: recovering
weak gravitational shear from the distorted velocity field of a single disc
galaxy. It supports two observing modes on a common forward model:
**integral-field units (IFUs)** and **slit/grism spectrographs** (+ imaging).

The scientific pipeline is:

```
intrinsic disc (image + Vlos) -> apply shear (g1,g2) -> instrument backend -> mock data
                                                                                   |
                          inference (recover g1,g2 posterior)  <--------------------
```

See `README.md` for the concept and the full phased roadmap. **The roadmap is the
source of truth for what to build next** — when you finish a task, check it off
there in the same change. See `TEST_DATA.md` for the validation-data plan (the
four-rung ladder from controlled mocks to real IFU survey data to KINO's own
instrument forecasts) that Phase 6 and the `kino/io/` adapter layer follow.

## Guiding principles

1. **Correctness of the physics first.** This is a scientific-inference code. A
   subtly wrong convention (shear sign, PA direction, ellipticity definition) is
   worse than a missing feature — it produces plausible, wrong answers. When a
   convention is involved, write a test that pins it down.
2. **One forward model, many instruments.** Never bake instrument details into
   the galaxy or lensing code. Instrument-specific behaviour lives behind the
   `Instrument` API in `kino/instruments/`.
3. **Everything reproducible.** A run is defined by a YAML config + an explicit
   random seed. The same config must reproduce the same mocks and results bit-for-bit.
4. **Test the invariants, not just the values.** Prefer tests that assert
   physical/mathematical properties (symmetries, limits, round-trips,
   conservation) over hard-coded numeric snapshots.
5. **Small, reviewable changes.** One phase item per PR where possible, with its
   tests and a short note in `CHANGELOG.md`.

## Conventions (do not deviate without updating this section AND the tests)

- **Language / stack:** Python ≥ 3.10. NumPy + SciPy + Astropy core. Inference
  via `nautilus` (neural-network-boosted importance nested sampling) behind a
  common wrapper — see `nautilus-sampler` on PyPI.
- **Units:** use `astropy.units` at public API boundaries. Internally, work in a
  documented unit system: positions in **arcsec**, velocities in **km/s**,
  wavelengths in **µm** (NIR) or **Å** (optical) — state units in every docstring.
- **Coordinates:** image-plane Cartesian `(x, y)` in arcsec, origin at the galaxy
  centroid; `x` increases with RA-like (East) direction, `y` North. Document any
  deviation at the point of use.
- **Position angle (PA):** degrees, **East of North**, of the disc major axis.
- **Inclination:** `i = 0` is face-on, `i = 90°` edge-on.
- **Ellipticity & shear:** use the **reduced shear** `g = g1 + i·g2` (complex),
  paired with the **distortion/ellipticity convention `ε = (a−b)/(a+b)`**. State
  this convention in `kino/lensing/` and enforce it with round-trip tests. If you
  ever need the `χ = (a²−b²)/(a²+b²)` convention, convert explicitly and label it.
- **Line-of-sight velocity:** `Vlos = Vsys + Vrot(r)·sin(i)·cos(θ)` in the disc
  frame; positive = redshifted (receding). Keep the sign convention consistent
  between the forward model and the likelihood.
- **Cosmology:** `astropy.cosmology.Planck18` as the default; make it injectable,
  never hard-code numeric distances.
- **Randomness:** pass a `numpy.random.Generator` (seeded) explicitly; never call
  the global `np.random` state.
- **Instrument flux calibration:** `DiscGalaxy.flux` (and everything derived from
  it, e.g. a `LensedScene`'s `surface_brightness`) is in the galaxy's own
  arbitrary units, not a physical flux — Phase 1 is intentionally unitless there.
  Physical calibration happens at `Instrument.observe()` time: pass an explicit
  `line_flux` (an `astropy.units.Quantity`, e.g. `erg/s/cm**2`) and the renderer
  rescales the shape by `line_flux / galaxy.flux`. Because a `LensedScene` is
  evaluated by sampling the *unlensed* galaxy at each observed (delensed) sky
  position, the image-integrated flux comes out magnified relative to
  `galaxy.flux` automatically (see `kino.lensing.shear.magnification` and
  `kino/instruments/ifu.py`'s module docstring) — `line_flux` is the source's
  intrinsic, unlensed brightness; never re-apply magnification on top of it.
- **Instrument PSF/LSF:** the spatial PSF is `galsim.OpticalPSF` (aperture
  diffraction) convolved with `galsim.Kolmogorov` (atmospheric seeing,
  `PSF.seeing_fwhm`), evaluated once per `observe()` call at the band's central
  wavelength (**achromatic within a cube** — a documented simplification, not a
  physical claim that real PSFs don't vary with wavelength). The LSF
  (`kino.instruments.base.LSF`) is Gaussian, so instrumental and intrinsic
  (turbulent) broadening combine in quadrature rather than by explicit
  numerical convolution.
- **Instrument noise / sky background:** `IFU.observe` returns detected
  **electrons** (`Datacube.flux_unit = astropy.units.electron`), combining
  Poisson source + sky + dark-current shot noise with Gaussian read noise
  (`variance` is always populated; `rng=None` returns the noiseless
  expected-count cube instead of drawing a realisation). Sky background
  (`IFU.sky_electron_rate`) is specified **already as a detected electron rate
  per spaxel** (as from an observatory's own exposure-time calculator), not a
  sky surface brightness — it does not go through the source's
  throughput/aperture pipeline. `SlitSpectrograph.observe` follows the same
  three conventions (electrons, sky as a per-along-slit-pixel rate,
  `rng=None` -> noiseless); see the next bullet for what's specific to it.
- **Slit geometry and losses:** `SlitSpectrograph`'s slit is a rectangle on
  sky — length `n_spatial * pixel_scale` along `slit_pa` (degrees East of
  North, same sense as `DiscGalaxy.pa`), width `slit_width`, centred at
  `slit_offset` from the scene's own origin. Losses (a source wider than the
  slit, or a misaligned `slit_pa`/`slit_offset`) are **not** a separate
  parameter — they fall directly out of rendering on a grid padded by
  `margin` native pixels beyond the slit's own footprint (so PSF-blurred
  flux can still cross the boundary) and then truncating to the true
  aperture at the oversampled (fine) pixel scale. The companion broadband
  image is the same rendering *without* that truncation or the wavelength
  split — see `kino/instruments/slit.py`'s module docstring for why it
  reuses the spectrum's own `line_flux`-normalised profile rather than a
  separately-calibrated continuum (kino has no continuum-flux input yet).
- **Moment-map extraction (`kino.analysis.moments`):** generic over any
  `Datacube`, not instrument-specific. Subtracts a per-spaxel background
  (median across the spectral axis) before computing flux-weighted
  moments by default (`subtract_background=True`) — turn it off for an
  already-background-subtracted cube. Per-voxel `variance` (if present)
  propagates to `flux`/`vlos`/`dispersion` uncertainties via the standard
  delta-method (first-order) approximation for a ratio of correlated sums;
  spaxels below `snr_threshold` have `vlos`/`dispersion` (not `flux`)
  masked to NaN. `lsf_sigma` (velocity units) optionally removes
  instrumental broadening from `dispersion` in quadrature — pass it
  yourself; the function has no notion of which instrument produced the
  cube. **Keep the cube's spectral window to a few times the line's actual
  extent before calling this** — `dispersion`'s `(v - vlos)**2` weighting
  amplifies noise from channels far from the line far more than
  `flux`/`vlos`'s weighting does, and can silently collapse `dispersion` to
  0 (clipped) even at a flux/vlos S/N that looks fine. The failure isn't
  one-sided — too *narrow* a window truncates the line's own wings and
  biases `dispersion` low, which collapses just as easily; there's a sweet
  spot, not just an upper bound. See the module docstring's "window width
  matters" section and `notebooks/07_moment_window_width_caveat.ipynb`
  for a quantified sweep.
- **External-data readers (`kino.io.manga`, `kino.io.download`):** convert
  to KINO's conventions in the reader, not downstream (`TEST_DATA.md`
  Section 5) — but check each reader's own module docstring for which
  conventions that reader has actually confirmed against a real file vs.
  assumed from documentation. `read_mangia_cube` is currently the latter
  (unverified; see the status banner above) — treat its wavelength-axis,
  pixel-scale, flux-unit, and mask-polarity defaults as starting points to
  check, not facts, until tested against a real MaNGIA cube.

## Repository layout (target)

```
kino/
  galaxy/        intrinsic disc: surface brightness, rotation curve, Vlos, sigma
  lensing/       reduced-shear transform of image + velocity field
  instruments/   base Instrument API + IFU + SlitSpectrograph backends
  mocks/         assemble galaxy+shear+instrument -> mock data products
  inference/     likelihoods, priors, samplers, posterior objects
  analysis/      ensembles, shape-noise, m/c bias, forecasts, plotting
  config/        YAML configs (galaxies, instruments, runs)
  io/            Datacube container, FITS/HDF5 read-write, provenance
tests/           pytest suite mirroring kino/ layout
examples/        runnable CLI scripts (simulate a galaxy, run inference)
notebooks/       Jupyter notebooks; one-galaxy inference is the main showcase
docs/            Sphinx/mkdocs sources
data/            git-ignored cache for downloaded/curated test data (TEST_DATA.md)
```

## Commands

These are the intended commands; create the wiring in Phase 0 if missing.

```bash
# environment
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# quality gates (run before every commit)
ruff check . && ruff format --check .
pytest -q
mypy kino            # if/when typing is enabled

# run a mock + inference example (currently NotImplementedError, see below)
python examples/simulate_one_galaxy.py --config config/gmtifs.yaml
python examples/run_inference.py       --config config/gmtifs.yaml

# notebooks (needs the `notebooks` extra: pip install -e ".[notebooks]")
jupyter lab notebooks/
```

## Definition of done (per change)

- Code has docstrings with explicit units and conventions.
- New behaviour has tests, including at least one **invariant/limit** test.
- `ruff` and `pytest` pass locally and in CI.
- The relevant `README.md` roadmap checkbox is ticked and `CHANGELOG.md` updated.
- No global RNG use; no hard-coded instrument constants outside `config/`.

## Things to be careful about

- **Shear applied to velocity, not just image.** Both the surface brightness and
  the velocity field transform under shear — a common bug is shearing only the
  morphology. Cross-check with the round-trip test.
- **PSF/LSF before sampling.** Convolve with the PSF (spatial) and LSF (spectral)
  *before* binning to spaxels/pixels, not after.
- **Slit losses & alignment.** The slit backend must model finite slit width and
  misalignment; these are first-order effects on recovered kinematics.
- **Degeneracies.** Inclination–velocity and shear–ellipticity degeneracies are
  real; rely on the Tully–Fisher prior and document which parameters are fixed vs.
  free in each run.
- **Don't silently change a convention.** If a refactor touches signs, angles, or
  ellipticity definitions, update this file, the affected docstrings, and the tests
  together.

## Scientific references

The KL method and its conventions are described in the papers listed in
`README.md` (Blain 2002; Morales 2006; Huff et al. 2013; Pranjal R. S. et al.
2022; etc.). When implementing the forward model or likelihood, follow Huff et al.
(2013) and the Pranjal R. S. et al. (2022) inference framework as primary
references, and note in the code which equations/sections a routine implements.
