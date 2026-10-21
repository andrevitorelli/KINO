# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/) once
`v0.1.0` is released.

## [Unreleased]

### Added

- **`kino/galaxy/protocol.py`: `GalaxyModel`**, a `typing.Protocol` formalizing the structural
  contract `kino.lensing.scene.LensedScene`/`apply_shear` already relied on implicitly
  (`surface_brightness`/`vlos`/`velocity_dispersion`/`x0`/`y0`) -- confirms the forward model is
  already model-agnostic; `DiscGalaxy` is just the only implementation today. Generalizing the
  inference/mocks layers (`kino.inference.likelihood`, `kino.mocks.catalogue`,
  `kino.analysis.injection`) to accept any conforming model instead of hardcoding `DiscGalaxy`
  is real, deliberately deferred future work (2026-10-21 user direction: not expected soon, not
  blocking the companion paper) -- see README.md's Phase 6 `model_compare.py` bullet.

### Fixed

- **`kino.io.manga` was wrong, not just unverified against real MaNGIA data** (verified
  2026-10-21 with a real TNG-project API key). MaNGIA (Sarmiento et al. 2022/2023) does not
  release a single flux `Datacube`: it ships "maps" (already-gridded, but 41 named 2-D
  ground-truth property maps -- `VEL`/`DISP`/`MASS`/etc. -- with no spectral axis at all) and
  "rss" (real flux-calibrated per-fibre spectra, with real fibre sky positions, *not*
  pre-gridded). `read_mangia_cube`/`DEFAULT_PIXEL_SCALE`/`DEFAULT_FLUX_UNIT` removed entirely
  and replaced with `read_mangia_maps`/`read_mangia_rss` matching the real structure, plus new
  `regrid_fibers_to_grid`/`regrid_rss_to_cube` (a documented, simplified Gaussian-kernel
  regridding -- not a flux-conserving reproduction of the real MaNGA DRP cubing algorithm, Law
  et al. 2016 -- that builds a genuine `Datacube` from an `rss` file). Validated empirically,
  not just unit-tested: regridding a real galaxy's per-fibre ground-truth `VEL`/`DISP` and
  comparing against that *same* galaxy's official, pre-gridded `maps` product agrees to a few
  percent for most of an 8-galaxy real sample (one informative outlier at ~30-65%, likely a
  poor match for that galaxy's finer-scale structure -- see `scripts/validate_mangia_regrid.py`
  and `notebooks/06_mangia_datacube_demo.ipynb`, both rewritten against real downloaded data).
  `kino.io.download`'s MaNGIA layer was also wrong (its `MANGIA_FIXTURES`/`fetch()` design
  assumed a per-galaxy download URL that doesn't exist): MaNGIA exposes exactly three bulk
  `tar.gz` archives (catalog ~480KB, maps ~37GB, rss ~345GB) with no narrower listing. New
  `stream_tar_member_names`/`stream_extract_tar_member(s)` stream a remote archive and extract
  only the requested member(s) in one pass without buffering it, with
  `fetch_mangia_member(s)`/`mangia_archive_url`/`list_mangia_members`/`mangia_member_name` as
  the MaNGIA-specific wrappers.

### Added

- **README.md Phase 4 (mock data generation), complete**: `kino.mocks.single.
  simulate_observation` (galaxy + shear + instrument -> mock data product,
  with provenance -- every `DiscGalaxy` field, `g1`/`g2`, `line_flux`/
  `line_wavelength` -- attached to the result's `Datacube.meta`) and
  `kino.mocks.catalogue.generate_batch` (a YAML catalogue of galaxies,
  explicit or sampled from a small set of distributions, sharing one
  instrument and **one global** `(g1, g2)` -- a spatially-varying shear
  field is explicitly out of scope). New `kino.io.fitsio`/`kino.io.hdf5`:
  a symmetric FITS/HDF5 `write_datacube`/`read_datacube`/
  `write_slit_observation`/`read_slit_observation` API, generalizing the
  ad hoc round-trip that previously lived only inside
  `examples/simulate_one_galaxy.py`/`run_inference.py`, now with
  `VARIANCE`/`MASK` extensions and JSON-encoded `meta` provenance
  (`kino.io._meta`) that neither ad hoc version carried. Reproducibility:
  one top-level `seed` spawns per-galaxy `numpy.random.SeedSequence`
  children exactly as `kino.analysis.injection._trial_seeds` already does,
  so re-running `generate_batch` on the same catalogue regenerates
  byte-identical files and manifest -- pinned by
  `tests/mocks/test_catalogue.py`. `kino/config/catalogue_example.yaml`
  (documented template, explicit-list and population variants),
  `examples/generate_catalogue.py` (CLI wrapper), and
  `docs/manual/batch-mocks.md` (worked-example manual page, wired into
  `mkdocs.yml`'s nav alongside a new `docs/api/mocks.md`). Both example
  scripts refactored to use the new API instead of their private
  `_datacube_hdu`/`_datacube_from_hdu` helpers -- `run_inference.py` now
  reads every fixed galaxy/shear/line parameter back from the mock's own
  `Datacube.meta` rather than a bespoke FITS header. 19 new tests across
  `tests/mocks/` and `tests/io/test_fitsio.py`/`test_hdf5.py`.
- **`CITATION.cff`**, ahead of the public release (version/DOI fields are
  explicit placeholders, updated when `v0.1.0` is actually tagged).
- **`examples/simulate_one_galaxy.py` and `examples/run_inference.py`**
  implemented as real, thin CLI wrappers (previously `NotImplementedError`
  stubs tied to phases that are now done). `simulate_one_galaxy.py` builds
  a `DiscGalaxy`, shears it, observes it through either backend
  (`--kind ifu|slit`), and writes a minimal FITS round-trip format (a
  `FLUX`/`SPECTRUM` + `VARIANCE` extension, plus `IMAGE` for the slit case).
  `run_inference.py` reads that back and runs real nested sampling
  (`fit_shear`) on `(g1, g2)` with every other `DiscGalaxy` field held fixed
  at the simulated truth. Both were smoke-tested end to end against real
  configs (`sifs.yaml`, `goodman.yaml`) -- which caught a real bug before
  it shipped: `run_inference.py`'s first draft omitted `inc`/`v_flat` from
  `fixed_galaxy_params`, since neither is a free parameter this CLI exposes,
  so `DiscGalaxy()` construction failed inside every likelihood call.
  Nested sampling here is genuinely slow (~0.4s per likelihood evaluation,
  thousands of evaluations even at modest `n_live`/`n_eff`) -- documented
  explicitly in both the script and `docs/manual/inference.md` rather than
  promised away.

### Fixed

- **`kino/analysis/model_compare.py`**'s module docstring said it was
  blocked on `kino.inference` and the regular-disc model -- both are done
  now. The real remaining blocker is narrower: a tidally-disturbed/
  interacting disc model (`kino.galaxy`) doesn't exist yet, and nothing in
  this module can be implemented without it. Docstring corrected to say so;
  `compare_models` itself still raises `NotImplementedError`, a deliberate
  scope choice for `v0.1.0`, not an oversight (it's the one remaining 0%
  test-coverage file in the whole package).

- **Documentation site**: `mkdocs` + `mkdocs-material` + `mkdocstrings`
  (not Sphinx -- reads the existing numpydoc-style docstrings directly via
  `griffe`). `docs/index.md`, `docs/install.md`, `docs/quickstart.md` (a
  real, tested snippet replacing README's stale "planned API" one),
  `docs/manual/` (forecasting a target, adding an instrument config,
  running inference, conventions), `docs/api/` (one page per subpackage).
  `.github/workflows/docs.yml` builds with `mkdocs build --strict` on
  every push/PR and deploys to GitHub Pages from `main`.
- **Fixed**: `forecast_ifu`/`forecast_slit` (`kino/analysis/forecast.py`)
  built their docstrings by assigning `.__doc__` as an f-string *after*
  the `def`, which `mkdocstrings` -- or any static doc tool reading the
  AST -- never sees, so both rendered with no Parameters section at all.
  Found while standing up the docs site; both now carry a normal, in-place
  docstring (duplicating the shared parameter text between the two rather
  than building it from the removed `_FORECAST_PARAMS` constant -- no
  functional change, confirmed by `pytest tests/analysis/test_forecast*.py`).
- **`apply_shear`** (`kino/lensing/scene.py`) had a one-line docstring with
  no Parameters/Returns section -- the one real gap found in an otherwise
  autodoc-ready public API. Expanded.

- **Three new real-telescope IFU configs, ahead of the public release**:
  `kino/config/keck_kcwi.yaml` (Keck II KCWI, Medium slicer, BM grating),
  `kino/config/gemini_gmos_ifu.yaml` (Gemini South GMOS, two-slit mode, R831
  grating), `kino/config/vlt_muse.yaml` (VLT UT4 MUSE, Wide Field Mode,
  cropped to a 10"x10" sub-grid of the real 60"x60" field for rendering
  speed). Same convention as `sifs.yaml`/`goodman.yaml`: every sourced
  number cites its origin in the file's own header, every unsourced one is
  an explicitly labelled `PLACEHOLDER`/fiducial (never silently assumed) --
  MUSE's throughput curve is a rare case where all three header numbers are
  real, unmodified ESO-quoted values (14%/35%/14% at 480/750/930 nm).
  MUSE is also the first config at kino.observing.skymodel's own native
  site (Paranal), so it needs no Pachon/Paranal cross-site correction.
  Each gets a pinning test (`tests/instruments/test_<name>_config.py`,
  mirroring `test_goodman_config.py`) and a real `IFU.observe()` call to
  confirm it forecasts a finite, sane cube end to end, not just that the
  YAML parses.

- **Two AO-fed near-IR IFU configs (Tier B / stretch goal, ahead of the public release)**:
  `kino/config/gemini_nifs.yaml` (Gemini North NIFS, 3"x3" field at the real 29-slice count,
  R~5290 H/K) and `kino/config/keck_osiris.yaml` (Keck II OSIRIS, 0.050" plate scale, Kn3 K-band
  field, R~3800). Same sourced/placeholder convention as every config above, with a pinning test
  each (`tests/instruments/test_gemini_nifs_config.py`, `test_keck_osiris_config.py`). Both needed
  a design decision KINO hadn't faced yet: it has no dedicated Strehl-ratio/core+halo AO PSF
  model, only a single `seeing_fwhm` (plus optional, currently-unused Zernike aberration terms).
  Resolved by plugging the instrument's own quoted (NIFS: Altair's ~0.1" full AO correction) or
  computed-diffraction-limit (OSIRIS: 1.22*lambda/D at 2.2um/9.96m = 0.056") AO-corrected FWHM
  directly into `seeing_fwhm` as an effective single-Gaussian-equivalent width -- documented in
  both configs' headers and in `docs/manual/adding-an-instrument.md` as an approximation that
  understates halo flux and makes AO forecasts optimistic until a real AO PSF model exists.
  OSIRIS's throughput curve deliberately does *not* reuse the one real OSIRIS efficiency number
  found (78.0+/-1.6% grating-only efficiency at 1.31um) as if it were end-to-end system
  throughput; the config uses a lower, separately-labelled fiducial instead.

- **`kino/config/goodman_red.yaml`, `kino/config/sifs_700r.yaml`**: redder-wavelength variants
  of `goodman.yaml`/`sifs.yaml`, needed to reach Halpha above z ~ 0.11-0.13, where neither
  instrument's real measured throughput extends (Goodman's EG 21 curve stops at 685nm; SIFS's
  700M curve stops at 730nm). Both extend their real measured curve with a clearly-labelled
  **fiducial** continuation (flat at the last measured value -- conservative, not optimistic),
  checked (2026-10-09) against NOIRLab's current pages for a real measurement first and finding
  none yet. Kept as separate configs so the real, measured, test-pinned `goodman.yaml`/`sifs.yaml`
  stay untouched. `notebooks/12_redder_wavelength_configs.ipynb` validates both against a real
  SDSS Halpha spectrum at z~0.15 (found by a blind query, unrelated to any candidate list).

- **Slit/grism shear likelihood and forecast**: `kino.inference.likelihood.SlitCubeLikelihood`, the
  `SlitSpectrograph` analogue of `IFUCubeLikelihood` (fits `SlitObservation.spectrum` voxel by
  voxel; slit losses fall out of the same rendering the data went through). `shear_fisher`'s cube
  estimator now duck-types over an `IFU` or a `SlitSpectrograph`, and `kino.analysis.forecast`
  gained `forecast_slit`/`narrowed_slit` alongside a new `slit_sky_electron_rate`
  (`kino.observing.skymodel`, per along-slit pixel: `pixel_area = pixel_scale * slit_width`).
  `forecast_ifu`/`forecast_slit` now share one internal implementation; the shared return type was
  renamed `IFUForecast` -> `TargetForecast`. `SlitSpectrograph` gained `baseline_electrons()`
  (mirroring `IFU`'s). Validated the same way as the IFU case: maximum-likelihood ensembles on the
  Goodman config recovered the injected shear unbiased with scatter matching the Fisher bound
  (`notebooks/data/fisher_validation.json`, `mle_slit_cube`).
- **SOAR Goodman config** (`kino/config/goodman.yaml`): real geometry, 930 l/mm dispersion/
  resolution, and red-camera read noise from NOIRLab's published pages. Its throughput was
  first a cross-instrument guess, then replaced with a **real measurement**:
  `scripts/derive_goodman_throughput.py` derives a system throughput from an actual on-sky
  NOIRLab exposure of the spectrophotometric standard EG 21 (400 l/mm grating, airmass and
  seeing read from the FITS header) combined with EG 21's published absolute flux and KINO's
  own extinction/PSF slit-loss models, cross-checked in shape and magnitude against NOIRLab's
  published blue-camera 930/1200 l/mm curve. `tests/instruments/test_goodman_config.py` pins
  the measured values.
- **`kino.analysis.fisher.shear_fisher_multi`**: generalises the cube-estimator Fisher forecast to
  several *independent exposures of the same galaxy* -- e.g. a slit revisited at a few position
  angles/offsets to cover more of its velocity field than one (possibly misaligned) slit would,
  or repeat visits on different nights. Each exposure's own voxel-level Fisher information is
  summed before the shared nuisance priors are applied once, not per exposure. Validated by
  construction (two identical half-exposures reproduce one full exposure to <1% on the shear
  sub-block) and by the standard Fisher-information monotonicity argument; not yet validated
  against real MLE ensembles, since that needs a joint multi-exposure likelihood that does not
  exist yet. Forecast-only for now: no `psf_error`/`psf_bias`, no moment estimator, and no
  `forecast_slit`/`forecast_ifu`-level (time/seeing-tiled) wrapper.
- **`notebooks/11_goodman_forecast.ipynb`**: the slit analogue of `10_sifs_forecast.ipynb`, using
  a real archival spectrum and a physically derived shear range instead of synthetic/fiducial
  ones. A real SDSS DR17 H-alpha spectrum (found via a SkyServer SQL query for a strong,
  high-S/N, southern-declination emitter) is read with `kino.io.spectrum`, its line
  independently re-measured and cross-checked against the MPA-JHU catalogue (<1% agreement),
  and aperture-corrected to a whole-galaxy flux via the Hopkins et al. (2003) photometric
  prescription (cross-checked against a Sersic-profile enclosed-light-fraction estimate,
  agreeing to ~10%). The shear sweep spans the range implied by a fiducial 1e14 Msun foreground
  cluster (Evrard et al. 2008 M-sigma relation + a singular-isothermal-sphere shear profile) at
  projected radii of 100 kpc-2 Mpc, rather than one fiducial `g` value. Finds that for this
  specific target (r_e=2.6" against Goodman's 1.0" slit) the per-exposure shear S/N is
  slit-loss-limited (statistical S/N below ~3 across the whole shear range) and that a 10%
  PSF-FWHM error biases the shear more than either the statistical error or the assumed 0.03
  intrinsic floor -- a genuine, target-specific result, not evidence about Goodman/KL in
  general. Includes SDSS spectrum FITS (`notebooks/data/sdss_j2208_2878-54465-0462.fits`) for
  reproducibility. Extended: the shear range is now picked by choosing the lens redshift that
  *maximises* the SIS shear at a configurable reference radius, for a configurable cluster mass
  and source redshift (`M200`/`Z_SOURCE`/`R_REF`), rather than a fixed lens/source pair -- with
  an inline comparison against an Abell-2261-mass (9.5e14 Msun) cluster at the same radius,
  which reaches ~4.5x higher shear. Added a Goodman-vs-SIFS comparison on the same real target
  and exposure budget (SIFS wins by ~4-5x, partly because `sifs.yaml` has no sky background
  modelled yet -- a documented, flagged caveat, not a fair like-for-like number). Added
  position-velocity/rotation-curve plots (Goodman 1D, SIFS 2D) in the style of Pranjal R. S. et
  al. (2024, arXiv:2409.08367) Figure 5, which also surfaced a real, useful finding: aligning
  the slit with the disc's own major axis (`slit_pa = gal.pa`) recovers real S/N that the rest of
  the notebook's default `slit_pa=0` throws away on top of the pure aperture loss. Also reports
  the target's whole-galaxy apparent magnitude: extinction-corrected SDSS `cModelMag` in $ugriz$
  (native AB system, $g=17.26$, $r=17.00$, $i=16.97$) plus a Johnson $V=17.11$ via the standard
  SDSS-to-Johnson transform (Lupton 2005). Added a demonstration of `shear_fisher_multi`: splitting
  the same total exposure time across two orthogonal guessed slit position angles (rather than
  betting it all on one) improves the marginal shear precision by 1.22x for this target's 30-degree
  single-guess misalignment -- real, but modest, since a 30-degree offset is far from the
  near-total loss a 90-degree one would cause.

- **Phase 0 — project setup**: git repository initialised; MIT `LICENSE`;
  GitHub Actions CI (lint + test matrix, Python 3.10-3.12); pre-commit config;
  `CONTRIBUTING.md`; issue and pull-request templates.
- **Phase 1 — intrinsic galaxy model**: `kino.galaxy.DiscGalaxy` with
  exponential/Sersic surface brightness, arctan/tanh/flat-plus-turnover
  rotation curves, line-of-sight velocity field, optional velocity-dispersion
  map, and inclination-derived ellipticity.
- **Phase 2 — lensing / shear**: `kino.lensing.shear` (reduced-shear matrix
  transform, Mobius ellipticity composition, small-shear linearisation,
  `magnification`), `kino.lensing.scene.LensedScene` / `apply_shear` applying
  shear jointly to surface brightness and velocity field, and
  `kino.lensing.moments` for flux-moment shape measurement.
- **Phase 6 (partial) — io layer**: `kino.io.base.Datacube` container and
  `kino.io.kinms` KinMS-backed generator for known-truth injection tests, per
  the validation-data plan in `TEST_DATA.md`.
- **Phase 3 — IFU instrument backend**: `kino.instruments.base`
  (`Instrument` ABC, `PSF` -- `galsim.OpticalPSF` diffraction convolved with
  `galsim.Kolmogorov` seeing, `LSF`, `Throughput`, `Detector`) and
  `kino.instruments.ifu.IFU`, which renders a `LensedScene`'s emission line
  into a `Datacube` of detected electrons: spatially oversampled PSF
  convolution, analytic (erf-based) Gaussian line binning onto the wavelength
  axis with instrumental and intrinsic broadening combined in quadrature, and
  a photon + sky + dark-current + read noise model (`rng=None` returns the
  noiseless expected-count cube with `variance` populated). Physical flux
  calibration is supplied at `observe()` time (`line_flux`), not on
  `DiscGalaxy`, and the observed (lensed) flux comes out correctly magnified
  automatically -- see `kino/instruments/ifu.py`'s module docstring and the
  "Instrument flux calibration" convention in `CLAUDE.md`. Config-loadable
  via `IFU.from_config` and a GMTIFS-like fiducial config,
  `kino/config/gmtifs.yaml`.
- `notebooks/05_ifu_instrument_demo.ipynb`: a "what works today" demo of the
  IFU backend -- loaded config, actual PSF kernel, pixelized/PSF-convolved
  cube vs. the intrinsic scene, noiseless vs. noisy realisation, and
  seeing/resolving-power sweeps.
- **Phase 3 — slit/grism instrument backend**: `kino.instruments.slit.SlitSpectrograph`,
  sharing `kino.instruments.ifu.IFU`'s line-profile and noise machinery
  (factored out into `kino.instruments.base`'s `gaussian_channel_fractions`,
  `photon_flux_to_electrons`, `apply_shot_and_read_noise`, `doppler_center`,
  `doppler_sigma`). Slit geometry (`slit_width`, `slit_pa`, `slit_offset`)
  and losses/misalignment fall directly out of rendering on a padded,
  slit-aligned grid and truncating to the true aperture -- no separate loss
  parameter. `observe()` returns a `SlitObservation`: the dispersed 2-D
  spectrum as a `(n_wave, n_spatial, 1)` `Datacube`, plus an un-truncated
  companion broadband image. Config-loadable via `SlitSpectrograph.from_config`
  and a GMACS-like fiducial config, `kino/config/gmacs.yaml` (one slit of a
  multi-object mask). `kino.instruments.config` now holds the YAML-parsing
  helpers shared by both backends' `from_config`.
- **Phase 3 / Phase 6 — moment-map extraction**: `kino.analysis.moments`
  (`compute_moment_maps`, `MomentMaps`) computes flux/Vlos/dispersion moment
  maps, with noise, from any `Datacube` (IFU, slit, KinMS, or real data) --
  background subtraction, delta-method error propagation from per-voxel
  `variance`, S/N-based masking, and optional LSF-broadening removal.
  Verified against a real `IFU.observe()` cube's known-truth Vlos/dispersion,
  and against Monte Carlo noise-realisation scatter for the error estimates.
  Its docstring also documents a real usage caveat surfaced while building
  the notebook below: the flux-weighted second moment (`dispersion`) is far
  more noise-fragile than `flux`/`vlos` over a spectral window much wider
  than the line.
- **Phase 6 (partial) — MaNGIA reader and downloader**: `kino.io.download.fetch_file`
  (generic download + sha256 verification, cached under a git-ignored `data/`
  dir) is implemented and tested against a local HTTP server. `kino.io.manga.read_mangia_cube`
  parses a MaNGIA (TNG50 mock MaNGA, Sarmiento et al. 2023) FITS cube into a
  `Datacube`. **Both are unverified against a real MaNGIA file**: MaNGIA's
  premade cubes live on the IllustrisTNG data portal behind free user
  registration and a personal API key, which this session didn't have --
  `kino.io.download.MANGIA_FIXTURES`'s entries are placeholders pending that
  access (see that module's docstring), and `read_mangia_cube` was written
  from the mock generator's documented (not inspected) HDU layout. See
  `notebooks/06_mangia_datacube_demo.ipynb` for a correctness demo against a
  synthetic stand-in cube instead -- built by rendering a known galaxy
  through `kino.instruments.ifu.IFU` and repackaging it into the same FITS
  layout, exercising the reader and `kino.analysis.moments` together.
- `notebooks/07_moment_window_width_caveat.ipynb`: a focused deep-dive on
  the window-width caveat above, on pure synthetic single-spaxel spectra
  (no dependence on `kino.instruments`/`kino.io`) -- one line at three
  window widths, a per-channel breakdown of what dominates the dispersion
  sum, and a 300-realisation Monte Carlo sweep mapping the collapse rate
  and the (much milder) degradation of `vlos` by contrast. Also refined
  the caveat itself: a too-*narrow* window is just as broken (truncation
  bias), so there is a sweet spot, not just an upper bound on window
  width -- both `kino.analysis.moments`' docstring and `CLAUDE.md` were
  updated accordingly.
- **Two mitigations for the window-width caveat**: `kino.analysis.moments.crop_to_line`
  crops a `Datacube`'s spectral axis to a window around a given centre
  (the safe path, made the easy path), and `compute_moment_maps` gained a
  `check_window` diagnostic (on by default) that stacks the cube's flux,
  estimates the fraction of channels "effectively" carrying it via a
  participation-ratio calculation, and emits a `UserWarning` when that
  fraction looks too small -- the too-wide failure mode specifically (it
  does not catch too-narrow, which has no comparably robust automatic
  signal). Calibrated against and demonstrated in
  `notebooks/07_moment_window_width_caveat.ipynb`.
- **Phase 5 (partial) — IFU shear inference**: `kino.inference.likelihood.IFUMomentLikelihood`,
  a `nautilus`-callable Gaussian log-likelihood over flux/Vlos/dispersion
  moment maps (`kino.analysis.moments`). Each trial parameter set is
  forward-modelled through the *same*, fixed `IFU.observe(..., rng=None)`
  pipeline used for the data before reduction to moments, so PSF-driven beam
  smearing cancels between model and data rather than biasing the fit; only
  spaxels the data's own S/N mask judged trustworthy are compared.
  `kino.inference.priors` wraps free parameters into a `nautilus.Prior`
  (`build_prior`), including a `tully_fisher_prior` on `v_flat` (explicitly
  a toy, uncalibrated relation -- `TOY_TF_ALPHA = 4.0`). `kino.inference.sampler.fit_shear`
  is a thin `nautilus` nested-sampling wrapper returning a `Posterior`
  (equal-weighted samples, percentile summary, `log_z`). Validated with a
  real (small, fast) nested-sampling shear injection-recovery test in
  `tests/inference/test_sampler.py`. **IFU only** -- no slit/grism
  likelihood yet.
- `notebooks/04_one_galaxy_inference.ipynb` rewritten from its earlier
  pre-Phase-3/5 aspirational scaffold to the real API: intrinsic
  `DiscGalaxy` -> `apply_shear` -> `IFU.observe` (a narrowed spectral
  window sized to the line, per the window-width caveat) -> moment maps
  -> `IFUMomentLikelihood`/`build_prior`/`fit_shear`, plus a
  `SlitSpectrograph` forward-model aside (no likelihood yet for that
  backend). Executes end to end and recovers an injected shear alongside a
  free inclination/`v_flat`, deliberately illustrating the
  inclination-velocity degeneracy (`CLAUDE.md`) rather than hiding it by
  fixing those parameters at their true values.
- **Phase 6 (partial) — injection-recovery machinery**: `kino.analysis.injection`
  (`recover_shear`, `run_injection`, `summarize_trials`). Injects a known
  shear into a `DiscGalaxy`, observes it through an `IFU` (one noisy
  realisation), fits it with `fit_shear`, and reduces an ensemble to
  multiplicative/additive bias `(m, c)`, residual `sigma_eps`, null-test mean
  bias and pulls. Reproducible from a single seed via per-trial
  `SeedSequence` children. **IFU only**; takes a `DiscGalaxy` rather than the
  `Datacube` the original stub sketched (a cube can't be re-sheared). This is
  the tooling only -- no science-scale ensemble has been run yet.
- `kino/config/sifs.yaml`: a SOAR SIFS (0.30"/fibre, 700 l/mm) config built from
  NOIRLab's published numbers -- 50x26 fibres, 0.69 A/pix, R=4200 at 5500 A as
  a constant 1.31 A FWHM, 3.9 e- read noise, 4.1 m aperture, and an overall
  system efficiency digitised by eye from the published 700M curve. **Sky and
  dark current are 0 placeholders** (no sourced values yet), the 700R setting
  has no efficiency curve so the config stops at 7300 A (Halpha at z < 0.11),
  and every unverified field is listed in the file's header. Not yet suitable
  for quoting a forecast.
- **Observing conditions (first part)**: new `kino.observing` package with a `Site`
  (SOAR, from Mamajek 2013's GPS geodetic position), `night_window` (the dark
  window of the night beginning on a calendar date), `target_track` (altitude,
  Kasten-Young airmass, parallactic angle), `moon_state` (illumination, phase
  angle, altitude, topocentric separation) and `seeing_at` (Kolmogorov scaling
  with airmass and wavelength). Invariants tested: zenith transit at
  dec = latitude, airmass symmetry about transit, parallactic angle sign flip,
  full/new moon illumination on known dates. Sky brightness, extinction, the
  `forecast` entry point and batching are not implemented yet.
- **Observing conditions (sky and extinction)**: `kino.observing.sky` (V-band
  moonlit sky brightness from Krisciunas & Schaefer 1991 on the CTIO dark-sky
  means of Krisciunas et al. 2007; equations checked against the paper's text,
  including that 21.587 mag/arcsec^2 = 79.0 nL, and against Gemini South's
  published sky-background bins) and `kino.observing.extinction` (the CTIO mean
  extinction curve from the IRAF table, 3200-10870 A). V band only: no moon
  colour, no airglow lines, no sky spectrum yet.
- **Sky spectrum model**: `kino.observing.skymodel` (`SkyLibrary`, `SkyConditions`,
  `SkySpectrum`, `sky_electron_rate`, `ifu_sky_electron_rate`) interpolates an offline
  library of ESO SkyCalc spectra (moonlight, airglow, zodiacal light, starlight) and
  converts them to per-channel detected electrons; `scripts/build_skycalc_library.py`
  builds the library (git-ignored `data/skycalc/`). Cross-checked against CTIO dark-sky
  means and Krisciunas & Schaefer (1991); interpolation within ~12% of direct SkyCalc.
  The model is for Cerro Paranal, and the solar flux must be supplied.
- `kino.observing.solar`: NOAA SWPC's predicted monthly 10.7 cm solar flux (with its
  low/high range) for 2026-09 to 2027-12, retrieved 2026-09-26, so the airglow level
  can be set for the semester being forecast. Across NOAA's 2027A range the mean airglow
  radiance changes by only ~10%.
- **Target-level shear forecast (IFU)**: `kino.analysis.fisher.shear_fisher` (a Fisher /
  Cramer-Rao bound on `(g1, g2)` through the same forward model and moment maps as
  `IFUMomentLikelihood`), `kino.analysis.forecast.forecast_ifu` (shear S/N against seeing
  through a night, with moon, airmass, sky spectrum, extinction and seeing scaling per
  time), `kino.io.spectrum` (strict-units ASCII/FITS spectrum reader and Gaussian
  line-flux measurement with a checked error), `IFU.baseline_electrons`, and
  `kino.observing.ecliptic_from_sun`. `scripts/validate_fisher_forecast.py` compares the
  forecast with real injection-recovery ensembles.
- **Cube-level likelihood and Fisher** (found by validating the moment version): `IFUCubeLikelihood`
  (voxel-level Gaussian likelihood, unbiased) and `shear_fisher(estimator="cube")`, now the
  default, with nuisance priors including Sersic n, `r_t` and the line flux, a `free=` subset,
  and `psf_error=` (shear bias from a wrong assumed PSF, by a converged fit of the mis-specified
  model). `forecast_ifu` gained `estimator`, `intrinsic_sigma` and `psf_error`.
  `kino.analysis.injection` now defaults to the cube likelihood. The moment likelihood is kept
  but documented as noise-biased: MLE ensembles measured ~1 sigma bias per galaxy at
  per-spaxel S/N ~ 20 and 2-3.6x under-estimated scatter at S/N a few
  (`scripts/validate_fisher_mle.py`, `notebooks/data/fisher_validation.json`).
- Notebooks `08_observing_conditions`, `09_sky_spectrum` and `10_sifs_forecast`.

### Fixed

- **`SlitSpectrograph.observe`'s companion broadband image** mis-broadcast a per-channel
  (array-valued) `sky_electron_rate` (only a scalar rate worked); only exercised once
  `slit_sky_electron_rate` was used to set it. Fixed by reusing `baseline_electrons()`'s correctly
  shaped `(n_wave, 1, 1)` array for both the spectrum and the image.
- **`compute_moment_maps` background estimate** (found by validating the shear forecast
  against injection-recovery): the default per-spaxel *median* background lands inside the
  line's wings when the line fills much of the spectral window, biasing `dispersion` and
  making `dispersion_err` up to 4.6x too small (a real ensemble had shear pulls of 3.3).
  New: `subtract_background` accepts a *known* baseline array; `IFU.baseline_electrons()`
  provides it; `IFUMomentLikelihood` (`known_baseline=True`, the default) and the Fisher
  forecast now use it, and the errors match Monte Carlo to ~1% at any window width. This
  changes the likelihood weighting: earlier shear-uncertainty estimates that used the
  median background were too optimistic.
- Corrected the MaNGIA citation (Sarmiento et al. 2023, A&A 673, A23,
  arXiv:2211.11790 -- not "2022") in report material and cross-checked
  against the docstrings/changelog entries that reference it.

[Unreleased]: https://github.com/andrevitorelli/KINO/compare/main...HEAD
