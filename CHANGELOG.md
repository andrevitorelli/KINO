# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/) once
`v0.1.0` is released.

## [Unreleased]

### Added

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

### Fixed

- Corrected the MaNGIA citation (Sarmiento et al. 2023, A&A 673, A23,
  arXiv:2211.11790 -- not "2022") in report material and cross-checked
  against the docstrings/changelog entries that reference it.

[Unreleased]: https://github.com/andrevitorelli/KINO/compare/main...HEAD
