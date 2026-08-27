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

[Unreleased]: https://github.com/andrevitorelli/KINO/compare/main...HEAD
