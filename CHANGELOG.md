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
- **Phase 3 (partial) — IFU instrument backend**: `kino.instruments.base`
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
  new "Instrument flux calibration" convention in `CLAUDE.md`. Config-loadable
  via `IFU.from_config` and a GMTIFS-like fiducial config,
  `kino/config/gmtifs.yaml`. The slit/grism backend and moment-map extraction
  from an IFU cube are not yet implemented.

[Unreleased]: https://github.com/andrevitorelli/KINO/compare/main...HEAD
