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

[Unreleased]: https://github.com/andrevitorelli/KINO/compare/main...HEAD
