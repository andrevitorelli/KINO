# CLAUDE.md — working guidance for KINO

This file orients Claude Code (and any AI assistant) working in this repository.
Read it before making changes. Keep it up to date as the project evolves.

> **Status: Phase 0 done; Phases 1-3 done; Phase 5's IFU likelihood done
> (slit/grism likelihood deferred); Phase 6 io-layer and one
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
> of Phase 6) are implemented and tested. **`kino.io.manga`/`kino.io.download`'s
> MaNGIA support is now verified against real downloaded files** (2026-10-21,
> with a real TNG-project API key) -- and the earlier version of both was
> *wrong*, not just unverified: there is no single flux `Datacube` MaNGIA
> product. MaNGIA ships as "maps" (already-gridded 2-D property maps --
> `VEL`/`DISP`/`MASS`/etc., read by `read_mangia_maps`, no spectral axis at
> all) and "rss" (real per-fibre flux spectra + real fibre sky positions,
> *not* pre-gridded, read by `read_mangia_rss`); `regrid_fibers_to_grid`/
> `regrid_rss_to_cube` build a real `Datacube` from an `rss` file via a
> documented, simplified Gaussian-kernel regridding, validated against the
> matching `maps` file's own ground-truth `VEL`/`DISP` for the same galaxy
> (not just unit-tested). `kino.io.download` gained
> `stream_tar_member_names`/`stream_extract_tar_members`(`_member` for one)
> and MaNGIA-specific wrappers (`fetch_mangia_member(s)`,
> `mangia_archive_url`, `list_mangia_members`) once it turned out MaNGIA has
> no per-galaxy download URL at all -- only three bulk `tar.gz` archives
> (`catalog` ~480 KB, `maps` ~37 GB, `rss` ~345 GB) -- so these stream a
> remote archive and extract only the requested member(s), never buffering
> the whole thing. Ten real (rss, maps) galaxy pairs are cached under the
> git-ignored `data/mangia/` for development/testing; `notebooks/
> 06_mangia_datacube_demo.ipynb` still shows the old, now-superseded
> synthetic-stand-in approach and needs rewriting against the real reader
> (not done this session). iMaNGA (the *other* TNG50 mock-MaNGA catalogue,
> a different team's product despite the similar name) is untouched.
> `kino.inference`
> (Phase 5: `IFUMomentLikelihood` -- a moment-map Gaussian likelihood that
> forward-models each trial through the *same* `IFU` pipeline as the data
> so PSF-driven beam smearing cancels rather than biasing the fit;
> `kino.inference.priors.build_prior`/`tully_fisher_prior` wrapping
> `nautilus.Prior`; `kino.inference.sampler.fit_shear`/`Posterior`, a thin
> wrapper around `nautilus` nested sampling) is implemented and tested,
> including a real (if deliberately small/fast) shear injection-recovery
> test -- **IFU only**; a slit/grism likelihood is not yet built.
> `kino.analysis.injection` (Phase 6: `recover_shear`/`run_injection`/
> `summarize_trials` -- ensemble injection-recovery giving (m, c), residual
> sigma_eps, null-test bias and pulls) exists and is unit-tested, but no
> science-scale ensemble has been run yet. Phase 4 (`kino.mocks`) is done:
> `kino.mocks.single.simulate_observation` (galaxy -> shear ->
> `IFU`/`SlitSpectrograph`, with provenance -- every `DiscGalaxy` field,
> `g1`/`g2`, `line_flux`/`line_wavelength` -- attached to the result's
> `Datacube.meta`) and `kino.mocks.catalogue.generate_batch` (a YAML
> catalogue of galaxies, explicit or sampled from a small, documented set
> of distributions, sharing one instrument and **one global** `(g1, g2)`
> -- a spatially-varying shear field is out of scope; reproducible via the
> same `SeedSequence`-spawning pattern as `kino.analysis.injection`).
> `kino.io.fitsio`/`kino.io.hdf5` give both a symmetric FITS/HDF5
> read/write API for `Datacube`/`SlitObservation`, with free-form `meta`
> (including the raw `Quantity` values already in it) JSON-encoded via
> `kino.io._meta`. `examples/simulate_one_galaxy.py`/`run_inference.py` now
> use this real API instead of their own ad hoc assembly/FITS-header
> logic; `examples/generate_catalogue.py` and
> `kino/config/catalogue_example.yaml` are new. The rest of
> `kino/analysis/` (`model_compare.py`) is still a placeholder that raises
> `NotImplementedError` -- unrelated to Phase 4, blocked on a
> tidally-disturbed disc model that doesn't exist in `kino.galaxy` yet.
> `notebooks/04_one_galaxy_inference.ipynb` has been rewritten against the
> real Phase 1-3/5 API (galaxy -> shear -> `IFU.observe` -> moment maps ->
> `IFUMomentLikelihood`/`fit_shear`) and executes end to end, recovering an
> injected shear alongside a free inclination/`v_flat` -- deliberately
> showing the shear posterior widened by the inclination-velocity
> degeneracy rather than hiding it by fixing those parameters at truth.
> A target-level forecast now exists end to end for **both the IFU and the slit** case:
> spectrum table -> line flux (`kino.io.spectrum`) -> night/moon/airmass/sky/extinction
> (`kino.observing`) -> Fisher shear precision against seeing
> (`kino.analysis.fisher`/`forecast_ifu`/`forecast_slit`; the default cube-level estimator is
> unbiased and validated for both instruments, but is a *perfect-model statistical* bound --
> the primary forecast is that idealised number, `intrinsic_sigma` an optional sensitivity
> comparison), with `kino/config/sifs.yaml` for SOAR SIFS (sky, dark current and the 700R
> efficiency are still unverified; see the file's header) and `kino/config/goodman.yaml` for
> SOAR Goodman (real geometry/detector, throughput a real EG 21 standard-star measurement --
> see the file's header for its own caveats). `SlitCubeLikelihood`
> (`kino.inference`) is the slit analogue of `IFUCubeLikelihood`; there is no slit moment-map
> likelihood (the moment one stays IFU-only) and no near-infrared. Notebooks 08-10 showcase
> the IFU side; `notebooks/11_goodman_forecast.ipynb` showcases the slit side, using a real
> SDSS DR17 H-alpha spectrum (cross-validated against the MPA-JHU catalogue to <1%) and a
> shear range derived from a fiducial 1e14 Msun foreground cluster (Evrard et al. 2008
> M-sigma relation + a singular-isothermal-sphere shear estimate) rather than one fiducial
> value -- and finds, for that specific target (r_e=2.6" against a 1.0" slit), that the
> per-exposure shear S/N is slit-loss-limited (stat S/N below ~3 across the whole range) and
> that a 10% PSF-FWHM error biases the shear more than either the statistical error or the
> assumed 0.03 floor -- a real result, not generalisable to Goodman/KL as such. It also picks
> the cluster's lens redshift by *maximising* the SIS shear at a configurable reference radius
> for a configurable mass/source redshift (`M200`/`Z_SOURCE`/`R_REF`), compares Goodman against
> SIFS on the same real target (SIFS wins ~4-5x, partly from its own unmodelled-sky caveat), and
> plots position-velocity/rotation-curve diagrams in the style of Pranjal R. S. et al. (2024,
> arXiv:2409.08367) -- which surfaced that aligning the slit with the disc's major axis recovers
> real S/N the rest of the notebook's default `slit_pa=0` throws away. It also demonstrates
> `shear_fisher_multi` (splitting the same exposure across two guessed slit position angles beats
> betting it all on one, by 1.22x for this target's 30-degree single-guess misalignment).

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
  for a quantified sweep. Two mitigations: `crop_to_line(cube, center,
  half_width)` crops a cube to near a line before you call
  `compute_moment_maps`, and `compute_moment_maps(..., check_window=True)`
  (the default) warns when the too-wide failure mode looks likely — it
  does not detect too-narrow, which has no comparably robust automatic
  signal. Pass `check_window=False` when deliberately exploring a bad
  window (e.g. in a test or sweep) rather than suppressing the warning
  globally. **Background: prefer a known baseline over the default median.**
  The default subtracts a per-spaxel *median over the window*, which is fragile:
  when the line fills more than about a third of the window the median lands
  inside the line's wings, biasing `dispersion` and adding noise the propagated
  errors omit (measured: `dispersion_err` 4.6x too small at a 20-channel window
  the line half-filled; a plain window-width diagnostic did *not* catch it).
  Pass the instrument's known sky + dark baseline as `subtract_background`
  (`IFU.baseline_electrons()`), which needs no estimate and stays calibrated to
  ~1% against Monte Carlo at any window width. `IFUMomentLikelihood` and the
  forecast do this by default (`known_baseline=True`).
- **Shear likelihoods (`kino.inference.likelihood`):** three, sharing one forward model and two
  common helpers (`_scene_from_params`, `_resolve_line_flux`). **The cube ones are the ones to
  use**: `IFUCubeLikelihood` (IFU) and its slit analogue `SlitCubeLikelihood` (fits
  `SlitObservation.spectrum` only -- the companion broadband image is a placement aid, never fit),
  a voxel-level Gaussian likelihood `-1/2 sum[(d-m)^2/v + ln v]` (variance floored at 1e-3 e-^2 so
  zero-noise test cubes don't divide by zero) with no estimator in between. On many noise
  realisations its maximum-likelihood shear was unbiased (~1e-4 IFU, ~1e-4 slit) and its scatter
  matched the Cramer-Rao bound at every per-spaxel S/N tested (5 to ~500), with inclination, PA and
  `v_flat` free as well. Both need *nonzero read noise* to be a sensible Gaussian model. Slit
  losses (a source wider than the slit, or a misaligned `slit_pa`/`slit_offset`) fall directly out
  of `SlitSpectrograph.observe()`'s own rendering -- inside the model exactly as they are inside
  the data, not a separate parameter. **`IFUMomentLikelihood`** (IFU only; there is no slit
  moment-map likelihood) compares flux/Vlos/dispersion *moment maps* and is kept for comparison:
  data moments are ratios of noisy sums while the model's are noiseless, so the estimator has a
  noise bias of ~1 sigma per galaxy at per-spaxel S/N ~ 20 that grows toward low S/N (an analytic
  second-order correction was tried and fails below S/N ~ 12; a moment-based fit of the notebook-04
  mock missed the truth by 3-5 sigma in every parameter). Do not use it for a stacked shear. All
  three forward-model each trial through the *same*, fixed instrument (`IFU`/`SlitSpectrograph`)
  `.observe(..., rng=None)` used for the data -- never compare a rendered cube against an
  analytically-evaluated (un-PSF-smeared) model, or the fit is biased by exactly the beam-smearing
  amount. For the moment version the data's moment maps are computed once (`check_window=True`,
  real `snr_threshold`) and each model evaluation reuses `check_window=False`, `snr_threshold=0`,
  compared over the spaxels the *data*'s own S/N mask judged trustworthy. `g1`/`g2`/`line_flux`
  are handled by the likelihood, not passed to `DiscGalaxy`; any other free parameter must be a
  `DiscGalaxy` field (`GALAXY_FIELDS`). An unphysical trial point (e.g. `|g| >= 1`) returns
  `-inf`. `kino.inference.priors` wraps this into a `nautilus.Prior` (`build_prior`), including an
  explicitly **toy, uncalibrated** Tully-Fisher prior (`tully_fisher_prior`, `TOY_TF_ALPHA = 4.0`)
  -- do not read numeric results from it as astrophysical predictions until it is replaced with a
  real calibration. `kino.inference.sampler.fit_shear` is a thin `nautilus` nested-sampling wrapper
  returning a `Posterior` (equal-weighted samples + `log_z`).
  Validation lives in `scripts/validate_fisher_forecast.py` (nested sampling),
  `scripts/validate_fisher_mle.py` (fast maximum-likelihood ensembles) and
  `notebooks/data/fisher_validation.json`; a pull far from 1 means the reported errors, and every
  forecast built on them, cannot be trusted.
- **Injection-recovery (`kino.analysis.injection`):** injects into a
  `DiscGalaxy`, never a `Datacube` (a cube is already an observation and
  can't be re-sheared). Import it as `kino.analysis.injection`, not from
  `kino.analysis` -- it depends on `kino.inference`, which depends on
  `kino.analysis.moments`, so re-exporting it there is an import cycle.
  Every trial gets its own child of one `SeedSequence` (noise + sampler
  seed), so a run is reproducible from one `seed` and a trial's result
  doesn't depend on ensemble size. Recovered shear is the posterior
  *median*; with a finite uniform prior and weakly-constraining data it is
  pulled toward the prior centre, so `m < 0` at low S/N is expected
  shrinkage -- check `m` against S/N before reading it as a forward-model
  bug. `sigma_eps` is the per-galaxy scatter of `g_rec - g_true`; the pull
  std should be ~1 if the posterior widths are honest, and a pull far above
  1 means the reported errors are too small (sampler settings too loose, or
  the likelihood's delta-method errors are underestimated) -- do not quote
  a forecast until it is ~1 at production `n_live`/`n_eff`.
- **Observing conditions (`kino.observing`):** sits between a target and an
  instrument backend and knows nothing about instruments. A night is *the
  evening beginning on a local calendar date*, found from the site's mean
  solar noon (no timezone handling), so a "night" straddles a UT date change.
  Sites are only predefined once their coordinates are checked against a
  primary source (`SOAR`: Mamajek 2013, arXiv:1210.1616, Table 6; the
  height is *ellipsoidal*). Hour angle / parallactic angle use the target's
  RA **of date** (`TETE`), not J2000 -- the difference is ~0.3 deg of hour
  angle after 25 years of precession, which an earlier version got wrong.
  Moon phase uses the geocentric Sun and Moon; altitude and separation use
  the topocentric Moon (lunar parallax is ~1 deg). Airmass is Kasten & Young
  (1989), finite at the horizon and `inf` below it. `seeing_at` is the plain
  Kolmogorov scaling (FWHM ~ X^0.6 lambda^-0.2) -- an approximation for
  shaping a curve, not a per-image prediction. **Sky:** two layers. `kino.observing.sky` is the V-band-only
  Krisciunas & Schaefer (1991) moonlight model on CTIO dark-sky means
  (Krisciunas et al. 2007, Table 6) -- kept as an *independent cross-check*.
  The working model is `kino.observing.skymodel`: a **local library of ESO
  SkyCalc spectra** (Noll et al. 2012; Jones et al. 2013) for moonlight, airglow
  lines + continuum, zodiacal light and scattered starlight, built once by
  `scripts/build_skycalc_library.py` into git-ignored `data/skycalc/` (needs
  network and `skycalc-cli`; ~7 min) and interpolated offline, so batch runs
  stay reproducible -- never call the SkyCalc web service from a run. It was
  validated: dark zenith V = 21.9 vs the CTIO mean 21.79 +/- 0.22; moon dV
  within 0.1-0.4 mag of K&S91 at four geometries; interpolation within ~12% of
  direct SkyCalc calls at eight off-grid geometries. Known limits: the model is
  for **Paranal**, not Pachon (light pollution from La Serena and aerosols are
  absent); airglow is a whole-night, all-season average and scales with the
  10.7 cm solar flux, which **you must pass** (SkyCalc's own almanac has no
  value after 2025-03); `kino.observing.solar` tabulates NOAA SWPC's *predicted*
  monthly F10.7 through 2027-12 (retrieved 2026-09-26 -- refresh it, and prefer
  observed values once a month has passed). Across NOAA's stated range for
  2027A (96-134 s.f.u.) the mean airglow radiance changes by only ~10%, but a
  flux far off (say 200 vs 100) moves the continuum by tens of percent; moonlight is capped at airmass 2 and tapers to zero between a Moon
  zenith distance of 80 deg and the horizon; wavelengths are **vacuum**, so
  compare with air wavelengths only after converting; our ecliptic longitude
  (relative to the Sun) differs from SkyCalc's almanac by ~4 deg for one
  tested target (unresolved, matters little away from the Sun). Wire it into an
  instrument with `ifu_sky_electron_rate`, which gives the per-channel
  `sky_electron_rate` on the same collecting area and throughput as the source.
  **Extinction:** `kino.observing.extinction` is the IRAF/CTIO mean curve
  (Stone & Baldwin 1983; Baldwin & Stone 1984), measured at Tololo and assumed at
  Pachon, with no per-night aerosol variation and no cloud.
- **Forecasts (`kino.analysis.fisher`, `kino.analysis.forecast`, `kino.io.spectrum`):**
  `shear_fisher` is a Fisher/Cramer-Rao bound on `(g1, g2)` through the same forward model as the
  likelihoods, and takes **either an `IFU` or a `SlitSpectrograph`** (duck-typed: `estimator="cube"`
  works with both, dispatching to `IFUCubeLikelihood`'s or `SlitCubeLikelihood`'s bound; a
  `SlitSpectrograph.observe()` result is unwrapped to its `.spectrum` internally).
  **Default estimator `"cube"`**: every voxel, so flux image + kinematics; the line flux is a
  nuisance with a fractional prior; central differences; a few seconds (IFU) to ~1 minute per point
  for Goodman's finer, longer slit -- narrow the window first, see below. Validated against
  maximum-likelihood fits on many noise realisations, for both instruments (scatter / Fisher
  0.8-1.3 with inclination, PA and `v_flat` free; bias consistent with zero). `"moments"` is the
  bound of the moment likelihood, **IFU only** (raises `AttributeError` for a slit -- there is no
  slit moment likelihood to bound). It returns
  `sigma_fixed` (only the shear free) and `sigma_marginal` (nuisance parameters free under
  `DEFAULT_CUBE_PRIORS`: inc +/-5 deg, pa +/-2 deg, r_e +/-5%, Sersic n +/-0.3, v_flat +/-25% (toy
  TF), r_t +/-20%, centroid +/-0.05", sigma0 +/-20 km/s, line flux +/-5% -- *assumptions*, override
  them). **What it is not:** a *statistical* bound for a perfect model. Because the data pin the
  nuisance parameters far below any prior (inclination to ~0.1 deg), the marginal shear precision
  comes out 1e-3-1e-2 per galaxy for bright targets -- **not achievable on real galaxies**, whose
  non-circular motions, thickness, warps and PSF/model errors set a floor the bound ignores.
  Real-galaxy calibration of that floor is explicitly **out of scope for now** (it belongs to a
  future project applying this code, not to validating the pipeline itself -- 2026-09-27 user
  direction): the primary forecast KINO reports is the **idealised, statistical** S/N
  (`shear_snr_stat`), clearly labelled as such rather than blocked on a floor nobody has measured
  yet. `intrinsic_sigma=...` remains available for a *sensitivity* comparison (no default) and
  `psf_error=` adds the shear bias from a wrong assumed PSF FWHM (the converged Levenberg-Marquardt
  fit of the mis-specified model, not the linear formula, which is 2-3x off at high S/N).
  `forecast_ifu`/`forecast_slit` tile this over `(time, seeing)` through a night: airmass, the sky
  spectrum (`kino.observing.skymodel`; `ifu_sky_electron_rate` per spaxel, `slit_sky_electron_rate`
  per along-slit pixel with `pixel_area = pixel_scale * slit_width`, matching the sky entering
  through the slit's full width), CTIO extinction on the line flux (off with
  `include_extinction=False` if the throughput already includes the atmosphere), and the quoted
  500 nm/zenith seeing converted to the line's wavelength and airmass; the spectral window is
  cropped to +/-`window_kms` (default 500) at the instrument's native dispersion
  (`narrowed_ifu`/`narrowed_slit`) -- for Goodman's default (wide, MOS-derived) window this alone
  is a ~7x speedup. Both wrap one shared, instrument-agnostic `_forecast` implementation and return
  the same `TargetForecast`. `forecast_slit` rejects `estimator="moments"` outright (no slit moment
  likelihood to wrap). Limits: optical only (sky library to 1 micron, extinction to 1.087 micron),
  per galaxy (stack by sqrt(N) only for the statistical part), and the Gaussian cube model needs
  nonzero read noise. `kino.analysis.fisher.shear_fisher_multi` generalises the cube estimator to
  **several independent exposures of the same galaxy** (e.g. a slit revisited at a few position
  angles/offsets to cover more of its velocity field than one misaligned slit would, or repeat
  visits): each exposure's own voxel-level Fisher information is summed before the shared
  nuisance priors are applied once (not per exposure). Validated by construction (two identical
  half-exposures reproduce one full exposure to <1% on the shear sub-block) and by the standard
  Fisher-information monotonicity argument (more independent information can only shrink the
  marginal covariance, never grow it) -- not yet validated against real MLE ensembles the way the
  single-exposure estimators are, since that needs a joint multi-exposure likelihood that does not
  exist yet. Does not support `psf_error`/`psf_bias` or the moment estimator, and there is no
  `forecast_slit`/`forecast_ifu`-level (time/seeing-tiled) wrapper for it yet -- it is Fisher-level
  only, called directly with a list of already-configured instruments.
  `kino.io.spectrum.read_spectrum` is strict about units -- a flux column with no unit raises
  rather than assuming SDSS/DESI's 1e-17 -- and treats wavelengths as vacuum;
  `measure_line_flux` returns what the *aperture* collected (pass `aperture_correction` for the
  whole galaxy) with an error whose pull was checked to be ~1. Candidate target lists are private:
  never commit one or use a real target name in code, tests, notebooks or reports.
- **Instrument configs from partial data (`kino/config/sifs.yaml`, `kino/config/goodman.yaml`):**
  unlike the fully-fiducial Phase 7 concept configs (`gmtifs.yaml`, `gmacs.yaml`), these use
  *real* SOAR geometry/detector numbers where published, and mark every unsourced field in the
  file's own header rather than silently defaulting. `goodman.yaml`'s throughput curve is a
  worked example of that: NOIRLab's Goodman red-camera efficiency is unpublished, so
  `scripts/derive_goodman_throughput.py` derives a **real, measured** curve instead of a
  cross-instrument guess (an earlier version of this file used SIFS's curve scaled by a stated
  efficiency ratio -- see git history): NOIRLab publishes an on-sky exposure (raw frame, 1-D
  counts spectrum, and the full FITS header) of the spectrophotometric standard EG 21, which
  combined with EG 21's independently published absolute flux and KINO's own extinction/PSF
  slit-loss models gives a system throughput. Caveat: it is the 400 l/mm grating, not the 930
  l/mm `goodman.yaml` uses for resolution, and one star on one night (NOIRLab's own page notes
  ~50% night-to-night scatter); a published 930/1200 l/mm curve exists only for the *blue*
  camera and is used as a shape/magnitude cross-check, not a substitute. See the script's and
  the config's own docstrings for the full derivation and caveats.
  `tests/instruments/test_goodman_config.py` pins the exact values so a better measurement
  replacing them is a deliberate, visible change. Before adding another such config, look for
  the real number first; prefer a *measured proxy* (as here) over a cross-instrument guess, and
  fall back to a documented fiducial only when neither exists. `kino/config/goodman_red.yaml`/
  `sifs_700r.yaml` are that fallback in practice: neither instrument has a measured throughput
  redward of ~685-730nm (needed for Halpha above z ~ 0.11-0.13), so each extends its own real
  curve with a flat, explicitly fiducial continuation, kept in a *separate* config file so the
  real, measured, test-pinned originals are untouched.
- **Mocks and mock I/O (`kino.mocks`, `kino.io.fitsio`, `kino.io.hdf5`):**
  `kino.mocks.single.simulate_observation(galaxy, g1, g2, instrument, ...)`
  is `apply_shear` then `instrument.observe()` -- its only value-add over
  calling those two directly is writing **provenance** into the result's
  `Datacube.meta` (every `DiscGalaxy` field via `dataclasses.asdict`, plus
  `g1`/`g2`/`line_flux`/`line_wavelength`); for a `SlitObservation` result
  this goes on `result.spectrum.meta`, the only `Datacube` it carries.
  `kino.mocks.catalogue.generate_batch` reads a YAML catalogue (one
  instrument, **one global `(g1, g2)` applied to every galaxy** -- every
  existing forecast/notebook already treats shear this way; a
  spatially-varying shear field is explicitly out of scope) and either an
  explicit `galaxies:` list or a `population:` sampled from a small, fixed
  set of distributions (`uniform`, `loguniform`, `lognormal` with
  `median`/`scatter_dex` -- the same `log10`-scatter convention as
  `tully_fisher_prior`). A catalogue row may set any
  `dataclasses.fields(DiscGalaxy)` name plus `line_flux`/`line_wavelength`
  (parsed via `kino.instruments.config.parse_quantity`, so both take the
  same `"<value> <unit>"` string form every instrument config already
  uses); an unknown key raises `ValueError` naming the allowed set.
  `kino.mocks` deliberately does **not** import `kino.inference` to get
  this field list (that dependency direction doesn't otherwise exist in
  the codebase) -- `tests/mocks/test_catalogue.py` pins that its
  independently-built set still matches
  `kino.inference.likelihood.GALAXY_FIELDS`. Reproducibility: one
  top-level `seed` spawns `n + 1` `numpy.random.SeedSequence` children
  (child 0 for the population sampler, one per galaxy for its noise) --
  the exact pattern `kino.analysis.injection._trial_seeds` uses -- so
  re-running `generate_batch` on the same catalogue regenerates
  byte-identical files and an identical `manifest.json`.
  `kino.io.fitsio`/`kino.io.hdf5` give a symmetric `write_datacube`/
  `read_datacube`/`write_slit_observation`/`read_slit_observation` API on
  each backend; `Datacube.meta` is free-form and already holds raw
  `astropy.units.Quantity` values (`IFU.observe`/`SlitSpectrograph.observe`
  put `line_wavelength_rest`/`line_flux_intrinsic`/etc. there themselves),
  so both back ends JSON-encode it through `kino.io._meta.dumps_meta`
  (anything not natively JSON-serializable is stringified) rather than
  assuming plain scalars -- reading it back is a **text** round-trip
  (`"0.65628 um"`), not a type-preserving one. FITS stores this in one
  `METAJSON` header card; astropy's `fits.Header` auto-folds a long string
  across `CONTINUE` cards, so no manual chunking is needed (verified for a
  representative galaxy-provenance blob). `examples/simulate_one_galaxy.py`/
  `run_inference.py` use this real API end to end now, including reading
  every fixed galaxy/shear/line parameter back from `Datacube.meta` rather
  than a bespoke FITS header.
- **External-data readers (`kino.io.manga`, `kino.io.download`):** convert
  to KINO's conventions in the reader, not downstream (`TEST_DATA.md`
  Section 5) — but check each reader's own module docstring for which
  conventions that reader has actually confirmed against a real file vs.
  assumed from documentation. `kino.io.manga` is now verified against real
  MaNGIA files (see the status banner above) for the structural facts
  (extension names, `TYPE<i>` map keys, wavelength grid, fibre positions);
  the fibre East-of-North orientation and the exact flux-unit
  normalisation (`/Angstrom` implicit, not spelled out in the file's own
  header) remain unconfirmed assumptions — check before trusting either.

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

# run a mock + inference example (see examples/README.md for the full CLI)
python examples/simulate_one_galaxy.py --config kino/config/sifs.yaml \
    --line-flux 1e-15 --line-wavelength 0.65628 --g1 0.03 --g2 -0.01 --out mock.fits
python examples/run_inference.py --data mock.fits --config kino/config/sifs.yaml

# notebooks (needs the `notebooks` extra: pip install -e ".[notebooks]")
jupyter lab notebooks/

# sky library for kino.observing.skymodel / notebooks 09-10 (network, ~7 min, needs the `sky` extra)
pip install -e ".[sky]" && python scripts/build_skycalc_library.py
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
