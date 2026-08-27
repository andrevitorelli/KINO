# KINO

**KI**nematic lensing i**N**ference and mock-**O**bservations

*(kino — from the Greek* kinema*, "movement" — a nod to the galaxy kinematics at the
heart of the method.)*

KINO is a Python toolkit to **simulate** and **test the inference of kinematic
lensing (KL)** — the measurement of weak gravitational shear from the distorted
velocity field of an individual disc galaxy. It is built to work across both
observing modes used in the KL literature:

- **Integral-field units (IFUs)** — resolved 2-D velocity maps / datacubes
  (e.g. GMTIFS-like AO-fed spectrographs).
- **Slit / grism spectrographs** — 1-D or partially-resolved spectra combined
  with imaging (e.g. GMACS-like multi-object spectrographs, Roman/DESI-style
  slit+grism setups).

The goal is a single, well-tested forward model of a sheared disc galaxy, a set
of interchangeable **instrument backends** that turn that model into realistic
mock observations, and an **inference engine** that recovers the shear and
characterises its uncertainty. KINO is intended to become the analysis
foundation for a KL science case on the Giant Magellan Telescope (GMT), but it is
instrument-agnostic by design.

---

## 1. The idea

Weak lensing is limited by **shape noise**: the intrinsic scatter of galaxy
ellipticities (σ_ε ≈ 0.26) dwarfs the ~1% shear signal, so conventional cosmic
shear must average over huge numbers of galaxies. Kinematic lensing removes this
limitation. For a rotationally supported disc, the projected velocity field plus
the Tully–Fisher relation independently fix the galaxy's intrinsic inclination
and orientation. Lensing distorts the *image* but not the *observed frequency* of
light, so the mismatch between the shape predicted from the kinematics and the
observed shape yields the shear for a **single** galaxy, driving the effective
shape noise down to σ_ε ≈ 0.02–0.04.

KINO makes this concrete and testable end-to-end:

```
 intrinsic disc  ──►  apply shear  ──►  instrument model  ──►  mock data
 (image + Vlos)       (g1, g2)          (IFU or slit)          (cube / spectra + image)
        ▲                                                              │
        └──────────────  inference: recover (g1, g2) posterior  ◄──────┘
```

By simulating galaxies with a **known** input shear and running the inference, we
can measure the per-galaxy shear precision, quantify residual shape noise,
detect multiplicative/additive biases, and forecast performance for a given
instrument and exposure time — for both IFUs and slit spectrographs, on the same
footing.

---

## 2. Scope (and non-scope)

**In scope**
- Forward modelling of a single sheared disc galaxy (image + line-of-sight
  velocity field, optionally velocity dispersion).
- Interchangeable instrument backends: IFU datacube and slit/grism spectrograph
  (+ companion imaging).
- Bayesian shear inference per galaxy, with pluggable samplers.
- Ensemble studies: shape-noise characterisation, bias/calibration (m, c),
  null tests, and instrument forecasts.

**Out of scope (for now)**
- Real-data reduction pipelines (KINO consumes/produces mock data; a thin I/O
  layer can later read real cubes).
- Cluster mass-profile modelling and cosmological likelihoods — KINO stops at the
  per-galaxy shear estimate; downstream cosmology lives in a separate package.
- Full radiative transfer / spectral synthesis — emission lines are modelled
  parametrically.

---

## 3. Architecture (target)

```
kino/
├── galaxy/          # intrinsic disc: surface brightness, rotation curve, Vlos field
├── lensing/         # shear transformation of image-plane coords, image + velocity
├── instruments/     # backends: base API + IFU + slit/grism; PSF, LSF, sampling, noise
├── mocks/           # assemble galaxy+shear+instrument -> mock data products
├── inference/       # likelihoods, priors, nautilus sampler wrapper, posteriors
├── analysis/        # ensembles, shape-noise, m/c bias, forecasts, plots
├── config/          # YAML configs for galaxies, instruments, runs
└── io/              # read/write mock data (FITS), results, provenance

examples/            # runnable CLI scripts (simulate a galaxy, run inference)
notebooks/           # Jupyter notebooks; one-galaxy inference is the main showcase
```

Key design choices:
- **One forward model, many instruments.** The disc + shear model is
  instrument-independent; each backend implements a common `observe(scene)` API.
- **Config-driven & reproducible.** Runs are defined by YAML + a fixed random
  seed; every output carries its provenance.
- **Physical units throughout** via `astropy.units`; explicit coordinate and
  ellipticity conventions (see `CLAUDE.md`).

---

## 4. Task list / roadmap

Grouped into milestones. Check items off as they land. Each phase should ship with
tests and a runnable example before the next begins.

### Phase 0 — Project setup
- [x] Initialise repo, license (MIT), `pyproject.toml`, package skeleton `kino/`.
- [x] Dev tooling: `ruff` (lint/format), `mypy` (optional), `pytest`, `pre-commit`.
- [x] GitHub Actions CI: lint + test matrix (Python 3.10–3.12).
- [x] Core dependencies pinned: `numpy`, `scipy`, `astropy`, `nautilus-sampler`,
      `matplotlib`, `pyyaml`, `h5py`.
- [x] `CONTRIBUTING.md`, issue/PR templates, `CHANGELOG.md`.

### Phase 1 — Intrinsic galaxy model
- [x] Surface-brightness model (exponential / Sérsic disc) with inclination `i`,
      position angle `PA`, effective radius `r_e`, centroid, total flux.
      `kino.galaxy.DiscGalaxy` + `kino.galaxy.profiles`.
- [x] Rotation curve model(s): arctan, `tanh`, and a flat+turnover form.
      `kino.galaxy.rotation_curve`.
- [x] Line-of-sight velocity field `Vlos(x, y)` from rotation curve + geometry,
      including systemic velocity. `DiscGalaxy.vlos`; cross-checked against an
      independently generated KinMS cube (`tests/galaxy/test_kinms_consistency.py`).
- [x] Optional intrinsic velocity dispersion σ map (constant + gradient).
      `DiscGalaxy.velocity_dispersion`.
- [x] Intrinsic ellipticity from inclination. `DiscGalaxy.ellipticity`.
- [ ] Consistency check vs. Tully–Fisher — deferred to Phase 5's priors module,
      which needs a calibrated TF relation; no such calibration exists yet.
- [x] Unit tests: face-on ⇒ zero projected rotation; edge-on limits; symmetry.
      `tests/galaxy/`.

### Phase 2 — Lensing / shear application
- [x] Implement the reduced-shear coordinate transformation (g1, g2) on the
      image plane; document the ellipticity/shear convention precisely.
      `kino.lensing.shear` (matrix + Mobius ellipticity composition, full
      East-of-North-`pa`-to-standard-angle derivation in the module docstring).
- [x] Apply shear jointly to the surface brightness **and** the velocity field.
      `kino.lensing.scene.LensedScene` / `kino.lensing.apply_shear`.
- [x] Verify: zero shear ⇒ identity; known analytic ellipticity change under
      pure g1 / pure g2; round-trip (shear then unshear) recovers the input.
      `tests/lensing/test_shear.py`, `tests/lensing/test_scene.py`; also
      cross-checked against actual rendered-image moments
      (`kino.lensing.moments`) and against emergent physics not in the
      roadmap text but directly useful for KL: photometric/kinematic
      position-angle misalignment vanishes exactly when shear is aligned
      with the disc's own axes (`test_misalignment_vanishes_when_shear_aligned_with_disc_axes`).
- [x] Small-shear linearisation cross-check against the full transform.
      `test_small_shear_linearisation` — note the correct first-order
      expansion is `eps_int + g - eps_int**2 * conj(g)`, not the naive
      `eps_int + g` (see `kino.lensing.shear`'s docstring: the responsivity
      term is first order in `g` too whenever `eps_int != 0`).
- [x] Flux magnification from pure shear (not in the original bullet list,
      but needed once the misalignment blind spot — shear aligned with the
      disc's own axes — came up): `kino.lensing.shear.magnification`,
      `mu = 1/(1-|g|**2)`, nonzero even with no convergence. Verified against
      `np.linalg.det` of the shear matrix and against the actual rendered/
      integrated flux ratio of a lensed vs. unlensed image
      (`notebooks/03_tully_fisher_shear_signal.ipynb`).

### Phase 3 — Instrument backends
- [x] Define the `Instrument` base API: `observe(scene) -> MockData`, plus PSF,
      LSF, spatial sampling, throughput, and noise hooks. `kino.instruments.base`:
      `PSF` (`galsim.OpticalPSF` diffraction convolved with `galsim.Kolmogorov`
      seeing), `LSF`, `Throughput`, `Detector`, and the `Instrument` ABC.
- [x] **IFU backend**: spaxel grid, AO/seeing PSF convolution, spectral LSF,
      datacube assembly. `kino.instruments.ifu.IFU.observe` -> `Datacube`; see
      that module's docstring for the flux-calibration convention (physical
      `line_flux` supplied at `observe()` time, not on `DiscGalaxy`) and the
      PSF/LSF-before-sampling rendering pipeline.
- [x] Extract moment maps (flux, Vlos, σ) with noise from any `Datacube` -- a
      measurement *on* a cube rather than part of the forward model, so it
      lives in `kino.analysis.moments` (Phase 6) rather than in
      `kino.instruments`: flux-weighted spectral moments, delta-method error
      propagation from per-voxel `variance`, S/N-based masking, and optional
      LSF-broadening removal in quadrature. Works on an IFU or slit-spectrum
      `Datacube`, a KinMS cube, or real survey data alike.
- [x] **Slit / grism backend**: slit mask geometry & placement, dispersion to a
      2-D spectrum, companion broadband image; handle slit losses & misalignment.
      `kino.instruments.slit.SlitSpectrograph.observe` -> `SlitObservation`
      (a `(n_wave, n_spatial, 1)` `Datacube` spectrum plus an un-truncated
      broadband image); slit losses and misalignment fall directly out of
      rendering on a padded, slit-aligned grid and truncating to the true
      aperture -- see that module's docstring.
- [x] Noise models: photon + read noise + sky; configurable exposure time and
      throughput curves. `IFU.observe`/`SlitSpectrograph.observe` combine
      Poisson source/sky/dark shot noise with Gaussian read noise (shared via
      `kino.instruments.base.apply_shot_and_read_noise`); `rng=None` returns
      the noiseless expected-count cube with `variance` still populated (for
      S/N without drawing a realisation).
- [x] Instrument config files for a GMTIFS-like (`kino/config/gmtifs.yaml`) and
      a GMACS-like (`kino/config/gmacs.yaml`, one slit of a multi-object mask)
      setup, loaded via `IFU.from_config` / `SlitSpectrograph.from_config`.
- [x] Tests: flux conservation, PSF/LSF broadening, S/N vs. exposure scaling,
      slit losses (width/offset/misalignment), moment-map recovery against
      known truth, and delta-method error propagation against Monte Carlo
      scatter.
      `tests/instruments/`.

### Phase 4 — Mock data generation
- [ ] `mocks` layer assembling galaxy + shear + instrument into data products.
- [ ] FITS/HDF5 writers for cubes, spectra, images; metadata & provenance.
- [ ] Batch generation from a YAML catalogue (galaxy population + shear field).
- [ ] Reproducibility: seeded RNG; regenerate identical mocks from a config.

### Phase 5 — Inference engine
- [ ] Likelihood for the **IFU** case (fit model cube / moment maps to data).
- [ ] Likelihood for the **slit/grism** case (fit 2-D spectrum + image jointly).
- [ ] Priors module (Tully–Fisher prior on rotation, geometry priors).
- [ ] Sampler wrapper: `nautilus` (neural-network-boosted importance nested
      sampling), producing posterior samples and the Bayesian evidence.
- [ ] Recover (g1, g2) posterior per galaxy; save chains + summary statistics.
- [ ] Tests: injection–recovery on noiseless mocks recovers input shear.

### Phase 6 — Validation & forecasting
See `TEST_DATA.md` for the full validation-data plan (the four-rung ladder:
controlled mocks -> TNG50-based realistic mocks -> real IFU survey data ->
KINO's own instrument forecasts) and task breakdown; tracked here at a
summary level.
- [x] `kino/io/base.py`: common `Datacube` container (flux cube,
      wavelength/velocity axis, spaxel scale, variance/mask, redshift, units).
- [x] `kino/io/kinms.py`: KinMS wrapper -> `Datacube`, for known-truth
      injection tests (Tier A). Position-angle convention pinned by tests.
- [ ] `kino/io/manga.py`: reader for MaNGA/MaNGIA/iMaNGA FITS cubes ->
      `Datacube` (Tiers B/C) — pending a real fixture file to read against.
- [ ] `kino/io/download.py`: fetch-on-demand for the curated fixture subset,
      with checksums, cached under a git-ignored `data/` dir.
- [ ] `kino/analysis/injection.py`: known-shear injection-recovery on any
      `Datacube`, reporting recovered (g1, g2), bias (m, c), residual σ_ε.
      Blocked on Phases 2, 3, 5 (`kino.lensing`, `kino.instruments`,
      `kino.inference`).
- [ ] `kino/analysis/model_compare.py`: rank lensed / unlensed / tidally
      disturbed models by `nautilus` evidence (TEST_DATA.md Section 3).
      Blocked on Phases 1, 5.
- [ ] Injection–recovery suites: bias vs. S/N, size, inclination, shear amplitude.
- [ ] Shape-noise characterisation: residual σ_ε over galaxy ensembles.
- [ ] Multiplicative/additive bias (m, c) estimation and reporting.
- [ ] Null tests (zero-shear ensembles) and calibration diagnostics.
- [ ] Reproduce a published KL forecast (target σ_ε ≈ 0.02–0.04) as a benchmark.
- [ ] Compare IFU vs. slit/grism performance on matched galaxy populations.

### Phase 7 — Instrument case studies
- [ ] GMTIFS-like forecast: AO PSF, 25–50 mas spaxels, 4″×2″ field, R≈5–10k,
      0.9–2.5 µm; Hα/[O III] kinematics of z≈0.4–1.5 background discs.
- [ ] MANIFEST+GMACS-like forecast: fibre-IFU bundles / multi-slit, R≈2–5k,
      optical; survey-scale multiplex.
- [ ] Cluster-background proof-of-concept scenario (O(1) bright galaxies).

### Phase 8 — Docs, examples, release
- [ ] Tutorial notebooks in `notebooks/`: (1) simulate one galaxy, (2) run
      inference, (3) forecast. `notebooks/04_one_galaxy_inference.ipynb` is the
      main showcase and is scaffolded as a placeholder pending Phases 1–5.
- [ ] API docs (Sphinx or mkdocs) + published site.
- [ ] Worked examples for both instrument backends.
- [ ] Versioned release (`v0.1.0`), Zenodo DOI, citation file.

---

## 5. Quickstart (planned API)

```python
from kino.galaxy import DiscGalaxy
from kino.lensing import apply_shear
from kino.instruments import IFU, SlitSpectrograph
from kino.inference import fit_shear

# 1. intrinsic disc
gal = DiscGalaxy(r_e=0.5, inc=60, pa=35, v_flat=180, z=0.8)

# 2. lens it
scene = apply_shear(gal, g1=0.03, g2=-0.01)

# 3. observe (swap the backend for the other mode)
ifu   = IFU.from_config("config/gmtifs.yaml")
data  = ifu.observe(scene, exp_time=3*3600)

# 4. infer the shear back
posterior = fit_shear(data, sampler="nautilus")
print(posterior.summary())   # g1, g2 with credible intervals
```

---

## 6. References

Foundational and recent KL literature that KINO builds on:

- Blain (2002), *ApJ* **570**, L51 — original concept.
- Morales (2006), *ApJ* **650**, L21 — velocity-map shear technique.
- Huff, Krause, Eifler, George & Schlegel (2013), arXiv:1311.1489 — modern KL framework.
- de Burgh-Day et al. (2015), *MNRAS* **451**, 2161 — direct shear mapping.
- Gurri, Taylor & Fluke (2020), *MNRAS* **499**, 4591 — first measurement.
- DiGiorgio et al. (2021), arXiv:2109.14044 — joint imaging+spectroscopy framework.
- Pranjal R. S. et al. (2022), arXiv:2209.11811 — KL inference I (shape noise).
- Xu et al. (2023), *MNRAS* **519**, 2535 — Roman forecast.
- Xu et al. (2024), arXiv:2407.20867 — DESI KL.
- Pranjal R. S. et al. (2024/25), *MNRAS* staf845 / arXiv:2409.08367 — cluster lensing, O(1) galaxies.

## 7. License

MIT (see `LICENSE`).
