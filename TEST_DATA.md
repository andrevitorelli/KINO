# TEST_DATA.md — instructions for building KINO's test-data layer

**Audience:** a Claude Code session working in the KINO repo.
**Purpose:** integrate external simulated/real IFU datacubes as test and validation
inputs for the kinematic-lensing (KL) forward model and inference. Follow this
alongside `README.md` (roadmap) and `CLAUDE.md` (conventions). This work mostly
lands under **Phase 6 — Validation & forecasting**, plus a new `kino/io/` adapter
layer.

---

## 0. The validation ladder (why these datasets, in order)

Build and test the inference against progressively more realistic data. Each rung
has a distinct job; implement them in this order.

1. **Controlled mocks (known ground truth)** — cubes we generate with exact input
   parameters. Job: injection–recovery and unit tests. A lensed input *must* be
   recovered with the correct shear.
2. **Realistic cosmological-sim mocks (TNG50-based)** — real astrophysical
   messiness, including genuinely interacting/tidally disturbed galaxies. Job: test
   the **KL-vs-contamination model comparison** (Section 3). Inject a *known* shear
   on an intrinsically regular sim galaxy and confirm the evidence prefers the
   lensing model; on a genuinely tidal galaxy, confirm it does not.
3. **Real IFU survey data (low-z)** — MaNGA/SAMI/etc. Job: does the pipeline
   survive real instrument/reduction messiness at all.
4. **KINO instrument forecasts** — our own simulator for the GMTIFS / MANIFEST+GMACS
   regime. Job: the actual GMT science forecast.

> **Caveat to honour everywhere:** rungs 2–3 are **low-redshift, optical, coarse
> spatial resolution**. They validate the *method*, not the GMT regime (z ≈ 0.4–1.5,
> AO resolution, NIR). Never present a low-z validation as a GMTIFS forecast. Keep
> KINO's own simulator (rung 4) as the instrument-forecast path.

---

## 1. Data sources

Do **not** commit large data files to the repo. Add a downloader + a small curated
subset (see Section 4). Record citation info for each source in `CITATION`/docs.

### Tier A — controlled mock generators (build cubes with known truth)
- **KinMS** (Davis et al. 2013) — arbitrary rotating-disc emission-line cubes.
  https://github.com/TimothyADavis/KinMSpy  · pip: `kinms`
- **SimSpin** (Harborne et al. 2020) — mock IFS kinematic/spectral cubes from
  N-body/hydro particle data; configurable PSF, spaxel scale, LSF. (R package;
  use as an *independent* cross-generator.) https://github.com/kateharborne/SimSpin
- **MARTINI** (Oman et al.) — mock **HI** datacubes from simulations, for the
  radio velocity-map variant (cf. Morales 2006). https://github.com/kyleaoman/martini

### Tier B — ready-made realistic mock IFU cubes (TNG50) ← most valuable
- **MaNGIA** (Sarmiento et al. 2022) — large set of TNG50 mock MaNGA cubes with
  stellar kinematics. Code: https://github.com/reginasar/TNG_MaNGA_mocks
  (mirror: https://github.com/illustristng/MaNGIA_TNG)
- **iMaNGA** (Nanni et al. 2022–2023) — TNG50 mock MaNGA cubes built with MaStar
  stellar spectra; catalogue + data: http://www.icg.port.ac.uk/imanga/

### Tier C — real IFU survey datacubes (final low-z realism check)
- **MaNGA** (SDSS DR17, ~10k galaxies) — public DAP velocity maps + cubes.
- **SAMI**, **CALIFA**, **MUSE archive** — additional real IFS cubes.

---

## 2. What to build (tasks)

Implement in `kino/io/` (readers) and `kino/analysis/` (validation drivers). Tick
these off in `README.md` Phase 6 as they land, and add tests for each.

- [ ] `kino/io/base.py`: a common in-memory **`Datacube`** container (flux cube,
      wavelength/velocity axis, WCS/spaxel scale, variance/mask, redshift, units)
      that the inference likelihood consumes — regardless of source.
- [ ] `kino/io/manga.py`: reader for MaNGA/MaNGIA/iMaNGA FITS cubes → `Datacube`
      (and a helper to pull the DAP stellar/gas velocity + dispersion maps).
- [ ] `kino/io/kinms.py`: thin wrapper to generate a KinMS cube → `Datacube`, for
      known-truth injection tests.
- [ ] `kino/io/download.py`: fetch-on-demand for a **curated subset** (Section 4),
      with checksums; cache under a git-ignored `data/` dir. Never auto-commit data.
- [ ] `kino/analysis/injection.py`: take any `Datacube` (or a sim galaxy), apply a
      **known** shear via KINO's forward model, re-observe, run inference, and
      report recovered vs. input (g1, g2), bias (m, c), and residual σ_ε.
- [ ] `kino/analysis/model_compare.py`: run the model set (Section 3) on one
      dataset via the nested sampler; return ranked evidences + Bayes factors.
- [ ] `tests/`: mirror each reader/driver with an invariant test (Section 5).

---

## 3. Model-comparison test (the key science check)

The distinctive KL systematic: a distorted velocity field can arise from lensing
**or** from intrinsic causes. Define a comparable set of forward models and rank
them by Bayesian evidence using **nautilus** (see `CLAUDE.md`).

> **Amended:** this section originally proposed dynesty as an independent
> evidence cross-check alongside nautilus. The project has since standardised
> on nautilus as the sole sampler everywhere (see `CLAUDE.md`'s "Language /
> stack" convention) to keep the sampler stack simple; the log-Z cross-check
> is dropped. The prior-sensitivity analysis below still applies.

Minimum model set:
1. **Lensed regular disc** (the KL hypothesis: intrinsic disc + shear g1, g2).
2. **Unlensed regular disc** (null: shear fixed to zero).
3. **Tidally disturbed / interacting disc** (intrinsic non-lensing distortion).
4. (stretch) warp, bar/non-circular streaming, beam-smearing-only.

Validation expectations (write these as assertions where feasible):
- On a TNG50 galaxy that is intrinsically **regular**, with a known shear injected:
  evidence should favour model **1** over **2** and **3**.
- On a TNG50 galaxy that is intrinsically **interacting/tidal**, with **no** shear:
  evidence should favour **3**, not **1** — i.e. the method does not mistake tidal
  distortion for lensing.
- Report a **prior-sensitivity analysis**: Bayes factors depend on priors (Jeffreys–
  Lindley); use proper, physically-motivated priors and show how the ranking moves
  under reasonable prior variations.

---

## 4. Curated test subset (keep the repo light)

- Assemble a **small** fixture set (a handful of cubes) covering: a clean rotator,
  an inclined rotator, and an interacting/tidal galaxy from MaNGIA or iMaNGA, plus
  one or two KinMS known-truth cubes.
- Store under a git-ignored `data/` directory; provide `kino/io/download.py` to
  fetch them by checksum. Ship only tiny synthetic cubes (if any) inside the repo,
  for fast unit tests.
- Document exact provenance (source, object ID, download date) for every fixture.

---

## 5. Conventions & correctness (read `CLAUDE.md` first)

- Convert every external cube into KINO's documented conventions on read: positions
  in arcsec, velocities in km/s, PA East-of-North, inclination 0=face-on, the
  reduced-shear/ellipticity convention fixed in `kino/lensing/`. Do the conversion
  **in the reader**, not scattered downstream.
- Handle each survey's WCS, spectral axis (wavelength vs. velocity), masks, and
  variance arrays explicitly; never assume a layout — verify against the file.
- Tests should assert **invariants**: a KinMS cube injected with zero shear recovers
  g≈0; a known injected shear is recovered within the credible interval; reading →
  writing → reading a cube is idempotent; unit conversions round-trip.
- Respect data licences and cite each dataset (MaNGIA, iMaNGA, KinMS, SimSpin,
  MARTINI, MaNGA/SDSS) in the docs and any resulting paper.

---

## 6. Suggested order of work

1. `Datacube` container + KinMS reader + an injection–recovery test (fast, known truth).
2. MaNGA/MaNGIA reader + fetch one interacting and one regular TNG50 galaxy.
3. `model_compare.py` with models 1–3 and the two validation expectations in §3.
4. Prior-sensitivity analysis on the model-comparison evidences.
5. (later) real MaNGA cube smoke test; then hand off to KINO instrument forecasts.
