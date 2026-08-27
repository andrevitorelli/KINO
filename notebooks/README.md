# Notebooks

Jupyter notebooks. Install with the `notebooks` extra (and `data` for
`00_kinms_datacube_demo.ipynb`, which needs the `kinms` package):

```bash
pip install -e ".[notebooks,data]"
jupyter lab notebooks/
```

Numbered in the order each piece of the pipeline was built, not necessarily
the order to read them in.

- `00_kinms_datacube_demo.ipynb` — **runs today.** Demonstrates the io layer:
  `kino.io.base.Datacube` and the KinMS-backed known-truth cube generator
  (`kino.io.kinms`, `TEST_DATA.md` Tier A). Generates a disc cube, visualizes
  its flux/velocity maps, and shows the face-on and position-angle
  invariants that `tests/io/test_kinms.py` also pins down.
- `01_discgalaxy_demo.ipynb` — **runs today.** Demonstrates `kino.galaxy`
  (Phase 1): the intrinsic, unlensed `DiscGalaxy` model — surface
  brightness, the three rotation-curve shapes, `Vlos`, velocity dispersion,
  and the ellipticity-from-inclination relation. Includes the same KinMS
  cross-check as `tests/galaxy/test_kinms_consistency.py`, shown visually.
- `02_shear_misalignment_demo.ipynb` — **runs today.** Demonstrates
  `kino.lensing` (Phase 2): shears a `DiscGalaxy` and reproduces the classic
  kinematic-lensing diagnostic plot — the lensed velocity field with the
  true kinematic position angle and the (lensed) photometric position angle
  overplotted, visibly misaligned. Also shows the misalignment vanishing at
  zero shear and when the shear is aligned with the disc's own axes (see
  `tests/lensing/test_scene.py`), and sweeps the misalignment angle against
  shear orientation.
- `03_tully_fisher_shear_signal.ipynb` — **runs today.** The complementary
  case to `02`: when shear *is* aligned with the disc's own axes (its PA
  misalignment blind spot), two other signals survive — the photometric
  ellipticity magnitude still shifts, and pure shear magnifies flux
  (`kino.lensing.shear.magnification`, `mu = 1/(1-|g|^2)`) even with no
  convergence, biasing a Tully-Fisher-based rotation-velocity estimate that
  doesn't correct for it. Uses a toy (not calibrated) TF relation to isolate
  both effects cleanly.
- `04_one_galaxy_inference.ipynb` — the eventual main showcase: intrinsic
  galaxy + known shear -> instrument backend -> mock data (image, IFU
  datacube, or slit/grism spectrum) -> `nautilus`-sampled (g1, g2) posterior,
  compared against the injected truth. Still a scaffold — markdown cells lay
  out the full narrative and code cells sketch the intended calls, but most
  cells will not execute until `kino.instruments` and `kino.inference` land
  (see the roadmap in the top-level `README.md`; the galaxy and lensing
  halves are done, see `01_discgalaxy_demo.ipynb`,
  `02_shear_misalignment_demo.ipynb`, and
  `03_tully_fisher_shear_signal.ipynb`). Fill in and re-run each section as
  its phase ships, rather than restructuring the narrative.
