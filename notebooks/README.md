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
  cells will not execute until `kino.inference` lands (see the roadmap in the
  top-level `README.md`; the galaxy, lensing, and IFU-instrument halves are
  done, see `01_discgalaxy_demo.ipynb`, `02_shear_misalignment_demo.ipynb`,
  `03_tully_fisher_shear_signal.ipynb`, and `05_ifu_instrument_demo.ipynb`).
  Fill in and re-run each section as its phase ships, rather than
  restructuring the narrative.
- `05_ifu_instrument_demo.ipynb` — **runs today.** Demonstrates
  `kino.instruments` (Phase 3): renders a lensed `DiscGalaxy` scene through
  `kino.instruments.ifu.IFU`, loaded from the GMTIFS-like fiducial config
  (`kino/config/gmtifs.yaml`) -- detector/pixelization/PSF/LSF/throughput
  settings printed straight from the config, the actual `galsim`
  optical+seeing PSF kernel, the rendered (pixelized, PSF-convolved) cube
  next to the intrinsic continuous scene for comparison, a noiseless vs.
  noisy realisation with per-spaxel S/N, and sweeps of seeing and spectral
  resolution. See `tests/instruments/` for the invariants (flux
  conservation, PSF/LSF broadening, S/N vs. exposure) this notebook shows
  visually.
- `06_mangia_datacube_demo.ipynb` — **runs today, against synthetic data.**
  Demonstrates `kino.io.manga` (Phase 6, MaNGIA/TNG50 mock MaNGA cubes):
  reads a FITS cube through `read_mangia_cube` and runs
  `kino.analysis.moments.compute_moment_maps` on it. No real MaNGIA cube was
  available (downloading one needs a free IllustrisTNG API key this session
  didn't have -- see `kino/io/download.py`'s module docstring), so the cube
  is a known galaxy rendered through `kino.instruments.ifu.IFU` and
  repackaged into MaNGIA's documented FITS layout instead, which still
  genuinely exercises the reader and the moment-map recovery against a
  known truth. Also demonstrates, deliberately, a real numerical caveat:
  `compute_moment_maps`'s `dispersion` collapsing under noise when the
  spectral window is too wide relative to the line (see that function's
  module docstring).
