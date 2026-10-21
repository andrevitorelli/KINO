# Examples

Runnable CLI scripts that exercise the KINO pipeline end-to-end, config-driven
and reproducible per `CLAUDE.md`.

- `simulate_one_galaxy.py` — intrinsic disc + shear -> instrument backend ->
  mock data (image, IFU datacube, or slit/grism spectrum, depending on the
  instrument config), via `kino.mocks.single.simulate_observation` +
  `kino.io.fitsio`.
- `run_inference.py` — mock data + instrument config -> `kino.inference`
  (`nautilus` sampler) -> (g1, g2) posterior.
- `generate_catalogue.py` — a YAML catalogue of galaxies (explicit list or
  a sampled population) + one global shear -> a batch of mock files + a
  `manifest.json`, via `kino.mocks.catalogue.generate_batch`.

```bash
python examples/simulate_one_galaxy.py --config kino/config/sifs.yaml \
    --line-flux 1e-15 --line-wavelength 0.65628 --g1 0.03 --g2 -0.01 \
    --out mock.fits

python examples/run_inference.py --data mock.fits --config kino/config/sifs.yaml

python examples/generate_catalogue.py \
    --catalogue kino/config/catalogue_example.yaml --out mock_batch/
```

`run_inference.py` is genuinely slow (nested sampling needs thousands of
likelihood evaluations, each a full PSF-convolved render) -- budget minutes,
not seconds, and raise `--n-live`/`--n-eff` for a production-quality
posterior rather than the modest defaults. `--help` on either script lists
every option.

`simulate_one_galaxy.py` only frees `(g1, g2)` in `run_inference.py` (every
other `DiscGalaxy` field is held fixed at the simulated truth); for freeing
inclination/`v_flat`/etc. too, see `docs/manual/inference.md` or
`notebooks/04_one_galaxy_inference.ipynb`'s own, more complete pattern.

For an end-to-end narrative rather than a CLI, see
`notebooks/04_one_galaxy_inference.ipynb` and the published docs site
(`docs/quickstart.md`, `docs/manual/`).
