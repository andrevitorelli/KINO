# Examples

Runnable CLI scripts that exercise the KINO pipeline end-to-end, config-driven
and reproducible per `CLAUDE.md`.

- `simulate_one_galaxy.py` — intrinsic disc + shear -> instrument backend ->
  mock data (image, IFU datacube, or slit/grism spectrum, depending on the
  instrument config).
- `run_inference.py` — mock data + instrument config -> `kino.inference`
  (`nautilus` sampler) -> (g1, g2) posterior.

Both scripts currently raise `NotImplementedError`: they wire up the intended
CLI, but the underlying `kino.galaxy` / `kino.lensing` / `kino.instruments` /
`kino.mocks` / `kino.inference` modules are empty skeletons pending Phases
1–5 of the roadmap in `README.md`. Fill in the marked steps as each phase
lands, keeping the CLI shape stable.

For an end-to-end narrative rather than a CLI, see
`notebooks/04_one_galaxy_inference.ipynb`.
