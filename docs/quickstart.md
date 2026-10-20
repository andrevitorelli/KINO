# Quickstart

This is a real, runnable example — every line below was executed against
the current API before being written down here. It builds a disc galaxy,
shears it, observes it through a real SOAR/SIFS configuration, and forecasts
how precisely that shear could be recovered.

```python
import numpy as np
import astropy.units as u
from kino.galaxy import DiscGalaxy
from kino.lensing import apply_shear
from kino.instruments.ifu import IFU
from kino.analysis.fisher import shear_fisher

# 1. intrinsic disc (positions in arcsec, velocities in km/s, angles in degrees)
gal = DiscGalaxy(r_e=0.5, inc=60, pa=35, v_flat=180, z=0.05, sigma0=30.0)

# 2. lens it (reduced shear g = g1 + i*g2)
scene = apply_shear(gal, g1=0.03, g2=-0.01)

# 3. observe it through a real instrument config
ifu = IFU.from_config("kino/config/sifs.yaml")
rng = np.random.default_rng(42)
cube = ifu.observe(
    scene,
    line_flux=1e-15 * u.erg / u.s / u.cm**2,
    line_wavelength=0.65628 * u.um,  # rest-frame Halpha
    rng=rng,
)
print(cube.flux.shape, cube.flux_unit)   # (290, 26, 50) electron

# 4. how precisely could this shear be recovered?
result = shear_fisher(
    ifu, gal,
    line_flux=1e-15 * u.erg / u.s / u.cm**2,
    line_wavelength=0.65628 * u.um,
    g=(0.03, -0.01),
)
sigma_g1, sigma_g2 = result.sigma_marginal[:2]
print(f"sigma(g1)={sigma_g1:.3f}  sigma(g2)={sigma_g2:.3f}")
```

That's the whole forward model end to end: **intrinsic galaxy → shear →
instrument → mock data**, plus the Fisher-matrix shortcut for "how good
would this measurement be" without running a full posterior fit.

`rng=None` instead of a seeded generator returns the noiseless,
expected-count cube rather than drawing a noise realisation — useful for
inspecting the signal itself before adding noise.

## Next steps

- Swap `"kino/config/sifs.yaml"` for any other config in `kino/config/` —
  `keck_kcwi.yaml`, `gemini_gmos_ifu.yaml`, `vlt_muse.yaml`, `goodman.yaml`
  (a `SlitSpectrograph`, not an `IFU` — see
  [Forecasting a target](manual/forecasting.md)), or your own.
- [Forecasting a target](manual/forecasting.md) goes from a real line flux
  and real observing conditions to a shear S/N forecast through a night.
- [Running inference](manual/inference.md) replaces the Fisher shortcut
  above with a full `(g1, g2)` posterior from nested sampling.
