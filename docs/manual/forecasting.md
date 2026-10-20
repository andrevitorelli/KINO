# Forecasting a target

This is the same workflow that produced the real forecast numbers in
KINO's own SOAR 2027A proposal — going from a target's coordinates and
line flux to a shear S/N forecast through a real night at a real site.

```python
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from kino.galaxy import DiscGalaxy
from kino.instruments.ifu import IFU
from kino.analysis.forecast import ForecastTarget, forecast_ifu
from kino.observing import SOAR, night_window, SkyLibrary, solar_flux

gal = DiscGalaxy(r_e=1.26, inc=55.0, pa=30.0, v_flat=180.0, z=0.4988, sigma0=40.0)
coord = SkyCoord(ra=328.62827 * u.deg, dec=0.63725466 * u.deg)
target = ForecastTarget(
    coord, gal, 8.4e-16 * u.erg / u.s / u.cm**2, "OII3727", name="demo"
)

ifu = IFU.from_config("kino/config/sifs.yaml")
lib = SkyLibrary.load("data/skycalc")        # needs the `sky` extra's data, see Install
night = night_window(SOAR, "2027-07-28")
F107 = solar_flux("2027-07")[0]              # monthly-mean 10.7cm solar flux

forecast = forecast_ifu(
    ifu, target, night, lib,
    g=(0.14, 0.0),
    seeing_ref=[1.0] * u.arcsec,
    solar_flux=F107,
    time_step=60 * u.min,
    max_airmass=2.0,
    window_kms=350.0,
)
print("shear S/N:", np.nanmax(forecast.shear_snr_stat[:, 0]))
```

## The `window_kms` gotcha

Don't drop `window_kms=350.0` from that call — it matters more than it
looks. `forecast_ifu`/`forecast_slit` default to `window_kms=500.0`, a
spectral half-width wider than most emission lines actually need. The
per-spaxel detection gate inside `shear_fisher`'s cube estimator aborts
the *entire* calculation (returning an undetectable result, not just a
worse one) if literally zero spaxels individually clear `snr_threshold`
— and a too-wide window adds pure-noise channels that push marginal
spaxels below that threshold sooner, especially at low exposure time or
poor seeing, long before the shear information itself is actually lost.

This produced a real false "seeing cliff" in an early draft of KINO's own
SOAR proposal — a sharp, physically implausible dropout that very nearly
shipped as a stated hard requirement before it was traced to this exact
mechanism. Pick `window_kms` from the line's own real extent instead of
trusting the default: render the model with `rng=None` and measure how
much flux actually falls within a candidate half-width before committing
to it (`kino.analysis.moments.crop_to_line` does this cropping once you
know the right value). This is a known pipeline sharp edge, not fixed at
the gate level yet — treat the default as a starting point to check, not
a safe choice.

## Scanning a whole night, or a whole semester

`forecast_ifu`/`forecast_slit` already tile `(time, seeing)` through one
night for you (that's what the loop over `night.times` inside them does).
To scan multiple candidate nights across a semester, call them once per
date with `night_window(SOAR, date)` and `solar_flux(date[:7])` for each
date — this is exactly what the semester-visibility scans in
`notebooks/10_sifs_forecast.ipynb`/`notebooks/11_goodman_forecast.ipynb`
do, just looped externally rather than built into the function itself.

## IFU vs. slit

`forecast_ifu` takes an `IFU`; `forecast_slit` takes a `SlitSpectrograph`
and additionally models slit losses and misalignment directly in the
render (not as a separate correction factor) — swap the instrument and the
function, keep everything else the same. See
[Adding an instrument config](adding-an-instrument.md) for how either kind
of config is built.
