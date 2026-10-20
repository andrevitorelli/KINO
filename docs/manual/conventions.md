# Conventions

Every module in KINO follows the same units and sign conventions. Getting
one of these backwards (a shear sign, a PA direction, an ellipticity
definition) produces a plausible, *wrong* answer rather than an obvious
crash — read this page before writing code that touches angles, signs, or
shear.

## Units

- **Positions**: arcsec, origin at the galaxy centroid.
- **Velocities**: km/s.
- **Wavelengths**: µm internally; Å or nm at public API boundaries where
  that's the field's own convention (config files, spectra).
- **Angles**: degrees.
- All public API boundaries use `astropy.units` explicitly — if a function
  takes a bare float where you'd expect a `Quantity`, check its docstring
  for which unit it assumes.

## Coordinates

Image-plane Cartesian `(x, y)` in arcsec, origin at the galaxy centroid;
`x` increases to the East, `y` to the North.

## Position angle (PA)

Degrees, **East of North**, of the disc's major axis.

## Inclination

`i = 0` is face-on, `i = 90°` is edge-on.

## Ellipticity & shear

**Reduced shear** `g = g1 + i·g2` (complex), paired with the
**distortion/ellipticity convention** `ε = (a−b)/(a+b)`. If you ever need
the `χ = (a²−b²)/(a²+b²)` convention instead, convert explicitly and label
it — don't assume the two are interchangeable.

## Line-of-sight velocity

```
Vlos = Vsys + Vrot(r) · sin(i) · cos(θ)
```

in the disc frame, `θ` measured from the major axis. Positive `Vlos` means
redshifted (receding). This sign convention is kept consistent between the
forward model and every likelihood — a common bug when extending KINO is
shearing only the morphology and forgetting the velocity field needs the
same shear transform (`apply_shear` already handles this correctly; see its
docstring for the delensing transform both quantities share).

## Cosmology

`astropy.cosmology.Planck18` is the default everywhere, always injectable
— KINO never hard-codes a numeric distance.

## Randomness

Every stochastic call takes an explicit, seeded `numpy.random.Generator`.
KINO never reads the global `np.random` state — this is what makes a run
(a config + a seed) exactly reproducible.

## Shear is applied to the image *and* the velocity field

A disc galaxy's surface brightness and its line-of-sight velocity field are
both remapped through the identical delensing transform
(`kino.lensing.scene.LensedScene`/`apply_shear`) — shear is never applied
to the morphology alone. This is the single most important invariant in
the forward model: a shear measurement that only distorted the image shape
would be measuring something else entirely.

## Instrument flux calibration

A `DiscGalaxy`'s `flux` (and anything derived from it, like a
`LensedScene`'s surface brightness) is in the galaxy's own arbitrary units
— KINO is intentionally unitless there. Physical calibration happens at
`Instrument.observe()` time: pass an explicit `line_flux`
(`astropy.units.Quantity`, e.g. `erg/s/cm**2`) and the renderer rescales
the shape by `line_flux / galaxy.flux`. Because a `LensedScene` is
evaluated by sampling the *unlensed* galaxy at each observed (delensed) sky
position, the image-integrated flux comes out magnified relative to
`galaxy.flux` automatically — `line_flux` is always the source's
intrinsic, unlensed brightness; never re-apply magnification on top of it.

## Noise and sky background

Instrument backends return detected **electrons**, combining Poisson
source/sky/dark-current shot noise with Gaussian read noise. `rng=None`
returns the noiseless expected-count cube instead of drawing a noise
realisation — useful for inspecting the pure signal. Sky background
(`sky_electron_rate`) is specified **already as a detected electron rate**
(as an observatory's own exposure-time calculator would give it), not a
sky surface brightness — it does not go through the source's own
throughput/aperture pipeline.
