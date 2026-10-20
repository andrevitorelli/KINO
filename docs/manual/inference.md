# Running inference

The [Quickstart](../quickstart.md)'s `shear_fisher` call gives a fast
Cramér–Rao *bound* on shear precision. To actually recover a `(g1, g2)`
posterior from mock (or real) data, fit a likelihood with nested sampling.
This is the real pattern `notebooks/04_one_galaxy_inference.ipynb` runs
end to end — reuse it rather than re-deriving one, since nested sampling
through the full cube likelihood takes real time (minutes, not seconds;
budget accordingly, and prefer running it as a script or notebook rather
than inline in a REPL).

```python
from kino.inference.likelihood import IFUCubeLikelihood
from kino.inference.priors import build_prior, tully_fisher_prior
from kino.inference.sampler import fit_shear

# Everything the likelihood should treat as *fixed* (known from imaging/
# redshift), as opposed to free parameters fit from the data cube itself.
fixed_galaxy_params = {
    "r_e": gal.r_e,
    "pa": gal.pa,
    "z": gal.z,
    "sigma0": gal.sigma0,
    "flux": gal.flux,
}

likelihood = IFUCubeLikelihood(
    ifu=ifu,
    line_wavelength=line_wavelength,
    data=data,                       # the observed (or mock) Datacube
    fixed_galaxy_params=fixed_galaxy_params,
    fixed_line_flux=line_flux,
)

# Free parameters and their priors. g1/g2/line_flux are handled by the
# likelihood itself and must NOT be passed as DiscGalaxy fields; any other
# free parameter must be a real DiscGalaxy field.
prior = build_prior(
    {
        "g1": (-0.15, 0.15),
        "g2": (-0.15, 0.15),
        "inc": (30.0, 80.0),
        "v_flat": tully_fisher_prior(gal.flux, v0=180.0, flux0=1.0, scatter_dex=0.15),
    }
)

posterior = fit_shear(prior, likelihood, n_live=150, seed=0, n_eff=300)
```

`posterior.samples` holds equal-weighted posterior samples (columns in
prior order), and `posterior.log_z` the Bayesian evidence. `nautilus`
(neural-network-boosted importance nested sampling) is KINO's sole
sampler — see `CLAUDE.md`'s "Language / stack" convention for why.

## Which likelihood?

Three are available (`kino.inference.likelihood`), sharing one forward
model:

| Likelihood | Instrument | Notes |
|---|---|---|
| `IFUCubeLikelihood` | `IFU` | Voxel-level; the one to use for an IFU. |
| `SlitCubeLikelihood` | `SlitSpectrograph` | Voxel-level; fits `SlitObservation.spectrum` only — the companion broadband image is a placement aid, never fit. |
| `IFUMomentLikelihood` | `IFU` | Compares flux/Vlos/dispersion *moment maps* instead of the raw cube. Kept for comparison only — it has a real noise bias at low S/N (see its own docstring). Prefer `IFUCubeLikelihood` for an actual shear measurement. |

All three forward-model each trial through the *same* instrument
`.observe(..., rng=None)` used for the data — never compare a rendered cube
against an analytically-evaluated model, or the fit is biased by exactly
the PSF/LSF beam-smearing amount.

## Sanity-check before sampling

Nested sampling is expensive; check the likelihood is sane first by
evaluating it directly at the injected truth and slightly off it — the
truth should score higher:

```python
ll_truth = likelihood({"inc": gal.inc, "v_flat": gal.v_flat, "g1": g1_true, "g2": g2_true})
ll_off = likelihood({"inc": gal.inc, "v_flat": gal.v_flat, "g1": g1_true + 0.05, "g2": g2_true})
assert ll_truth > ll_off
```
