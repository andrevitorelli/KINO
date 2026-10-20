# KINO

**KI**nematic lensing i**N**ference and mock-**O**bservations

*(kino — from the Greek* kinema*, "movement" — a nod to the galaxy kinematics at the
heart of the method.)*

KINO is a Python toolkit to **simulate** and **test the inference of
kinematic lensing (KL)** — the measurement of weak gravitational shear from
the distorted velocity field of an individual disc galaxy. It works across
both observing modes used in the KL literature:

- **Integral-field units (IFUs)** — resolved 2-D velocity maps / datacubes.
- **Slit / grism spectrographs** — 1-D or partially-resolved spectra
  combined with imaging.

## The idea

Weak lensing is limited by **shape noise**: the intrinsic scatter of galaxy
ellipticities (σ_ε ≈ 0.26) dwarfs the ~1% shear signal, so conventional
cosmic shear must average over huge numbers of galaxies. Kinematic lensing
removes this limitation. For a rotationally supported disc, the projected
velocity field fixes the galaxy's intrinsic inclination and orientation
directly from the resolved kinematics. Lensing distorts the *image* but not
the *observed frequency* of light, so the mismatch between the shape implied
by the kinematics and the observed shape yields the shear for a **single**
galaxy, driving the effective shape noise down to σ_ε ≈ 0.02–0.04.

```
 intrinsic disc  ──►  apply shear  ──►  instrument model  ──►  mock data
 (image + Vlos)       (g1, g2)          (IFU or slit)          (cube / spectra + image)
        ▲                                                              │
        └──────────────  inference: recover (g1, g2) posterior  ◄──────┘
```

By simulating galaxies with a **known** input shear and running the
inference, KINO measures the per-galaxy shear precision, quantifies
residual shape noise, detects multiplicative/additive biases, and forecasts
performance for a given instrument and exposure time — for both IFUs and
slit spectrographs, on the same footing.

## Where to go next

- [Install](install.md) KINO and its optional extras.
- [Quickstart](quickstart.md): a real, runnable example in under a minute.
- The [Manual](manual/forecasting.md) for task-oriented guides: forecasting
  a real target, adding a new instrument config, running full posterior
  inference, and the unit/sign/angle conventions every module follows.
- The [API reference](api/galaxy.md) for the full public interface.

## Scope

**In scope:** forward modelling of a single sheared disc galaxy (image +
line-of-sight velocity field), interchangeable IFU/slit instrument
backends, Bayesian shear inference per galaxy, and ensemble forecasting
(shape noise, m/c bias, null tests).

**Out of scope (for now):** real-data reduction pipelines, cluster
mass-profile modelling and cosmological likelihoods (KINO stops at the
per-galaxy shear estimate), and full radiative transfer (emission lines
are modelled parametrically).
