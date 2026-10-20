# Adding an instrument config

Every config in `kino/config/` is a plain YAML file loaded with
`IFU.from_config(path)` or `SlitSpectrograph.from_config(path)`
(`kino.instruments.config` does the parsing). This page is the recipe used
to build `keck_kcwi.yaml`, `gemini_gmos_ifu.yaml`, `vlt_muse.yaml`, and the
AO-fed `gemini_nifs.yaml`/`keck_osiris.yaml`, and it generalizes to any new
IFU or slit spectrograph.

## The fields a config needs

```yaml
pixel_scale: "0.20 arcsec"      # spatial sampling
nx: 50                          # spaxels across
ny: 50                          # spaxels along

wavelength_min: "0.6950 um"     # spectral window (narrow it for speed; see below)
wavelength_max: "0.7050 um"
n_wave: 80                      # channel count -> sets the per-channel dispersion

lsf:
  fwhm: "2.333 angstrom"        # Gaussian line-spread function

psf:
  seeing_fwhm: "0.74 arcsec"    # combined with aperture_diameter for the optical PSF

throughput:
  wavelength: ["0.480 um", "0.750 um", "0.930 um"]
  transmission: [0.14, 0.35, 0.14]

aperture_diameter: "8.2 m"
obstruction_fraction: 0.013     # 0.0 if your aperture is already obstruction-corrected

detector:
  read_noise: "3.0 electron"
  dark_current: "0.0005 electron / s"

exposure_time: "1200 s"
n_exposures: 3

sky_electron_rate: "0 electron / s"  # per spaxel, already-detected rate (not a sky brightness)
```

Every physical quantity is a string `astropy.units.Quantity` can parse.
`PSF` also accepts optical-aberration keys (`defocus`, `astig1`, `astig2`,
`coma1`, `coma2`, `trefoil1`, `trefoil2`, `spher`) for an AO-fed instrument
— none of the configs shipped so far use them (all default to zero,
unaberrated optics).

**AO-fed instruments (`gemini_nifs.yaml`, `keck_osiris.yaml`) still only set
`seeing_fwhm`, not the aberration keys.** KINO has no dedicated Strehl-ratio
or core+halo AO PSF model, so both configs plug the instrument's own
quoted (NIFS) or computed-diffraction-limit (OSIRIS) AO-corrected FWHM
directly into `seeing_fwhm` as an effective single-Gaussian-equivalent
width. This is a documented approximation: a real AO PSF has a
near-diffraction-limited core atop an extended seeing-like halo, so this
understates flux lost to the halo and makes AO forecasts optimistic until
KINO gets a real AO PSF model. Follow the same pattern for any future
AO-fed config rather than inventing a different convention.

## The recipe

1. **Find real numbers where they're published.** Start from the
   instrument's own characteristics page: field of view and spatial
   sampling, grating/grism resolving power and wavelength coverage,
   detector pixel size, telescope aperture. These are usually the easiest
   to find and the least likely to need a fiducial.

2. **Throughput is almost always the hard part.** In rough order of
   preference:
     - A real on-sky measurement, the way `scripts/derive_goodman_throughput.py`
       derives Goodman's own curve from a NOIRLab exposure of a
       spectrophotometric standard star — the strongest option, but needs a
       real calibration frame to exist and be found.
     - A public exposure-time calculator's own quoted throughput (Gemini's
       ITC, ESO's ETC): `vlt_muse.yaml`'s curve is exactly this — ESO's own
       MUSE overview page states 14%/35%/14% at 480/750/930 nm, used
       unmodified, no interpolation invented.
     - A clearly-labelled fiducial shape anchored to one real peak-efficiency
       number from the instrument paper (`keck_kcwi.yaml` does this: a bell
       curve peaking at the literature's own "in excess of 45%" figure).

3. **Never leave a config blocked on a missing number.** If detector read
   noise, dark current, or sky background aren't published, search for a
   defensible fiducial (a cross-instrument proxy from the same detector
   family, a typical value for that class of detector) rather than leaving
   the config unusable. Every placeholder number lives in the file's own
   header comment, explicitly labelled `UNVERIFIED`/`PLACEHOLDER`, with
   either its source or the reasoning behind the fiducial — **never** a
   silently-assumed number. A working config with labelled fiducials ships;
   a blocked one doesn't, and real users will want to override these values
   themselves, which only works if the config loads in the first place.

4. **Keep the spectral/spatial window small by default.** A real
   instrument's full field or wavelength range is often far larger than
   any single target needs (MUSE's real WFM field is 60″×60″; the shipped
   config crops it to 10″×10″ for rendering speed) and makes every render
   needlessly slow. Document the crop in the header and tell users how to
   override it for the real full field/range.

5. **Pin every sourced number with a test.** One test file per config,
   mirroring `tests/instruments/test_goodman_config.py`: assert the
   geometry, detector, LSF, and throughput match what the header claims,
   and pin the *fact* that any placeholder is still a placeholder (so
   replacing it later is a deliberate, visible test change, not a silent
   pass). Finish with one real `ifu.observe(...)` call asserting a finite,
   non-trivial cube — confirms the config actually forecasts something
   sane, not just that the YAML parses.

6. **Cite everything in the header.** Every sourced number says where it
   came from (a URL, a paper with its full citation, an instrument manual
   section); every placeholder says why it's a placeholder and what a real
   value would need. This is what makes a config trustworthy enough to
   build a real proposal's forecast on top of — which is exactly what
   `sifs.yaml`/`goodman.yaml` were used for.
