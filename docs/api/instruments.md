# kino.instruments

The common `Instrument` API (PSF, LSF, throughput, detector, noise) and its
two backends: `IFU` (datacubes) and `SlitSpectrograph` (slit/grism spectra
+ companion imaging). See
[Adding an instrument config](../manual/adding-an-instrument.md) for how a
new real telescope/instrument is turned into a config file.

::: kino.instruments
    options:
      show_root_heading: false
      show_root_toc_entry: false

::: kino.instruments.ifu.IFU
    options:
      show_root_heading: true

::: kino.instruments.slit.SlitSpectrograph
    options:
      show_root_heading: true

::: kino.instruments.config
    options:
      show_root_heading: true
