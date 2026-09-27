#!/usr/bin/env python
"""Derive a real Goodman system-throughput curve from an on-sky standard star.

``kino/config/goodman.yaml``'s throughput was originally a cross-instrument
guess (SIFS's own curve divided by a stated efficiency ratio, see git history).
This script replaces it with an actual measurement: NOIRLab publishes, on its
Goodman "Red Camera Exposure Times" page, the raw 2-D frame, the extracted 1-D
spectrum (in detector counts) and the full FITS header for an on-sky exposure
of the spectrophotometric standard EG 21 (400 l/mm grating, mode M1, 1.0"
slit, 2x2 binning, 150 s, airmass 1.41, seeing 0.77" FWHM -- all read from the
FITS header itself, not assumed). Given EG 21's independently published
absolute flux (the IRAF/CTIO "ctionewcal" standard-star table), the measured
counts imply a system throughput:

    T_sys(lambda) = counts(lambda) * gain
                    / [photon_flux(lambda) * dlambda_pix * A_eff * t_exp
                       * T_atm(lambda, airmass) * f_slit(lambda)]

where ``T_atm`` is KINO's own CTIO extinction curve (`kino.observing.extinction`)
and ``f_slit`` is the point-source slit-enclosed fraction computed from KINO's
own PSF kernel (`kino.instruments.base.PSF`) at the observation's seeing --
using the project's own atmosphere and PSF models keeps this consistent with
how ``kino.instruments.slit.SlitSpectrograph`` renders a mock, rather than
introducing a second, independent slit-loss formula.

Caveats (read before trusting the printed numbers):

* **Wrong grating.** This is the 400 l/mm grating; ``goodman.yaml`` uses 930
  l/mm for its higher spectral resolution. NOIRLab separately publishes a
  measured 930/1200 l/mm curve, but only for the *blue* camera
  (https://noirlab.edu/science/programs/ctio/instruments/goodman-high-throughput-spectrograph/instrument-characteristics/blue-camera-throughput);
  its shape and peak value (~0.30-0.38, blue-peaked, declining redward) agree
  with the number derived here to within the systematics below, which is the
  best cross-check available -- there is no published 930 l/mm red-camera
  curve to compare against directly.
* **One star, one night.** NOIRLab's own page notes total counts for the same
  object/setup vary by ~50% night to night (clouds, lunar phase/distance);
  treat the normalisation here as uncertain at that level.
* **Slit-loss and extinction models are KINO's own**, not independently
  verified against this specific observation.
* Below ~3700 A the derived curve is dominated by detector/extraction edge
  effects, not real throughput -- not used for the config's throughput nodes.

Usage: ``python scripts/derive_goodman_throughput.py`` (network; caches the
three small NOIRLab/GitHub files under git-ignored ``data/goodman_throughput/``).
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import astropy.units as u
import numpy as np
from astropy.constants import c as C_LIGHT
from astropy.constants import h as PLANCK_H
from astropy.io import fits

from kino.instruments.base import PSF
from kino.observing.extinction import atmospheric_transmission

CACHE = Path("data/goodman_throughput")
SOURCES = {
    "eg21_1d.dat": "https://noirlab.edu/science/sites/default/files/media/archives/documents/scidoc1824.txt",
    "eg21_2d_header.fits": "https://noirlab.edu/science/sites/default/files/media/archives/documents/scidoc1759.txt",
    "eg21_flux.dat": "https://raw.githubusercontent.com/iraf-community/iraf/main/noao/lib/onedstds/ctionewcal/eg21.dat",
}
#: Report-quality binning of the result (matches the flux table's own 50 A bins).
BIN_EDGES = np.arange(3700.0, 7150.0, 300.0)


def fetch(name: str, url: str, *, n_bytes: int | None = None) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / name
    if f.exists():
        return f
    if n_bytes is None:
        urlretrieve(url, f)  # noqa: S310 -- fixed NOIRLab/GitHub URLs, not user input
    else:
        import urllib.request

        req = urllib.request.Request(url, headers={"Range": f"bytes=0-{n_bytes - 1}"})  # noqa: S310
        with urllib.request.urlopen(req) as r, open(f, "wb") as out:  # noqa: S310
            out.write(r.read())
    return f


def slit_fraction(seeing: u.Quantity, slit_width: u.Quantity, wavelength: u.Quantity) -> float:
    """Point-source fraction enclosed by an (effectively infinite) slit, from KINO's own PSF."""
    pix_over = 0.02 * u.arcsec
    kernel = PSF(seeing_fwhm=seeing).kernel(
        pix_over, wavelength=wavelength, aperture_diameter=4.1 * u.m, obscuration=0.0
    )
    n = kernel.shape[0]
    x = (np.arange(n) - (n - 1) / 2.0) * pix_over.to_value(u.arcsec)
    profile = kernel.sum(axis=0)
    profile = profile / profile.sum()
    half = 0.5 * slit_width.to_value(u.arcsec)
    return float(profile[np.abs(x) <= half].sum())


def main() -> None:
    spec_path = fetch("eg21_1d.dat", SOURCES["eg21_1d.dat"])
    hdr_path = fetch("eg21_2d_header.fits", SOURCES["eg21_2d_header.fits"], n_bytes=28800)
    flux_path = fetch("eg21_flux.dat", SOURCES["eg21_flux.dat"])

    header = fits.Header.fromstring(hdr_path.read_bytes(), sep="")
    airmass = float(header["AIRMASS"])
    exptime = float(header["EXPTIME"])
    gain = float(header["GAIN"])
    print(f"AIRMASS={airmass} EXPTIME={exptime}s GAIN={gain} GRATING={header['WAVMODE']!r}")

    wave_A, counts = np.loadtxt(spec_path, unpack=True)
    flux_wave_A, flux_mag = np.loadtxt(flux_path, unpack=True, usecols=(0, 1))
    # IRAF onedstds magnitude convention: m = -2.5 log10(F_lambda) - 21.10, F_lambda in
    # erg / s / cm^2 / A (Stone & Baldwin 1983; Baldwin & Stone 1984 for this "ctionewcal" table).
    f_lambda_nodes = 10 ** (-0.4 * (flux_mag + 21.10))
    f_lambda = np.interp(wave_A, flux_wave_A, f_lambda_nodes)

    e_photon = (PLANCK_H * C_LIGHT / (wave_A * u.AA)).to(u.erg).value
    photon_flux = f_lambda / e_photon  # photons / s / cm^2 / A

    wave_clipped = np.clip(wave_A, 3200.0, 10870.0)  # extinction table's own range
    t_atm = atmospheric_transmission(wave_clipped * u.AA, airmass)

    nodes_um = np.array([0.35, 0.45, 0.55, 0.65, 0.75])
    frac_nodes = np.array(
        [slit_fraction(0.77 * u.arcsec, 1.0 * u.arcsec, w * u.um) for w in nodes_um]
    )
    f_slit = np.interp(wave_A / 1e4, nodes_um, frac_nodes)

    a_eff_cm2 = np.pi * (410.0 / 2) ** 2  # 4.1 m aperture, no obstruction netted in (as elsewhere)
    dlam = np.gradient(wave_A)
    denom = photon_flux * dlam * a_eff_cm2 * exptime * t_atm * f_slit
    t_sys = counts * gain / denom

    idx = np.digitize(wave_A, BIN_EDGES)
    print("\nwavelength[A]  T_sys (median in bin)")
    for i in range(1, len(BIN_EDGES)):
        mask = idx == i
        if mask.sum() > 3:
            centre = 0.5 * (BIN_EDGES[i - 1] + BIN_EDGES[i])
            print(f"  {centre:8.0f}   {np.median(t_sys[mask]):.3f}")


if __name__ == "__main__":
    main()
