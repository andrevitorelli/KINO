"""Reader for MaNGIA (TNG50 mock MaNGA) FITS cubes -> kino.io.base.Datacube.

**Status: unverified against a real file.** Written from the documented
output-file structure in Sarmiento et al. (2022) / the
``reginasar/TNG_MaNGA_mocks`` code (Primary HDU = flux datacube, extensions
``ERROR``/``MASK``/``GAS``), *without* an actual MaNGIA cube on hand to test
against -- this session had no IllustrisTNG API key (needed to browse
TNG50-1's MaNGIA catalog; see ``kino.io.download``'s module docstring), and
TEST_DATA.md Section 5's own rule is "verify against the file," which this
reader could not yet do. Every assumption below is a documented, overridable
default, not a hard-coded fact:

- **Wavelength axis**: tries a ``WAVE`` extension array first (most FITS
  cube conventions store this explicitly, since MaNGA-like spectra are not
  perfectly log/linear); falls back to a linear grid from the primary
  header's ``CRVAL3``/``CDELT3``/``CRPIX3``; falls back to the *documented*
  default grid in the mock code, ``3749-10352 Angstrom`` in 1 Angstrom
  steps, only if its length matches the cube's spectral axis.
- **Spatial pixel scale**: read from the primary header's ``CDELT1``/``CDELT2``
  (assumed degrees, per the FITS standard) if present; else falls back to
  MaNGA's real, well-documented instrument sampling, 0.5 arcsec/spaxel.
- **Flux unit**: real MaNGA DRP cubes use ``1e-17 erg/s/cm^2/Angstrom`` per
  spaxel; assumed here since MaNGIA is built to mimic that pipeline, but not
  confirmed for MaNGIA's own output.
- **``MASK`` polarity**: documented only as "valid mask array" -- assumed
  ``True`` = valid pixel (the ``pyPipe3D`` convention it says the mask is
  *for*), so it's inverted before being stored on ``Datacube.mask`` (which is
  ``True`` = invalid, see ``kino.io.base``).
- **Position angle / orientation**: MaNGIA's mock views are built at chosen
  synthetic viewing angles (a ``view``/``thet`` parameter in the generator
  code) rather than encoded as an on-sky WCS rotation the way a real
  astronomical PA would be -- this reader does **not** attempt to derive
  KINO's East-of-North ``pa`` convention from the file. Any orientation
  analysis needs to be pinned down against a real file first (``CLAUDE.md``
  principle 1: a wrong convention is worse than a missing feature).

Pass every keyword explicitly, or fetch a real cube (``kino.io.download``)
and inspect its actual header before trusting the defaults for anything
beyond exercising the read -> ``Datacube`` -> visualise pipeline.
"""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits

from kino.io.base import Datacube

#: MaNGA's real, well-documented spaxel scale -- used only as a fallback
#: when the file's own WCS doesn't specify one.
DEFAULT_PIXEL_SCALE = 0.5 * u.arcsec

#: MaNGA DRP's real flux-unit convention -- assumed, not confirmed for
#: MaNGIA's own output (see module docstring).
DEFAULT_FLUX_UNIT = 1e-17 * u.erg / u.s / u.cm**2 / u.AA

#: The mock-generator code's own documented default wavelength grid
#: (start, stop, step), Angstrom -- last-resort fallback only.
_DOCUMENTED_DEFAULT_WAVE_GRID = (3749.0, 10352.0, 1.0)

_REDSHIFT_KEYWORDS = ("REDSHIFT", "Z", "ZREDSHIFT")


def _wavelength_axis(
    header: fits.Header, wave_hdu: fits.ImageHDU | None, n_spec: int
) -> u.Quantity:
    if wave_hdu is not None:
        wave = np.asarray(wave_hdu.data, dtype=float)
        if wave.shape != (n_spec,):
            raise ValueError(f"WAVE extension shape {wave.shape} does not match n_spec={n_spec}")
        return wave * u.AA

    if all(k in header for k in ("CRVAL3", "CDELT3", "CRPIX3")):
        crval, cdelt, crpix = header["CRVAL3"], header["CDELT3"], header["CRPIX3"]
        return (crval + (np.arange(n_spec) + 1 - crpix) * cdelt) * u.AA

    start, stop, step = _DOCUMENTED_DEFAULT_WAVE_GRID
    fallback = np.arange(start, stop, step)
    if fallback.shape != (n_spec,):
        raise ValueError(
            "Could not determine the wavelength axis: no WAVE extension, no CRVAL3/"
            f"CDELT3/CRPIX3 in the primary header, and the documented default grid "
            f"({start}-{stop} Angstrom, step {step}) has length {len(fallback)}, not "
            f"n_spec={n_spec}. Pass the wavelength axis in yourself, or inspect this "
            "file's actual header (see this module's docstring)."
        )
    return fallback * u.AA


def _pixel_scale(header: fits.Header) -> u.Quantity:
    if "CDELT1" in header and "CDELT2" in header:
        scale = 0.5 * (abs(header["CDELT1"]) + abs(header["CDELT2"])) * u.deg
        return scale.to(u.arcsec)
    return DEFAULT_PIXEL_SCALE


def _redshift(header: fits.Header) -> float:
    for key in _REDSHIFT_KEYWORDS:
        if key in header:
            return float(header[key])
    return 0.0


def read_mangia_cube(
    path: str | Path,
    *,
    pixel_scale: u.Quantity | None = None,
    flux_unit: u.UnitBase | None = None,
    redshift: float | None = None,
    include_gas: bool = False,
) -> Datacube:
    """Read a MaNGIA mock-MaNGA cube FITS file into a ``Datacube``.

    See the module docstring for which conventions are read from the file
    vs. assumed defaults, and their unverified status.

    Parameters
    ----------
    path : str or Path
    pixel_scale, flux_unit, redshift : optional
        Override the value read from (or defaulted for) the file.
    include_gas : bool, optional
        If the file has a ``GAS`` extension, stash it in
        ``Datacube.meta["gas_flux"]`` (not merged into the main ``flux``
        array, which is the total stars+gas cube).

    Returns
    -------
    Datacube
    """
    with fits.open(path) as hdul:
        primary = hdul[0]
        flux = np.asarray(primary.data, dtype=float)
        header = primary.header
        if flux.ndim != 3:
            raise ValueError(
                f"expected a 3-D primary HDU (n_spec, n_y, n_x), got shape {flux.shape}"
            )
        n_spec = flux.shape[0]

        variance = None
        if "ERROR" in hdul:
            variance = np.asarray(hdul["ERROR"].data, dtype=float) ** 2

        mask = None
        if "MASK" in hdul:
            valid = np.asarray(hdul["MASK"].data).astype(bool)
            mask = ~valid  # see module docstring: MASK is documented as "valid", inverted here

        wave_hdu = hdul["WAVE"] if "WAVE" in hdul else None  # noqa: SIM401 -- HDUList has no .get()
        wavelength = _wavelength_axis(header, wave_hdu, n_spec)

        meta = {
            "source": "MaNGIA (Sarmiento et al. 2022)",
            "citation": "Sarmiento et al. 2022",
            "path": str(path),
            "header": dict(header),
            "unverified": True,
        }
        if include_gas and "GAS" in hdul:
            meta["gas_flux"] = np.asarray(hdul["GAS"].data, dtype=float)

        return Datacube(
            flux=flux,
            spectral_axis=wavelength,
            spectral_kind="wavelength",
            pixel_scale=pixel_scale if pixel_scale is not None else _pixel_scale(header),
            redshift=redshift if redshift is not None else _redshift(header),
            flux_unit=flux_unit if flux_unit is not None else DEFAULT_FLUX_UNIT,
            variance=variance,
            mask=mask,
            meta=meta,
        )
