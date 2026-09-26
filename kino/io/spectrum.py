"""1-D galaxy spectra from tables, and emission-line flux measurement.

A forecast needs the galaxy's line flux and redshift. Spectra come from
public archives (Legacy Survey/DESI/SDSS, southern surveys, ...) as ASCII or
FITS tables in differing conventions, so the reader is deliberately strict
about **units**: a flux column with no unit in the file or in the call raises
instead of guessing (SDSS/DESI's ``1e-17 erg/s/cm2/A`` convention is a classic
silent factor-of-1e17 error).

Wavelengths are treated as **vacuum**; SDSS and DESI spectra are vacuum,
many other sources quote air wavelengths -- convert first (the ~2 A shift at
H-alpha is irrelevant to the line flux but matters for sky-line overlap).

The measured flux is *what the spectrograph aperture (a fibre, a slit)
collected*, not the galaxy's total line flux. Pass ``aperture_correction`` when
you know the fraction of the galaxy inside that aperture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table
from scipy.optimize import curve_fit

C_KMS = 299792.458

#: Rest-frame **vacuum** wavelengths (Angstrom) of the usual lines (the SDSS line list).
LINES = {
    "Halpha": 6564.61,
    "Hbeta": 4862.68,
    "OIII5007": 5008.24,
    "NII6584": 6585.27,
    "OII3727": 3727.09,  # blue member of an unresolved doublet (3727.09 + 3729.88)
}

_WAVE_NAMES = ("wavelength", "wave", "lam", "lambda", "wl")
_FLUX_NAMES = ("flux", "f_lambda", "flam", "fl")
_ERR_NAMES = ("error", "err", "flux_err", "fluxerr", "sigma", "e_flux")


@dataclass(frozen=True)
class Spectrum1D:
    """A 1-D spectrum.

    Attributes
    ----------
    wavelength : Quantity (length), vacuum
    flux : Quantity (energy / time / area / wavelength)
    error : Quantity or None
        1-sigma flux uncertainty, same unit as ``flux``.
    meta : dict
    """

    wavelength: u.Quantity
    flux: u.Quantity
    error: u.Quantity | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.wavelength.shape != self.flux.shape or self.wavelength.ndim != 1:
            raise ValueError("wavelength and flux must be 1-D and the same length")
        if self.error is not None and self.error.shape != self.flux.shape:
            raise ValueError("error must match flux")
        if np.any(np.diff(self.wavelength.value) <= 0):
            raise ValueError("wavelength must be strictly increasing")


def _find(table: Table, names, explicit):
    if explicit is not None:
        if explicit not in table.colnames:
            raise ValueError(f"column {explicit!r} not found; have {table.colnames}")
        return explicit
    lower = {c.lower(): c for c in table.colnames}
    for n in names:
        if n in lower:
            return lower[n]
    return None


def read_spectrum(
    path: str | Path,
    *,
    wavelength_col: str | None = None,
    flux_col: str | None = None,
    error_col: str | None = None,
    wavelength_unit: u.UnitBase | str | None = None,
    flux_unit: u.UnitBase | str | None = None,
    hdu: int | str = 1,
    format: str | None = None,  # noqa: A002 -- astropy's own keyword
) -> Spectrum1D:
    """Read a spectrum from an ASCII or FITS binary table.

    Columns are found by common names (wavelength: ``wavelength/wave/lam``;
    flux: ``flux``; error: ``error/err/flux_err`` or ``ivar``) unless named.
    ``loglam`` (SDSS, log10 of the wavelength in Angstrom) is converted.
    ``wavelength_unit`` and ``flux_unit`` are used when the file carries no
    unit; **a missing flux unit raises** rather than defaulting.

    ``.fits``/``.fit`` files are read with ``hdu``; anything else through
    ``astropy.io.ascii`` (whitespace, CSV, ...; ``#`` comment lines are fine).
    A header-less two- or three-column text file is read as
    ``wavelength flux [error]``.
    """
    path = Path(path)
    if path.suffix.lower() in (".fits", ".fit", ".fits.gz"):
        table = Table.read(path, hdu=hdu)
    else:
        table = Table.read(path, format=format or "ascii")
        if all(c.startswith("col") for c in table.colnames):
            table.rename_columns(
                table.colnames, ["wavelength", "flux", "error"][: len(table.colnames)]
            )
    lam_name = _find(table, _WAVE_NAMES, wavelength_col)
    lam_is_log = False
    if lam_name is None and "loglam" in {c.lower() for c in table.colnames}:
        lam_name = next(c for c in table.colnames if c.lower() == "loglam")
        lam_is_log = True
    flux_name = _find(table, _FLUX_NAMES, flux_col)
    if lam_name is None or flux_name is None:
        raise ValueError(f"cannot identify wavelength/flux columns in {table.colnames}")

    lam = np.asarray(table[lam_name], dtype=float)
    if lam_is_log:
        lam, w_unit = 10.0**lam, u.AA
    else:
        w_unit = table[lam_name].unit or (
            None if wavelength_unit is None else u.Unit(wavelength_unit)
        )
        if w_unit is None:
            raise ValueError("wavelength has no unit: pass wavelength_unit (e.g. 'Angstrom')")
    f_unit = table[flux_name].unit or (None if flux_unit is None else u.Unit(flux_unit))
    if f_unit is None:
        raise ValueError(
            "flux has no unit in the file: pass flux_unit explicitly "
            "(e.g. 1e-17 * u.erg / u.s / u.cm**2 / u.AA for SDSS/DESI)"
        )
    if flux_unit is not None and table[flux_name].unit is not None and f_unit != u.Unit(flux_unit):
        raise ValueError(f"file says {f_unit}, call says {flux_unit}: refusing to guess")
    flux = np.asarray(table[flux_name], dtype=float) * f_unit

    err = None
    err_name = _find(table, _ERR_NAMES, error_col)
    if err_name is not None:
        e_unit = table[err_name].unit or f_unit
        err = np.asarray(table[err_name], dtype=float) * e_unit
    elif "ivar" in {c.lower() for c in table.colnames}:
        ivar = np.asarray(
            table[next(c for c in table.colnames if c.lower() == "ivar")], dtype=float
        )
        with np.errstate(divide="ignore"):
            err = np.where(ivar > 0, 1.0 / np.sqrt(ivar), np.inf) * f_unit

    good = np.isfinite(lam) & np.isfinite(flux.value)
    order = np.argsort(lam[good])
    lam_q = (lam[good][order] * w_unit).to(u.AA)
    return Spectrum1D(
        wavelength=lam_q,
        flux=flux[good][order],
        error=None if err is None else err[good][order],
        meta={"source": str(path)},
    )


@dataclass(frozen=True)
class LineMeasurement:
    """Result of :func:`measure_line_flux`.

    Attributes
    ----------
    flux, flux_err : Quantity (erg / s / cm^2)
        Integrated line flux and 1-sigma uncertainty (aperture-corrected).
    center : Quantity (length)
        Fitted observed line centre.
    velocity_offset : Quantity (km/s)
        Centre relative to the expected ``rest * (1 + z)``.
    sigma_velocity : Quantity (km/s)
        Fitted Gaussian sigma (observed-frame, includes instrumental broadening).
    snr : float
    """

    flux: u.Quantity
    flux_err: u.Quantity
    center: u.Quantity
    velocity_offset: u.Quantity
    sigma_velocity: u.Quantity
    snr: float


def _gauss_line(lam, amp, mu, sigma, c0, c1):
    return amp * np.exp(-0.5 * ((lam - mu) / sigma) ** 2) + c0 + c1 * (lam - mu)


def measure_line_flux(
    spectrum: Spectrum1D,
    redshift: float,
    line: str | u.Quantity = "Halpha",
    *,
    half_width: u.Quantity = 1500.0 * u.km / u.s,
    aperture_correction: float = 1.0,
) -> LineMeasurement:
    """Integrated flux of one emission line: Gaussian plus linear continuum.

    Fits ``amp * exp(-(lam - mu)^2 / 2 sigma^2) + c0 + c1 (lam - mu)`` in
    ``+/- half_width`` (velocity) around the expected observed wavelength and
    integrates the Gaussian: ``F = amp * sigma * sqrt(2 pi)``. Uses the
    spectrum's errors if present, otherwise the scatter of the residuals.

    ``aperture_correction`` (>= 1) multiplies the flux to estimate the whole
    galaxy's line flux from what the aperture collected; 1 leaves it as
    measured. The result is *not* corrected for Milky Way or internal dust.
    """
    rest = LINES[line] * u.AA if isinstance(line, str) else u.Quantity(line).to(u.AA)
    if aperture_correction < 1.0:
        raise ValueError("aperture_correction must be >= 1")
    expected = rest * (1.0 + redshift)
    dv = (half_width / (C_KMS * u.km / u.s)).decompose().value
    lo, hi = expected * (1 - dv), expected * (1 + dv)
    lam_all = spectrum.wavelength.to_value(u.AA)
    sel = (lam_all >= lo.value) & (lam_all <= hi.value)
    if sel.sum() < 8:
        raise ValueError(
            f"only {int(sel.sum())} spectrum pixels within +/-{half_width} of the expected "
            f"line at {expected:.1f} -- is the line inside the spectrum's coverage?"
        )
    lam = lam_all[sel]
    f_unit = spectrum.flux.unit
    y = spectrum.flux.value[sel]
    sigma_y = None
    if spectrum.error is not None:
        e = spectrum.error.to_value(f_unit)[sel]
        if np.all(np.isfinite(e)) and np.all(e > 0):
            sigma_y = e

    pix = float(np.median(np.diff(lam)))
    sig0 = max(1.0 * pix, expected.value * 100.0 / C_KMS)
    c0 = float(np.median(y))
    amp0 = float(np.max(y) - c0)
    p0 = [amp0, expected.value, sig0, c0, 0.0]
    try:
        popt, pcov = curve_fit(
            _gauss_line, lam, y, p0=p0, sigma=sigma_y, absolute_sigma=sigma_y is not None,
            maxfev=20000,
        )  # fmt: skip
    except RuntimeError as exc:
        raise ValueError(f"line fit did not converge: {exc}") from exc
    amp, mu, sig = popt[0], popt[1], abs(popt[2])
    # with sigma=None curve_fit already scales pcov by the residual variance
    # F = amp * sigma * sqrt(2 pi); propagate including the amp-sigma covariance
    grad = np.array([sig, amp]) * np.sqrt(2 * np.pi)
    cov2 = np.array([[pcov[0, 0], pcov[0, 2]], [pcov[2, 0], pcov[2, 2]]])
    var_f = float(grad @ cov2 @ grad)
    flux = amp * sig * np.sqrt(2 * np.pi) * f_unit * u.AA
    flux_err = np.sqrt(max(var_f, 0.0)) * f_unit * u.AA
    unit = u.erg / u.s / u.cm**2
    f = (flux * aperture_correction).to(unit)
    fe = (flux_err * aperture_correction).to(unit)
    return LineMeasurement(
        flux=f,
        flux_err=fe,
        center=mu * u.AA,
        velocity_offset=(mu - expected.value) / expected.value * C_KMS * u.km / u.s,
        sigma_velocity=sig / mu * C_KMS * u.km / u.s,
        snr=float(abs(f.value) / fe.value) if fe.value > 0 else float("inf"),
    )
