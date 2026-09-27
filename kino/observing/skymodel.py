"""Night-sky radiance spectra from a local library of ESO SkyCalc output.

``SkyLibrary`` interpolates, offline, spectra of the four sky components that
matter at optical wavelengths -- scattered moonlight, airglow (emission lines
plus residual continuum), zodiacal light and scattered starlight -- computed
with ESO's Cerro Paranal Advanced Sky Model (Noll et al. 2012, A&A 543, A92;
Jones et al. 2013, A&A 560, A91). The library is built once from the SkyCalc
service by ``scripts/build_skycalc_library.py`` (network, ~10 minutes) and kept
in ``data/skycalc/`` (git-ignored), so batch runs are offline and reproducible.

Radiance is in **photons / s / m^2 / micron / arcsec^2** at the ground (it
already includes the sky's own extinction); wavelengths are **vacuum**, in nm.

What was checked against independent references (V band, see the tests and
``CLAUDE.md``): the dark zenith sky is 21.9 mag/arcsec^2 at 100 s.f.u. against
the CTIO mean 21.79 +/- 0.22 (Krisciunas et al. 2007), and the moonlight
brightening agrees with Krisciunas & Schaefer (1991) to 0.1-0.4 mag across four
geometries.

Approximations and known limits -- read before trusting a number:

* the model is for Cerro *Paranal* (2640 m); Cerro Pachon (SOAR/Gemini South)
  has more water vapour (irrelevant below ~1 micron), different aerosols, and
  La Serena's light pollution at low elevation, none of which is modelled;
* airglow is averaged over seasons and the whole night, although it varies by
  tens of percent within a night; it scales with the 10.7 cm solar flux, which
  the caller must supply for the epoch (there is no default you can trust);
* moonlight is interpolated (log-linear) on a grid in phase, target-Moon
  separation, Moon zenith distance and airmass, capped at airmass 2 and at a
  Moon zenith distance of 80 deg with a linear taper to zero at the horizon;
* zodiacal light and starlight are stored at airmass 1 and scaled by the CTIO
  extinction curve; the ecliptic longitude is measured relative to the Sun and
  differs by ~4 deg from SkyCalc's own almanac for one tested target, which
  matters little far from the Sun;
* wavelengths are vacuum -- convert if you compare sky lines with air
  wavelengths.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from kino.observing.extinction import extinction_coefficient

DEFAULT_LIBRARY_PATH = Path("data/skycalc")

#: Limits of the airmass axis of the moonlight grid; beyond it the model clips.
MAX_MOON_AIRMASS = 2.0
_MIN_LAMBDA_NM = 320.0  # the CTIO extinction curve starts at 3200 A
_MOON_ZM_MAX = 80.0


@dataclass(frozen=True)
class SkyConditions:
    """Everything the sky spectrum depends on.

    Attributes
    ----------
    airmass : float
        Target airmass (>= 1).
    solar_flux : float
        Monthly-mean 10.7 cm solar radio flux in solar flux units. Drives the
        airglow level (the continuum rises ~+75% from 100 to 200 s.f.u.).
    moon_phase_angle : Quantity (angle)
        Lunar phase angle; 0 = full, 180 deg = new
        (``kino.observing.MoonState.phase_angle``).
    moon_target_separation : Quantity (angle)
    moon_zenith_distance : Quantity (angle)
        ``>= 90 deg`` means the Moon is down and contributes nothing.
    ecliptic_lon_from_sun, ecliptic_lat : Quantity (angle)
        Target ecliptic longitude relative to the Sun (either sign) and
        latitude, for the zodiacal light.
    """

    airmass: float
    solar_flux: float
    moon_phase_angle: u.Quantity = 180.0 * u.deg
    moon_target_separation: u.Quantity = 90.0 * u.deg
    moon_zenith_distance: u.Quantity = 120.0 * u.deg
    ecliptic_lon_from_sun: u.Quantity = 120.0 * u.deg
    ecliptic_lat: u.Quantity = 45.0 * u.deg


@dataclass(frozen=True)
class SkySpectrum:
    """Sky radiance averaged over wavelength channels.

    Attributes
    ----------
    wavelength_edges : Quantity (length), shape (n + 1,)
    components : dict[str, numpy.ndarray]
        ``moon``, ``airglow``, ``zodiacal``, ``starlight``: channel-mean
        radiance, ph/s/m^2/micron/arcsec^2, shape ``(n,)``.
    """

    wavelength_edges: u.Quantity
    components: dict[str, np.ndarray]

    @property
    def total(self) -> np.ndarray:
        return sum(self.components.values())

    @property
    def wavelength(self) -> u.Quantity:
        e = self.wavelength_edges
        return 0.5 * (e[:-1] + e[1:])

    @property
    def channel_width(self) -> u.Quantity:
        return np.diff(self.wavelength_edges)


def bin_to_channels(lam_nm: np.ndarray, radiance: np.ndarray, edges_nm: np.ndarray) -> np.ndarray:
    """Mean of ``radiance(lam)`` over each channel ``[edges[i], edges[i+1]]``.

    Integrates the piecewise-linear spectrum exactly via its cumulative
    integral, so narrow sky lines contribute their true flux to a wide channel
    instead of being sampled or missed. Channels partly outside the library
    range are averaged over the covered part; fully outside raises.
    """
    edges = np.asarray(edges_nm, dtype=float)
    if edges[0] < lam_nm[0] or edges[-1] > lam_nm[-1]:
        raise ValueError(
            f"channels {edges[0]:.1f}-{edges[-1]:.1f} nm extend outside the sky library "
            f"({lam_nm[0]:.1f}-{lam_nm[-1]:.1f} nm)"
        )
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (radiance[1:] + radiance[:-1]) * np.diff(lam_nm))])
    at_edges = np.interp(edges, lam_nm, cum)
    return np.diff(at_edges) / np.diff(edges)


class SkyLibrary:
    """Offline sky-radiance spectra interpolated from SkyCalc output."""

    def __init__(self, moon: dict, airglow: dict, zodi: dict, meta: dict | None = None):
        self.meta = meta or {}
        self._moon_lam = np.asarray(moon["lam_nm"], dtype=float)
        self._moon_axes = (
            np.asarray(moon["elongation"], dtype=float),
            np.asarray(moon["target_sep"], dtype=float),
            np.asarray(moon["zenith_dist_moon"], dtype=float),
            np.asarray(moon["airmass"], dtype=float),
        )
        # log-linear interpolation: moonlight spans orders of magnitude
        self._moon = RegularGridInterpolator(
            self._moon_axes,
            np.log10(np.maximum(np.asarray(moon["radiance"], dtype=float), 1e-6)),
            bounds_error=False,
            fill_value=None,
        )
        self._glow_lam = np.asarray(airglow["lam_nm"], dtype=float)
        self._glow = RegularGridInterpolator(
            (
                np.asarray(airglow["airmass"], dtype=float),
                np.asarray(airglow["solar_flux"], dtype=float),
            ),
            np.asarray(airglow["radiance"], dtype=float),
            bounds_error=False,
            fill_value=None,
        )
        self._zodi_lam = np.asarray(zodi["lam_nm"], dtype=float)
        self._zodi = RegularGridInterpolator(
            (np.asarray(zodi["ecl_lon"], dtype=float), np.asarray(zodi["ecl_lat"], dtype=float)),
            np.asarray(zodi["radiance"], dtype=float),
            bounds_error=False,
            fill_value=None,
        )
        self._starlight = np.asarray(zodi["starlight"], dtype=float)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_LIBRARY_PATH) -> SkyLibrary:
        """Load the library built by ``scripts/build_skycalc_library.py``."""
        path = Path(path)
        try:
            moon = dict(np.load(path / "moon.npz"))
            airglow = dict(np.load(path / "airglow.npz"))
            zodi = dict(np.load(path / "zodi.npz"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"no sky library in {path}/ -- build it once (needs network and "
                "`pip install skycalc-cli`): python scripts/build_skycalc_library.py"
            ) from exc
        meta = {}
        if (path / "meta.json").exists():
            import json

            meta = json.loads((path / "meta.json").read_text())
        return cls(moon, airglow, zodi, meta)

    @property
    def wavelength_range(self) -> tuple[u.Quantity, u.Quantity]:
        """Wavelength range every component covers, as ``(min, max)`` in nm."""
        lo = max(self._moon_lam[0], self._glow_lam[0], self._zodi_lam[0])
        hi = min(self._moon_lam[-1], self._glow_lam[-1], self._zodi_lam[-1])
        return max(lo, _MIN_LAMBDA_NM) * u.nm, hi * u.nm

    # -- components (each on its native grid) ---------------------------------

    def _extinction_scale(self, lam_nm: np.ndarray, airmass: float) -> np.ndarray:
        """``10^(-0.4 k (X - 1))`` for components stored at airmass 1."""
        # the CTIO table spans 320-1087 nm; hold its end values beyond that
        k = extinction_coefficient(np.clip(lam_nm, _MIN_LAMBDA_NM, 1087.0) * u.nm)
        return 10.0 ** (-0.4 * k * (airmass - 1.0))

    def moonlight(self, c: SkyConditions) -> np.ndarray:
        zm = float(c.moon_zenith_distance.to_value(u.deg))
        if zm >= 90.0:
            return np.zeros_like(self._moon_lam)
        x = float(c.airmass)
        if x > MAX_MOON_AIRMASS:
            warnings.warn(
                f"airmass {x:.2f} is beyond the moonlight grid (max {MAX_MOON_AIRMASS}); clipping",
                UserWarning,
                stacklevel=3,
            )
        x = min(x, MAX_MOON_AIRMASS)
        z = float(np.degrees(np.arccos(1.0 / x)))
        zm_eff = min(zm, _MOON_ZM_MAX)
        rho = float(c.moon_target_separation.to_value(u.deg))
        rho = float(np.clip(rho, abs(z - zm_eff), z + zm_eff))
        elong = 180.0 - float(c.moon_phase_angle.to_value(u.deg))
        axes = self._moon_axes
        point = (
            np.clip(elong, axes[0][0], axes[0][-1]),
            np.clip(rho, axes[1][0], axes[1][-1]),
            np.clip(zm_eff, axes[2][0], axes[2][-1]),
            np.clip(x, axes[3][0], axes[3][-1]),
        )
        rad = 10.0 ** self._moon(np.array([point]))[0]
        # the grid ends at zm = 80 deg: taper linearly to zero at the horizon
        taper = 1.0 if zm <= _MOON_ZM_MAX else (90.0 - zm) / (90.0 - _MOON_ZM_MAX)
        return rad * taper

    def airglow(self, c: SkyConditions) -> np.ndarray:
        axes = self._glow.grid
        point = (
            np.clip(c.airmass, axes[0][0], axes[0][-1]),
            np.clip(c.solar_flux, axes[1][0], axes[1][-1]),
        )
        return np.maximum(self._glow(np.array([point]))[0], 0.0)

    def zodiacal(self, c: SkyConditions) -> np.ndarray:
        axes = self._zodi.grid
        lon = abs(float(c.ecliptic_lon_from_sun.to_value(u.deg)))
        lat = abs(float(c.ecliptic_lat.to_value(u.deg)))
        point = (np.clip(lon, axes[0][0], axes[0][-1]), np.clip(lat, axes[1][0], axes[1][-1]))
        rad = np.maximum(self._zodi(np.array([point]))[0], 0.0)
        return rad * self._extinction_scale(self._zodi_lam, c.airmass)

    def starlight(self, c: SkyConditions) -> np.ndarray:
        return self._starlight * self._extinction_scale(self._zodi_lam, c.airmass)

    # -- public ---------------------------------------------------------------

    def spectrum(self, conditions: SkyConditions, wavelength_edges: u.Quantity) -> SkySpectrum:
        """Sky components averaged over the channels bounded by ``wavelength_edges``.

        ``wavelength_edges`` are *vacuum* wavelengths, shape ``(n + 1,)``.
        """
        if conditions.airmass < 1.0:
            raise ValueError("airmass must be >= 1")
        edges_nm = u.Quantity(wavelength_edges).to_value(u.nm)
        if edges_nm[0] < _MIN_LAMBDA_NM:
            raise ValueError(f"the sky model is used above {_MIN_LAMBDA_NM:.0f} nm")
        return SkySpectrum(
            wavelength_edges=u.Quantity(wavelength_edges).to(u.nm),
            components={
                "moon": bin_to_channels(self._moon_lam, self.moonlight(conditions), edges_nm),
                "airglow": bin_to_channels(self._glow_lam, self.airglow(conditions), edges_nm),
                "zodiacal": bin_to_channels(self._zodi_lam, self.zodiacal(conditions), edges_nm),
                "starlight": bin_to_channels(self._zodi_lam, self.starlight(conditions), edges_nm),
            },
        )


def sky_electron_rate(
    sky: SkySpectrum,
    *,
    pixel_area: u.Quantity,
    effective_area: u.Quantity,
    throughput,
) -> u.Quantity:
    """Detected sky electrons per second per pixel per channel.

    ``rate = radiance * channel_width * pixel_area * effective_area * throughput``
    -- the same collecting area and end-to-end throughput
    :func:`kino.instruments.base.photon_flux_to_electrons` applies to the
    source, so source and sky are on one footing. ``throughput`` is a
    :class:`~kino.instruments.base.Throughput`, evaluated at the channel
    centres. Suitable as ``IFU.sky_electron_rate`` (which multiplies by the
    total exposure time itself).
    """
    radiance = sky.total * (u.photon / u.s / u.m**2 / u.um / u.arcsec**2)
    photons = radiance * sky.channel_width.to(u.um) * pixel_area.to(u.arcsec**2) * effective_area
    rate = photons.to(u.photon / u.s) * throughput(sky.wavelength)
    return rate.value * (u.electron / u.s)


def ifu_sky_electron_rate(
    ifu,
    conditions: SkyConditions,
    library: SkyLibrary,
) -> u.Quantity:
    """Per-channel sky electron rate for an :class:`~kino.instruments.ifu.IFU`.

    Returns a ``(n_wave,)`` array to assign to ``IFU.sky_electron_rate`` (e.g.
    with ``dataclasses.replace``). The IFU's channel edges are taken as vacuum
    wavelengths.
    """
    edges = np.linspace(ifu.wavelength_min, ifu.wavelength_max, ifu.n_wave + 1)
    sky = library.spectrum(conditions, edges)
    return sky_electron_rate(
        sky,
        pixel_area=ifu.pixel_scale**2,
        effective_area=ifu.effective_area,
        throughput=ifu.throughput,
    )


def slit_sky_electron_rate(
    slit,
    conditions: SkyConditions,
    library: SkyLibrary,
) -> u.Quantity:
    """Per-channel sky electron rate for a :class:`~kino.instruments.slit.SlitSpectrograph`.

    Returns a ``(n_wave,)`` array to assign to ``SlitSpectrograph.sky_electron_rate``
    (e.g. with ``dataclasses.replace``) -- a rate *per along-slit pixel*, matching that
    field's own convention (``CLAUDE.md``'s "Slit geometry and losses"). The sky enters
    through the slit's full width and is sampled along its length in ``pixel_scale``
    steps, so the relevant sky-collecting area per along-slit pixel is
    ``pixel_scale * slit_width``, not ``pixel_scale ** 2`` as for an IFU's square
    spaxel. The slit's channel edges are taken as vacuum wavelengths.
    """
    edges = np.linspace(slit.wavelength_min, slit.wavelength_max, slit.n_wave + 1)
    sky = library.spectrum(conditions, edges)
    return sky_electron_rate(
        sky,
        pixel_area=slit.pixel_scale * slit.slit_width,
        effective_area=slit.effective_area,
        throughput=slit.throughput,
    )
