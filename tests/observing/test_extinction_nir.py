import astropy.units as u
import numpy as np
import pytest

from kino.observing.extinction_nir import (
    NIR_EXTINCTION_BANDS,
    NIR_EXTINCTION_FROGEL,
    nir_atmospheric_transmission,
    nir_extinction_coefficient,
)


class TestBandTable:
    def test_three_bands_in_j_h_ks_order(self):
        assert [b.name for b in NIR_EXTINCTION_BANDS] == ["J", "H", "Ks"]
        assert np.all(np.diff([b.center_um for b in NIR_EXTINCTION_BANDS]) > 0)

    def test_frogel_comparison_table_has_the_same_three_bands(self):
        assert set(NIR_EXTINCTION_FROGEL) == {"J", "H", "Ks"}

    def test_frogel_is_higher_than_the_primary_2mass_value_in_every_band(self):
        # a real, documented literature disagreement -- see the module docstring
        for band in NIR_EXTINCTION_BANDS:
            frogel_k, _ = NIR_EXTINCTION_FROGEL[band.name]
            assert frogel_k > band.mag_per_airmass


class TestCoefficient:
    def test_at_isophotal_centres(self):
        assert nir_extinction_coefficient(1.235 * u.um)[0] == pytest.approx(0.092)
        assert nir_extinction_coefficient(1.662 * u.um)[0] == pytest.approx(0.031)
        assert nir_extinction_coefficient(2.159 * u.um)[0] == pytest.approx(0.065)

    def test_flat_within_one_band_not_interpolated_across_bands(self):
        # two wavelengths inside the same (J) window give the identical value --
        # this is a broadband mean, not a continuous curve
        lo = nir_extinction_coefficient(1.16 * u.um)[0]
        hi = nir_extinction_coefficient(1.31 * u.um)[0]
        assert lo == hi == pytest.approx(0.092)

    def test_gap_between_bands_raises_rather_than_interpolating(self):
        # 1.45 um sits between the J and H windows (in the telluric water band) --
        # must not be silently interpolated across
        with pytest.raises(ValueError, match="outside every covered"):
            nir_extinction_coefficient(1.45 * u.um)

    def test_far_outside_every_band_raises(self):
        with pytest.raises(ValueError):
            nir_extinction_coefficient(0.5 * u.um)
        with pytest.raises(ValueError):
            nir_extinction_coefficient(3.5 * u.um)

    def test_gmtifs_demo_window_is_not_covered(self):
        # kino/config/gmtifs.yaml's own 1.10-1.14 um window falls just short of
        # the J band's lower edge (1.154 um) -- see the module docstring
        with pytest.raises(ValueError):
            nir_extinction_coefficient(1.12 * u.um)

    def test_vectorised(self):
        k = nir_extinction_coefficient(np.array([1.235, 1.662, 2.159]) * u.um)
        assert k == pytest.approx([0.092, 0.031, 0.065])


class TestTransmission:
    def test_matches_the_mag_formula(self):
        t = nir_atmospheric_transmission(1.235 * u.um, 1.0)[0]
        assert t == pytest.approx(10 ** (-0.4 * 0.092))

    def test_decreases_with_airmass(self):
        t = nir_atmospheric_transmission(1.662 * u.um, np.array([1.0, 1.5, 2.0]))
        assert np.all(np.diff(t) < 0)

    def test_infinite_airmass_gives_zero_transmission(self):
        assert nir_atmospheric_transmission(2.159 * u.um, np.inf)[0] == 0.0

    def test_rejects_airmass_below_one(self):
        with pytest.raises(ValueError):
            nir_atmospheric_transmission(1.235 * u.um, 0.9)

    def test_less_extinction_than_optical_at_every_band_centre(self):
        # NIR extinction is real but much weaker than optical (CLAUDE.md's
        # existing optical curve peaks near 1 mag/airmass in the blue)
        for band in NIR_EXTINCTION_BANDS:
            assert band.mag_per_airmass < 0.14  # weaker than CTIO's own V-band value
