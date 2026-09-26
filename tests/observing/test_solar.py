import pytest

from kino.observing import NOAA_SWPC_F107_PREDICTION, solar_flux


class TestSolarFlux:
    def test_lookup_accepts_month_or_full_date(self):
        assert solar_flux("2027-03") == solar_flux("2027-03-14")

    def test_range_brackets_the_prediction_everywhere(self):
        for month, (pred, low, high) in NOAA_SWPC_F107_PREDICTION.items():
            assert low < pred < high, month

    def test_flux_declines_through_the_2027a_semester(self):
        feb, jul = solar_flux("2027-02")[0], solar_flux("2027-07")[0]
        assert feb > jul

    def test_values_are_within_the_range_the_sky_library_covers(self):
        for _, (_pred, low, high) in NOAA_SWPC_F107_PREDICTION.items():
            assert low >= 70.0 and high <= 200.0

    def test_outside_the_table_raises_with_a_hint(self):
        with pytest.raises(KeyError, match="pass solar_flux"):
            solar_flux("2031-01")
