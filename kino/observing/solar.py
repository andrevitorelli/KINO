"""Solar 10.7 cm radio flux (F10.7) for the airglow level.

Airglow -- the dominant dark-sky term in the red and near-infrared -- rises
with solar activity (the continuum grows by ~75% from 100 to 200 s.f.u.), so
:class:`~kino.observing.skymodel.SkyConditions` needs the monthly-mean F10.7 of
the epoch being forecast.

The table below is NOAA SWPC's *predicted* monthly F10.7 for solar cycle 25,
``predicted_f10.7`` (central) with ``low_f10.7`` / ``high_f10.7`` (its stated
range), from
https://services.swpc.noaa.gov/json/solar-cycle/predicted-solar-cycle.json,
retrieved 2026-09-26. It is a *prediction* that NOAA revises monthly: refresh it
(or pass your own value) before quoting a forecast, and prefer the observed value
once a month has passed
(https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json).
Observed monthly means for 2026 were 120-138 s.f.u. from March to July and 116 in
August.
"""

from __future__ import annotations

#: month -> (predicted, low, high) F10.7 in solar flux units.
NOAA_SWPC_F107_PREDICTION: dict[str, tuple[float, float, float]] = {
    "2026-09": (129.6, 120.4, 137.0),
    "2026-10": (129.7, 119.6, 137.2),
    "2026-11": (129.4, 118.2, 137.0),
    "2026-12": (128.4, 116.4, 136.4),
    "2027-01": (126.7, 114.2, 135.5),
    "2027-02": (124.4, 111.6, 133.9),
    "2027-03": (121.9, 108.9, 132.1),
    "2027-04": (119.0, 105.9, 130.0),
    "2027-05": (116.2, 103.5, 127.8),
    "2027-06": (113.4, 100.9, 125.3),
    "2027-07": (110.9, 98.5, 123.1),
    "2027-08": (108.4, 96.2, 120.9),
    "2027-09": (106.1, 94.2, 118.6),
    "2027-10": (104.1, 92.5, 116.5),
    "2027-11": (102.0, 90.9, 114.3),
    "2027-12": (100.2, 89.6, 112.3),
}


def solar_flux(date: str) -> tuple[float, float, float]:
    """``(predicted, low, high)`` monthly-mean F10.7 for the month of ``date``.

    ``date`` is anything starting ``YYYY-MM`` (``"2027-03"``, ``"2027-03-14"``).
    Raises ``KeyError`` outside the tabulated months -- pass ``solar_flux``
    to ``SkyConditions`` explicitly then.
    """
    month = str(date)[:7]
    try:
        return NOAA_SWPC_F107_PREDICTION[month]
    except KeyError:
        first, last = min(NOAA_SWPC_F107_PREDICTION), max(NOAA_SWPC_F107_PREDICTION)
        raise KeyError(
            f"no F10.7 prediction tabulated for {month} (have {first} to {last}); "
            "pass solar_flux to SkyConditions explicitly"
        ) from None
