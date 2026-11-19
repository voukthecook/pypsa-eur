import numpy as np
import pandas as pd
import pytest

from scripts.lv_profile_split import PROFILES, apply_lv_profile_split
from scripts.lv_split.local_time_shapes import country_timezone, resolve_shape
from test.test_lv_profile_shapes import SHARES, _config, _network


def _evening_peak_table() -> pd.DataFrame:
    """Urban peaks at local 19:00, others flat. Same every month and day type."""
    rows = [
        {
            "month": m,
            "daytype": d,
            "hour": h,
            "urban": 3.0 if h == 19 else 1.0,
            "suburban": 1.0,
            "rural": 1.0,
        }
        for m in range(1, 13)
        for d in ("weekday", "weekend")
        for h in range(24)
    ]
    return pd.DataFrame(rows).set_index(["month", "daytype", "hour"])


def _hourly(start: str, periods: int):
    snaps = pd.date_range(start, periods=periods, freq="h")
    return snaps, pd.Series(1.0, index=snaps)


def test_peak_moves_with_country_clock_and_dst() -> None:
    table = _evening_peak_table()
    # winter: Athens = UTC+2, Berlin = UTC+1; summer: +3 / +2
    for start, athens_utc, berlin_utc in [("2013-01-15", 17, 18), ("2013-07-15", 16, 17)]:
        snaps, dur = _hourly(start, 24)
        out = resolve_shape(snaps, dur, table, ["DE", "GR"])
        assert out[("GR", "urban")].idxmax().hour == athens_utc
        assert out[("DE", "urban")].idxmax().hour == berlin_utc


def test_three_hour_snapshot_averages_its_window() -> None:
    table = _evening_peak_table()
    snaps = pd.date_range("2013-01-15", periods=8, freq="3h")
    out = resolve_shape(snaps, pd.Series(3.0, index=snaps), table, ["DE"])
    # local 19:00 = 18:00 UTC falls in the 18-21 UTC window: (3 + 1 + 1) / 3
    assert out[("DE", "urban")].loc["2013-01-15 18:00"] == pytest.approx(5 / 3)
    assert out[("DE", "urban")].loc["2013-01-15 00:00"] == pytest.approx(1.0)


def test_override_and_unknown_country() -> None:
    assert country_timezone("ES", {"ES": "Atlantic/Canary"}) == "Atlantic/Canary"
    with pytest.raises(KeyError, match="sector.lv_split_profiles.timezones"):
        country_timezone("ZZ")


def test_resolved_shapes_keep_conservation_in_the_split() -> None:
    n = _network()
    snaps = n.snapshots
    resolved = resolve_shape(
        snaps, pd.Series(3.0, index=snaps), _evening_peak_table(), ["NO"]
    )
    base = _network().loads_t.p_set["electric demand"]
    out = apply_lv_profile_split(
        n, _config(resolved_shapes={"electricity": resolved})
    )
    parts = pd.concat(
        [out.loads_t.p_set[f"electric demand {p}"] for p in PROFILES], axis=1
    )
    np.testing.assert_allclose(parts.sum(axis=1), base, rtol=1e-9)
    weights = out.snapshot_weightings.generators
    for p in PROFILES:
        np.testing.assert_allclose(
            (out.loads_t.p_set[f"electric demand {p}"] * weights).sum(),
            (base * weights).sum() * SHARES[p],
            rtol=1e-9,
        )
    assert not np.allclose(
        out.loads_t.p_set["electric demand urban"], base * SHARES["urban"]
    )


def test_greece_uses_pypsa_eur_code_el() -> None:
    assert country_timezone("EL") == country_timezone("GR") == "Europe/Athens"
