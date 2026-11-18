import numpy as np
import pandas as pd
import pypsa
import pytest

from scripts.lv_profile_split import (
    PROFILES,
    apply_lv_profile_split,
    household_electricity_fraction,
)

SHARES = {"urban": 0.5, "suburban": 0.3, "rural": 0.2}


def _network() -> pypsa.Network:
    n = pypsa.Network()
    # Wednesday 2013-01-02, 3-hourly, plus a Saturday (2013-01-05)
    n.set_snapshots(
        list(pd.date_range("2013-01-02", periods=8, freq="3h"))
        + list(pd.date_range("2013-01-05", periods=8, freq="3h"))
    )
    n.add(
        "Bus",
        ["AC node", "node low voltage", "node EV battery"],
        carrier=["AC", "low voltage", "EV battery"],
        location="node",
    )
    n.add(
        "Link",
        ["distribution", "BEV charger"],
        bus0=["AC node", "node low voltage"],
        bus1=["node low voltage", "node EV battery"],
        carrier=["electricity distribution grid", "BEV charger"],
        p_nom=[90, 60],
        p_nom_max=[120, 80],
    )
    n.add(
        "Load",
        ["electric demand", "EV demand"],
        bus=["node low voltage", "node EV battery"],
        carrier=["electricity", "land transport EV"],
        p_set=[0, 0],
    )
    n.loads_t.p_set["electric demand"] = np.linspace(10, 20, len(n.snapshots))
    n.loads_t.p_set["EV demand"] = np.linspace(3, 6, len(n.snapshots))
    return n


def _shape_table(urban_evening: float) -> pd.DataFrame:
    rows = []
    for month in range(1, 13):
        for daytype in ("weekday", "weekend"):
            for hour in range(24):
                evening = 1.0 + (urban_evening if 17 <= hour <= 21 else 0.0)
                rows.append(
                    {
                        "month": month,
                        "daytype": daytype,
                        "hour": hour,
                        "urban": evening,
                        "suburban": 1.0,
                        "rural": 2.0 - (evening - 1.0) if hour < 12 else 1.0,
                    }
                )
    return pd.DataFrame(rows)


def _config(**extra):
    return {"enable": True, "shares": SHARES, **extra}


def test_without_shapes_nothing_changes_versus_plain_shares() -> None:
    n = apply_lv_profile_split(_network(), _config())
    base = _network().loads_t.p_set["electric demand"]
    for p in PROFILES:
        pd.testing.assert_series_equal(
            n.loads_t.p_set[f"electric demand {p}"],
            base * SHARES[p],
            check_names=False,
        )


def test_shapes_keep_node_total_in_every_snapshot_but_change_profiles() -> None:
    cfg = _config(shapes={"electricity": _shape_table(0.8), "ev": _shape_table(0.4)})
    n = apply_lv_profile_split(_network(), cfg)
    base = _network()
    for load in ("electric demand", "EV demand"):
        total = sum(n.loads_t.p_set[f"{load} {p}"] for p in PROFILES)
        pd.testing.assert_series_equal(
            total, base.loads_t.p_set[load], check_names=False, rtol=1e-12
        )
    for load in ("electric demand", "EV demand"):
        for p in PROFILES:
            np.testing.assert_allclose(
                n.loads_t.p_set[f"{load} {p}"].sum(),
                SHARES[p] * base.loads_t.p_set[load].sum(),
                rtol=1e-9,
            )
    urban = n.loads_t.p_set["electric demand urban"] / SHARES["urban"]
    suburban = n.loads_t.p_set["electric demand suburban"] / SHARES["suburban"]
    assert not np.allclose(urban, suburban)


def test_shape_changes_only_distribution_not_energy_per_profile_when_flat() -> None:
    flat = _shape_table(0.0).assign(rural=1.0)
    n = apply_lv_profile_split(
        _network(), _config(shapes={"electricity": flat})
    )
    base = _network().loads_t.p_set["electric demand"]
    for p in PROFILES:
        np.testing.assert_allclose(
            n.loads_t.p_set[f"electric demand {p}"], base * SHARES[p]
        )


def test_missing_hours_in_shape_table_raises() -> None:
    table = _shape_table(0.5)
    table = table[table["hour"] != 18]
    with pytest.raises(ValueError, match="does not cover"):
        apply_lv_profile_split(_network(), _config(shapes={"electricity": table}))


def test_unknown_shape_key_raises() -> None:
    with pytest.raises(ValueError, match="unsupported keys"):
        apply_lv_profile_split(
            _network(), _config(shapes={"heat": _shape_table(0.5)})
        )


def test_household_fraction_zero_means_plain_shares() -> None:
    n = apply_lv_profile_split(
        _network(),
        _config(
            shapes={"electricity": _shape_table(0.8)}, household_fraction=0.0
        ),
    )
    base = _network().loads_t.p_set["electric demand"]
    for p in PROFILES:
        np.testing.assert_allclose(
            n.loads_t.p_set[f"electric demand {p}"], base * SHARES[p]
        )


def test_household_fraction_keeps_both_conservation_rules() -> None:
    n = apply_lv_profile_split(
        _network(),
        _config(
            shapes={"electricity": _shape_table(0.8)}, household_fraction=0.6
        ),
    )
    base = _network().loads_t.p_set["electric demand"]
    total = sum(n.loads_t.p_set[f"electric demand {p}"] for p in PROFILES)
    np.testing.assert_allclose(total, base, rtol=1e-12)
    for p in PROFILES:
        np.testing.assert_allclose(
            n.loads_t.p_set[f"electric demand {p}"].sum(),
            SHARES[p] * base.sum(),
            rtol=1e-9,
        )


def test_household_fraction_from_energy_totals() -> None:
    totals = pd.DataFrame(
        {
            "electricity residential": [10.0, 4.0],
            "electricity residential space": [2.0, 1.0],
            "electricity residential water": [1.0, 0.0],
            "electricity services": [8.0, 6.0],
            "electricity services space": [1.0, 2.0],
            "electricity services water": [1.0, 0.0],
        },
        index=["node", "other"],
    )
    fraction = household_electricity_fraction(totals)
    assert fraction["node"] == pytest.approx(7 / (7 + 6))
    assert fraction["other"] == pytest.approx(3 / (3 + 4))


def test_per_node_household_fraction_uses_bus_location() -> None:
    n = apply_lv_profile_split(
        _network(),
        _config(
            shapes={"electricity": _shape_table(0.8)},
            household_fraction=pd.Series({"node": 0.0}),
        ),
    )
    base = _network().loads_t.p_set["electric demand"]
    np.testing.assert_allclose(
        n.loads_t.p_set["electric demand urban"], base * SHARES["urban"]
    )
