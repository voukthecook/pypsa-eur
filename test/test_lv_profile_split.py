import numpy as np
import pypsa

from scripts.prepare_sector_network import apply_lv_profile_split


def _network_with_lv_components() -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots([0, 1])
    n.add(
        "Bus",
        [
            "AC node",
            "node low voltage",
            "node home battery",
            "node EV battery",
            "node rural heat",
            "node rural water tanks",
            "gas node",
        ],
        carrier=[
            "AC",
            "low voltage",
            "home battery",
            "EV battery",
            "rural heat",
            "rural water tanks",
            "gas",
        ],
        location=["node"] * 6 + ["gas"],
    )
    n.add(
        "Link",
        ["distribution", "home charger", "BEV charger", "V2G", "rural heat pump"],
        bus0=[
            "AC node",
            "node low voltage",
            "node low voltage",
            "node EV battery",
            "node low voltage",
        ],
        bus1=[
            "node low voltage",
            "node home battery",
            "node EV battery",
            "node low voltage",
            "node rural heat",
        ],
        carrier=[
            "electricity distribution grid",
            "home battery charger",
            "BEV charger",
            "V2G",
            "rural heat pump",
        ],
        p_nom=[90, 30, 60, 45, 50],
        p_nom_max=[120, 40, 80, 60, 70],
    )
    n.add(
        "Load",
        ["electric demand", "EV demand", "rural heat demand"],
        bus=["node low voltage", "node EV battery", "node rural heat"],
        carrier=["electricity", "land transport EV", "rural heat"],
        p_set=[0, 0, 0],
    )
    n.loads_t.p_set.loc[:, "electric demand"] = [10, 20]
    n.loads_t.p_set.loc[:, "EV demand"] = [3, 4]
    n.loads_t.p_set.loc[:, "rural heat demand"] = [30, 40]
    n.add(
        "Generator",
        ["rooftop solar", "rural solar thermal"],
        bus=["node low voltage", "node rural heat"],
        carrier=["solar rooftop", "rural solar thermal"],
        p_nom=[0, 0],
        p_nom_max=[60, 40],
        p_nom_extendable=True,
    )
    n.add(
        "Store",
        ["home battery", "EV battery", "rural tank"],
        bus=[
            "node home battery",
            "node EV battery",
            "node rural water tanks",
        ],
        carrier=["home battery", "EV battery", "rural water tanks"],
        e_nom=[0, 20, 0],
        e_nom_max=[100, 30, 80],
        e_nom_extendable=True,
    )
    return n


def test_profile_split_clones_components_with_configured_shares() -> None:
    n = _network_with_lv_components()
    config = {
        "enable": True,
        "shares": {"urban": 0.4, "suburban": 0.3, "rural": 0.3},
        "pv_shares": {"urban": 0.2, "suburban": 0.3, "rural": 0.5},
        "heat_shares": {
            "rural": {"urban": 0.2, "suburban": 0.3, "rural": 0.5}
        },
        "ev_shares": {"urban": 0.1, "suburban": 0.2, "rural": 0.7},
    }

    apply_lv_profile_split(n, config)

    assert "node low voltage" not in n.buses.index
    assert "node EV battery" not in n.buses.index
    assert "node rural heat" not in n.buses.index
    assert {
        f"node low voltage {profile}" for profile in ("urban", "suburban", "rural")
    }.issubset(n.buses.index)

    np.testing.assert_allclose(
        [
            n.loads_t.p_set.at[0, f"electric demand {profile}"]
            for profile in ("urban", "suburban", "rural")
        ],
        [4, 3, 3],
    )
    np.testing.assert_allclose(
        [
            n.loads_t.p_set.at[0, f"EV demand {profile}"]
            for profile in ("urban", "suburban", "rural")
        ],
        [0.3, 0.6, 2.1],
    )
    np.testing.assert_allclose(
        [
            n.loads_t.p_set.at[0, f"rural heat demand {profile}"]
            for profile in ("urban", "suburban", "rural")
        ],
        [6, 9, 15],
    )

    distribution = n.links[n.links.carrier == "electricity distribution grid"]
    assert distribution.p_nom.sum() == 90
    assert distribution.p_nom_max.sum() == 120
    np.testing.assert_allclose(
        [
            distribution.at[f"distribution {profile}", "p_nom"]
            for profile in ("urban", "suburban", "rural")
        ],
        [36, 27, 27],
    )
    rooftop = n.generators[n.generators.carrier == "solar rooftop"]
    np.testing.assert_allclose(
        [
            rooftop.at[f"rooftop solar {profile}", "p_nom_max"]
            for profile in ("urban", "suburban", "rural")
        ],
        [12, 18, 30],
    )
    chargers = n.links[n.links.carrier == "BEV charger"]
    assert chargers.p_nom.sum() == 60
    assert set(chargers.bus1) == {
        f"node EV battery {profile}" for profile in ("urban", "suburban", "rural")
    }
    stores = n.stores[n.stores.carrier == "EV battery"]
    assert stores.e_nom.sum() == 20
    assert stores.e_nom_max.sum() == 30
    assert not any(name in n.loads.index for name in ("electric demand", "EV demand"))


def test_profile_split_defaults_to_equal_shares() -> None:
    n = _network_with_lv_components()

    apply_lv_profile_split(n, {"enable": True})

    np.testing.assert_allclose(
        [
            n.loads_t.p_set.at[0, f"electric demand {profile}"]
            for profile in ("urban", "suburban", "rural")
        ],
        [10 / 3, 10 / 3, 10 / 3],
    )
    np.testing.assert_allclose(n.generators.p_nom_max.sum(), 100)


def test_profile_split_includes_urban_central_heat_and_pit_storage() -> None:
    n = pypsa.Network()
    n.set_snapshots([0, 1])
    n.add(
        "Bus",
        [
            "AC node",
            "node low voltage",
            "node urban central heat",
            "node urban central water pits",
            "gas node",
        ],
        carrier=[
            "AC",
            "low voltage",
            "urban central heat",
            "urban central water pits",
            "gas",
        ],
        location=["node"] * 4 + ["gas"],
    )
    n.add(
        "Link",
        "central boiler",
        bus0="gas node",
        bus1="node urban central heat",
        carrier="urban central gas boiler",
        p_nom=90,
        p_nom_max=120,
    )
    n.add(
        "Load",
        "central heat demand",
        bus="node urban central heat",
        carrier="urban central heat",
        p_set=0,
    )
    n.loads_t.p_set["central heat demand"] = [20, 30]
    n.add(
        "Store",
        "central pit",
        bus="node urban central water pits",
        carrier="urban central water pits",
        e_nom=0,
        e_nom_max=150,
        e_nom_extendable=True,
    )

    shares = {"urban": 0.4, "suburban": 0.3, "rural": 0.3}
    apply_lv_profile_split(
        n,
        {"enable": True, "shares": shares, "heat_shares": {"urban central": shares}},
    )

    np.testing.assert_allclose(
        [n.loads_t.p_set.at[0, f"central heat demand {profile}"] for profile in shares],
        [8, 6, 6],
    )
    central_links = n.links[n.links.carrier == "urban central gas boiler"]
    np.testing.assert_allclose(central_links.p_nom.sum(), 90)
    np.testing.assert_allclose(central_links.p_nom_max.sum(), 120)
    central_stores = n.stores[n.stores.carrier == "urban central water pits"]
    np.testing.assert_allclose(central_stores.e_nom_max.sum(), 150)
    assert set(central_stores.bus) == {
        f"node urban central water pits {profile}" for profile in shares
    }


def test_disabled_profile_split_leaves_network_unchanged() -> None:
    n = _network_with_lv_components()
    original_buses = set(n.buses.index)
    original_loads = set(n.loads.index)

    returned = apply_lv_profile_split(n, {"enable": False})

    assert returned is n
    assert set(n.buses.index) == original_buses
    assert set(n.loads.index) == original_loads
