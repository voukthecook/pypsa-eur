import pypsa

from scripts.prepare_sector_network import (
    constrain_lv_links,
    split_low_voltage_buses,
    split_transport,
)


def _network_with_ev() -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots([0])
    n.add(
        "Bus",
        ["AC node", "node low voltage", "node EV battery"],
        carrier=["AC", "low voltage", "EV battery"],
        location=["node", "node", "node"],
    )
    n.add(
        "Link",
        "distribution",
        bus0="AC node",
        bus1="node low voltage",
        carrier="electricity distribution grid",
        p_nom=10,
    )
    n.add(
        "Link",
        "charger",
        bus0="node low voltage",
        bus1="node EV battery",
        carrier="BEV charger",
        p_nom=100,
        efficiency=0.9,
    )
    n.add(
        "Link",
        "V2G",
        bus0="node EV battery",
        bus1="node low voltage",
        carrier="V2G",
        p_nom=80,
        efficiency=0.9,
    )
    n.add(
        "Load",
        "EV demand",
        bus="node EV battery",
        carrier="land transport EV",
        p_set=20,
    )
    n.add(
        "Store",
        "EV store",
        bus="node EV battery",
        carrier="EV battery",
        e_nom=50,
    )
    return n


def test_lv_split_leaves_ev_chain_shared_until_transport_split_is_enabled() -> None:
    n = _network_with_ev()

    split_low_voltage_buses(n)

    assert "node EV battery" in n.buses.index
    assert "node EV battery a" not in n.buses.index
    assert n.loads.at["EV demand", "bus"] == "node EV battery"
    assert n.loads.at["EV demand", "p_set"] == 20
    assert n.stores.at["EV store", "bus"] == "node EV battery"
    assert n.stores.at["EV store", "e_nom"] == 50

    chargers = n.links.loc[n.links.carrier == "BEV charger"]
    assert set(chargers.bus0) == {"node low voltage a", "node low voltage b"}
    assert set(chargers.bus1) == {"node EV battery"}
    assert set(chargers.p_nom) == {50}

    v2g = n.links.loc[n.links.carrier == "V2G"]
    assert set(v2g.bus0) == {"node EV battery"}
    assert set(v2g.bus1) == {"node low voltage a", "node low voltage b"}
    assert set(v2g.p_nom) == {40}


def test_transport_split_splits_ev_chain_after_lv_links_are_cloned() -> None:
    n = _network_with_ev()

    split_low_voltage_buses(n)
    split_transport(
        n,
        ev_scaling=(1.5, 0.5),
        ev_charger_scaling=(1.5, 0.5),
        ev_v2g_scaling=(0.5, 1.5),
        ev_store_scaling=(1.5, 0.5),
    )

    chargers = n.links.loc[n.links.carrier == "BEV charger"].sort_index()
    v2g = n.links.loc[n.links.carrier == "V2G"].sort_index()
    assert list(chargers.p_nom) == [75, 25]
    assert list(v2g.p_nom) == [20, 60]
    assert set(chargers.bus1) == {"node EV battery a", "node EV battery b"}
    assert set(v2g.bus0) == {"node EV battery a", "node EV battery b"}
    assert list(
        n.loads.loc[n.loads.carrier == "land transport EV"].sort_index().p_set
    ) == [15, 5]
    assert list(n.stores.loc[n.stores.carrier == "EV battery"].sort_index().e_nom) == [
        37.5,
        12.5,
    ]


def test_transport_split_defaults_to_an_even_conserved_split() -> None:
    n = _network_with_ev()

    split_low_voltage_buses(n)
    split_transport(n)

    assert "node EV battery" not in n.buses.index
    assert {"node EV battery a", "node EV battery b"}.issubset(n.buses.index)
    assert set(n.loads.loc[n.loads.carrier == "land transport EV", "bus"]) == {
        "node EV battery a",
        "node EV battery b",
    }
    assert set(n.stores.loc[n.stores.carrier == "EV battery", "bus"]) == {
        "node EV battery a",
        "node EV battery b",
    }
    assert set(n.links.loc[n.links.carrier == "BEV charger", "bus1"]) == {
        "node EV battery a",
        "node EV battery b",
    }
    assert set(n.links.loc[n.links.carrier == "V2G", "bus0"]) == {
        "node EV battery a",
        "node EV battery b",
    }


def test_shared_ev_demand_is_attributed_by_charger_capacity() -> None:
    n = pypsa.Network()
    n.set_snapshots([0])
    n.add(
        "Bus",
        ["AC node", "node low voltage a", "node low voltage b", "node EV battery"],
        carrier=["AC", "low voltage", "low voltage", "EV battery"],
        location=["node", "node", "node", "node"],
    )
    n.add(
        "Link",
        ["grid a", "grid b"],
        bus0=["AC node", "AC node"],
        bus1=["node low voltage a", "node low voltage b"],
        carrier="electricity distribution grid",
        p_nom=10,
    )
    n.add(
        "Link",
        ["charger a", "charger b"],
        bus0=["node low voltage a", "node low voltage b"],
        bus1="node EV battery",
        carrier="BEV charger",
        p_nom=[10, 30],
        efficiency=0.8,
    )
    n.add(
        "Load",
        ["demand a", "demand b", "EV demand"],
        bus=["node low voltage a", "node low voltage b", "node EV battery"],
        carrier=["electricity", "electricity", "land transport EV"],
    )
    n.loads_t.p_set.loc[:, ["demand a", "demand b", "EV demand"]] = [
        [10, 20, 40]
    ]

    constrain_lv_links(n, [1.0, 1.0])

    assert n.links.at["grid a", "p_nom"] == 22.5
    assert n.links.at["grid b", "p_nom"] == 57.5
