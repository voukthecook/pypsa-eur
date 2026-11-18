"""
Show what the split_low_voltage_buses() distribution-grid approach
actually did in a solved network: for each node, compares its two
low-voltage sub-buses ("... low voltage a" / "... low voltage b") side
by side -- built solar rooftop capacity, served demand, and home
battery charge/discharge -- so you can see the split (and any
solar_scaling asymmetry) reflected in the optimizer's actual solution.

Run from the repo root with the pypsa-eur conda environment active:

    python check_lv_split.py results/networks/base_s_15_lv1.2_Co2L0-168H_2030.nc
"""

import sys
from pathlib import Path

import pandas as pd
import pypsa


def resolve_network_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None

    candidate = Path(raw_path)
    if candidate.exists():
        return candidate

    # Accept repo-root-relative paths when launched from a different cwd.
    repo_root = Path(__file__).resolve().parent
    repo_candidate = repo_root / candidate
    if repo_candidate.exists():
        return repo_candidate

    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate

    return None


path = sys.argv[1] if len(sys.argv) > 1 else None
if path is None:
    sys.exit(
        "Usage: python check_lv_split.py <results/networks/base_s_40_lv1.2_Co2L0-168H_2030.nc>"
    )

network_path = resolve_network_path(path)
if network_path is None:
    sys.exit(
        f"Network file not found: {path}\n"
        "Pass a valid absolute path or a path relative to the repo root, e.g.\n"
        "  python check_lv_split.py results/networks/base_s_15_lv1.2_Co2L0-168H_2030.nc"
    )

n = pypsa.Network(str(network_path))

ac_buses = n.buses.index[n.buses.carrier == "AC"]
lv_buses = n.buses.index[n.buses.carrier == "low voltage"]
split_buses = [b for b in lv_buses if b.endswith(" a") or b.endswith(" b")]
lv_bus_set = set(lv_buses)
ac_bus_set = set(ac_buses)
expected_lv_buses = {
    f"{ac_bus} low voltage {suffix}"
    for ac_bus in ac_buses
    for suffix in ("a", "b")
}
split_nodes = {
    ac_bus: (f"{ac_bus} low voltage a", f"{ac_bus} low voltage b")
    for ac_bus in ac_buses
}
fully_split_ac_buses = [
    ac_bus
    for ac_bus, pair in split_nodes.items()
    if all(bus in lv_bus_set for bus in pair)
]

print(f"Network: {network_path}")
print(f"Total AC buses: {len(ac_buses)}")
print(f"Total LV buses: {len(lv_buses)}")
print(f"Split LV buses: {len(split_buses)}")
print(f"AC buses with both split LV buses: {len(fully_split_ac_buses)} / {len(ac_buses)}")
print(f"Split LV buses per AC bus: {len(split_buses) / len(ac_buses) if len(ac_buses) else 0:.2f}")
print()

if not split_buses:
    print(f"First low-voltage bus names: {list(lv_buses[:10])}")
    print(
        "No split low-voltage buses (ending in ' a' / ' b') found in this "
        "network -- it looks like this run did NOT have "
        "electricity_distribution_grid_split_lv enabled, or used the plain "
        "single-bus / SimBench-detailed distribution grid instead."
    )
    sys.exit(0)

# Group split buses back into their parent node, e.g.
# "DK0 0AC low voltage a" / "DK0 0AC low voltage b" -> node "DK0 0AC"
nodes = sorted({b.rsplit(" ", 1)[0] for b in split_buses})

topology_rows = []
topology_issues = []
distribution_links = n.links[
    n.links.carrier == "electricity distribution grid"
]
for ac_bus, (bus_a, bus_b) in split_nodes.items():
    row = {"AC bus": ac_bus}
    for suffix, lv_bus in (("a", bus_a), ("b", bus_b)):
        present = lv_bus in lv_bus_set
        forward_links = distribution_links[
            (distribution_links.bus0 == ac_bus)
            & (distribution_links.bus1 == lv_bus)
        ]
        reverse_links = distribution_links[
            (distribution_links.bus0 == lv_bus)
            & (distribution_links.bus1 == ac_bus)
        ]
        reversible_single_link = (
            len(forward_links) + len(reverse_links) == 1
            and float(
                (forward_links if not forward_links.empty else reverse_links)
                .p_min_pu.iloc[0]
            )
            < 0
        )
        row[f"LV {suffix} exists"] = present
        row[f"AC->LV links {suffix}"] = len(forward_links)
        row[f"LV->AC links {suffix}"] = len(reverse_links)
        row[f"loads {suffix}"] = int((n.loads.bus == lv_bus).sum())
        row[f"generators {suffix}"] = int((n.generators.bus == lv_bus).sum())
        row[f"other links {suffix}"] = int(
            (
                (n.links.bus0 == lv_bus)
                | (n.links.bus1 == lv_bus)
                | (n.links.bus2 == lv_bus)
                | (n.links.bus3 == lv_bus)
            ).where(n.links.carrier != "electricity distribution grid", False).sum()
        )
        if not present:
            topology_issues.append(f"{ac_bus}: missing LV bus '{lv_bus}'")
        if not (
            (len(forward_links) == 1 and len(reverse_links) == 1)
            or reversible_single_link
        ):
            topology_issues.append(
                f"{ac_bus}: expected one distribution connection in each direction "
                f"for LV {suffix} (or one reversible link); found "
                f"{len(forward_links)} AC-to-LV and {len(reverse_links)} LV-to-AC links"
            )
    topology_rows.append(row)

unexpected_split_buses = sorted(set(split_buses) - expected_lv_buses)
if unexpected_split_buses:
    topology_issues.append(
        "split LV buses do not map to an AC bus: "
        + ", ".join(unexpected_split_buses)
    )
unexpected_lv_buses = sorted(lv_bus_set - expected_lv_buses)
if unexpected_lv_buses:
    topology_issues.append(
        "low-voltage buses are not part of an expected AC split pair: "
        + ", ".join(unexpected_lv_buses)
    )

bus_reference_columns = {
    "links": (n.links, [f"bus{i}" for i in range(4)]),
    "loads": (n.loads, ["bus"]),
    "generators": (n.generators, ["bus"]),
    "storage_units": (n.storage_units, ["bus"]),
    "stores": (n.stores, ["bus"]),
    "lines": (n.lines, ["bus0", "bus1"]),
    "transformers": (n.transformers, ["bus0", "bus1"]),
}
invalid_references = []
for component, (frame, columns) in bus_reference_columns.items():
    existing_columns = [column for column in columns if column in frame.columns]
    references = frame[existing_columns].stack().dropna()
    invalid_references.extend(
        (component, str(name), str(bus))
        for (name, _), bus in references.items()
        if str(bus).strip() and bus not in n.buses.index
    )
if invalid_references:
    topology_issues.append(
        f"{len(invalid_references)} component bus reference(s) point to missing buses"
    )

topology_table = pd.DataFrame(topology_rows).set_index("AC bus")
print("--- Topology validation ---")
if topology_issues:
    print(f"FAIL: {len(topology_issues)} topology issue(s)")
    for issue in topology_issues:
        print(f"  - {issue}")
else:
    print(
        "PASS: every AC bus has both LV halves, bidirectional distribution "
        "connections to each half, and all component bus references resolve"
    )
if invalid_references:
    print("First invalid component bus references:")
    for component, name, bus in invalid_references[:10]:
        print(f"  - {component} '{name}' references missing bus '{bus}'")
print("\nLoads, generators, and non-distribution links attached to each LV half:")
print(topology_table.to_string())
print()

weighting = n.snapshot_weightings.generators

rows = []
for node_lv in nodes:
    bus_a = f"{node_lv} a"
    bus_b = f"{node_lv} b"
    row = {"node": node_lv}

    for suffix, bus in [("a", bus_a), ("b", bus_b)]:
        # Solar rooftop capacity actually built and the available rooftop potential
        solar = n.generators[
            (n.generators.bus == bus) & (n.generators.carrier == "solar rooftop")
        ]
        row[f"solar_p_nom_max_{suffix}_MW"] = solar.p_nom_max.sum()
        row[f"solar_p_nom_opt_{suffix}_MW"] = solar.p_nom_opt.sum()

        # Total served demand on this sub-bus (MWh, weighted)
        loads = n.loads.index[n.loads.bus == bus]
        if len(loads):
            demand = (
                n.loads_t.p_set[loads].multiply(weighting, axis=0).sum().sum()
            )
        else:
            demand = 0.0
        row[f"demand_MWh_{suffix}"] = demand

        # Capacity of the AC<->LV distribution link feeding this sub-bus
        dist_links = n.links[
            (n.links.bus1 == bus) & (n.links.carrier == "electricity distribution grid")
        ]
        row[f"dist_link_p_nom_opt_{suffix}_MW"] = dist_links.p_nom_opt.sum()

        # Home battery charger / discharger capacity attached to this sub-bus
        charger_links = n.links[
            (n.links.bus0 == bus) & (n.links.carrier == "home battery charger")
        ]
        discharger_links = n.links[
            (n.links.bus1 == bus) & (n.links.carrier == "home battery discharger")
        ]
        row[f"home_battery_charger_p_nom_opt_{suffix}_MW"] = charger_links.p_nom_opt.sum()
        row[f"home_battery_discharger_p_nom_opt_{suffix}_MW"] = (
            discharger_links.p_nom_opt.sum()
        )

        if not charger_links.empty:
            charge_mwh = (
                n.links_t.p0[charger_links.index].clip(lower=0).multiply(weighting, axis=0).sum().sum()
            )
        else:
            charge_mwh = 0.0
        if not discharger_links.empty:
            discharge_mwh = (
                n.links_t.p1[discharger_links.index].clip(lower=0).multiply(weighting, axis=0).sum().sum()
            )
        else:
            discharge_mwh = 0.0
        row[f"home_battery_charge_MWh_{suffix}"] = charge_mwh
        row[f"home_battery_discharge_MWh_{suffix}"] = discharge_mwh

    rows.append(row)

table = pd.DataFrame(rows).set_index("node")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

print(f"Found {len(nodes)} node(s) with split low-voltage buses.\n")
print(table)

print("\n--- Sanity checks ---")
demand_ratio = (table["demand_MWh_a"] / table["demand_MWh_b"]).round(3)
print("Demand a/b ratio per node (should be ~1.0, the even 50/50 split):")
print(demand_ratio)

solar_ratio = (
    table["solar_p_nom_opt_a_MW"] / table["solar_p_nom_opt_b_MW"].replace(0, float("nan"))
).round(3)
print(f"\nBuilt solar capacity a/b ratio per node (reflects solar_scaling AND")
print("the optimizer's own siting choice within that scaled potential):")
print(solar_ratio)

battery_charge_ratio = (
    table["home_battery_charge_MWh_a"] / table["home_battery_charge_MWh_b"].replace(0, float("nan"))
).round(3)
print("\nHome battery charging a/b ratio per node:")
print(battery_charge_ratio)

print(
    "\nThis split-LV view shows the main distribution-grid decisions at the sub-bus level: "
    "rooftop solar buildout, served demand, distribution connection sizing, and home battery "
    "charging/discharging on each side of the node. In this model, each AC bus is split into "
    "two low-voltage buses, labelled '... a' and '... b'."
)

if topology_issues:
    sys.exit(1)
