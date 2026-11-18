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

import pandas as pd
import pypsa

path = sys.argv[1] if len(sys.argv) > 1 else None
if path is None:
    sys.exit("Usage: python check_lv_split.py <results/networks/base_s_15_lv1.2_Co2L0-168H_2030.nc>")

n = pypsa.Network(path)

lv_buses = n.buses.index[n.buses.carrier == "low voltage"]
split_buses = [b for b in lv_buses if b.endswith(" a") or b.endswith(" b")]

if not split_buses:
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
weighting = n.snapshot_weightings.generators

rows = []
for node_lv in nodes:
    bus_a = f"{node_lv} a"
    bus_b = f"{node_lv} b"
    row = {"node": node_lv}

    for suffix, bus in [("a", bus_a), ("b", bus_b)]:
        # Solar rooftop capacity actually built on this sub-bus
        solar = n.generators[
            (n.generators.bus == bus) & (n.generators.carrier == "solar rooftop")
        ]
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

    rows.append(row)

table = pd.DataFrame(rows).set_index("node")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

print(f"Network: {path}")
print(f"Found {len(nodes)} node(s) with split low-voltage buses.\n")
print(table)

print("\n--- Sanity checks ---")
demand_ratio = (table["demand_MWh_a"] / table["demand_MWh_b"]).round(3)
print(f"Demand a/b ratio per node (should be ~1.0, the even 50/50 split):")
print(demand_ratio)

solar_ratio = (
    table["solar_p_nom_opt_a_MW"] / table["solar_p_nom_opt_b_MW"].replace(0, float("nan"))
).round(3)
print(f"\nBuilt solar capacity a/b ratio per node (reflects solar_scaling AND")
print("the optimizer's own siting choice within that scaled potential):")
print(solar_ratio)

print(
    "\nNote: the a/b solar ratio is not guaranteed to exactly equal your "
    "solar_scaling ratio -- the scaling only sets each bus's rooftop "
    "*potential* (p_nom_max); the optimizer still decides how much of "
    "that potential to actually build based on cost and value. A ratio "
    "close to the configured solar_scaling suggests the potential itself "
    "is the binding constraint; a ratio closer to 1.0 despite asymmetric "
    "scaling would suggest the optimizer wasn't incentivized to use the "
    "larger side's extra headroom."
)