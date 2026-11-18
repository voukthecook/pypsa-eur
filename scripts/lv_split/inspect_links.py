import pypsa

n = pypsa.Network(r".\results\C2SplitLV_GR_3_clusters_3H\networks\base_s_3_lv1.2_Co2L0-3H_2030.nc")

d = n.links[n.links.carrier == "electricity distribution grid"]
print(d[["bus0", "bus1", "p_nom_extendable", "p_nom", "p_nom_opt", "capital_cost"]])

# links whose bus1 is not one of the split halves (the "-" rows)
odd = d[~d.bus1.str.endswith((" a", " b"))]
for bus in odd.bus1.unique():
    print("\n==", bus, "| bus carrier:", n.buses.at[bus, "carrier"])
    for comp in ["loads", "generators", "stores", "links"]:
        df = getattr(n, comp)
        cols = [c for c in df.columns if c.startswith("bus")]
        m = df[df[cols].isin([bus]).any(axis=1)]
        print(f"  {comp}: {len(m)}", list(m.index[:8]))
    
        
      
        
        

n = pypsa.Network(r".\results\B5SplitLV_GR_3_clusters_3H\networks\base_s_3_lv1.2_Co2L0-3H_2030.nc")
d = n.links[n.links.carrier == "electricity distribution grid"]
print(d[["bus0", "bus1", "p_nom_extendable", "p_nom", "p_nom_opt", "capital_cost"]])


n = pypsa.Network(r".\results\B5SplitLV_GR_3_clusters_3H\networks\base_s_3_lv1.2_Co2L0-3H_2030.nc")
for half in ("a", "b"):
    l = n.loads[(n.loads.bus.str.endswith(f"low voltage {half}")) & n.loads.carrier.str.contains("electric")]
    print(half, n.loads_t.p_set[l.index].sum(axis=1).max())
    
n = pypsa.Network("results/B7SplitLV_GR_3_clusters_24H/networks/base_s_3_lv1.2_Co2L0-24H_2030.nc")
print(n.statistics.energy_balance(bus_carrier="H2").abs().sum())
print(n.buses.carrier.value_counts())

import pypsa, pandas as pd

paths = {
    "tutorial": "results/test-sector-overnight/networks/base_s_5___2030.nc",
    "B7": "results/B7SplitLV_GR_3_clusters_24H/networks/base_s_3_lv1.2_Co2L0-24H_2030.nc",
}
for name, p in paths.items():
    n = pypsa.Network(p)
    print(f"\n===== {name} =====")
    print("objective:", n.objective)
    print("buses:", len(n.buses), "| x,y zero or NaN:",
          int(((n.buses.x == 0) & (n.buses.y == 0)).sum() + n.buses.x.isna().sum()))
    print("generators p_nom_opt (GW) by carrier:")
    print((n.generators.groupby("carrier").p_nom_opt.sum() / 1e3).round(2))
    print("links p_nom_opt (GW) by carrier:")
    print((n.links.groupby("carrier").p_nom_opt.sum() / 1e3).round(2))
    print("total load (TWh):", round(n.loads_t.p_set.sum().sum() * n.snapshot_weightings.generators.iloc[0] / 1e6, 2))
    print(n.statistics.energy_balance().abs().groupby(level="carrier").sum().round(0).head(15))