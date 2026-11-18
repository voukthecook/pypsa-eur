"""
Per-profile EV sizing and flows of one solved network, plus the period it covers.

    python check_ev_split.py path\\to\\network.nc

Use it to check that EV demand, store size and charger/V2G capacity follow the same
shares, and to see how many hours the network actually covers.
"""

import sys

import pandas as pd
import pypsa

PROFILES = ("urban", "suburban", "rural", "a", "b")


def prof(name):
    last = str(name).split(" ")[-1]
    return last if last in PROFILES else "all"


n = pypsa.Network(sys.argv[1])
sw = n.snapshot_weightings
wg = sw.generators
print(
    f"{len(n.snapshots)} snapshots from {n.snapshots[0]} to {n.snapshots[-1]}; "
    f"generator weights sum {wg.sum():.0f} h, objective weights sum {sw.objective.sum():.0f} h\n"
)

load = n.loads.index[n.loads.carrier == "land transport EV"]
chg = n.links.index[n.links.carrier == "BEV charger"]
v2g = n.links.index[n.links.carrier == "V2G"]
sto = n.stores.index[n.stores.carrier == "EV battery"]


def load_energy(i):
    if i in n.loads_t.p_set.columns:
        return float(n.loads_t.p_set[i].mul(wg).sum())
    return float(n.loads.at[i, "p_set"] * wg.sum())


rows = {}
for p in sorted({prof(i) for i in [*load, *chg, *v2g, *sto]}):
    sel = lambda idx: [i for i in idx if prof(i) == p]
    rows[p] = {
        "EV demand MWh": sum(load_energy(i) for i in sel(load)),
        "store MWh": n.stores.loc[sel(sto), "e_nom_opt"].sum(),
        "charger MW": n.links.loc[sel(chg), "p_nom_opt"].sum(),
        "V2G MW": n.links.loc[sel(v2g), "p_nom_opt"].sum(),
        "charged MWh": float(n.links_t.p0[sel(chg)].sum(axis=1).mul(wg).sum()),
        "V2G out MWh": float((-n.links_t.p1[sel(v2g)].sum(axis=1)).mul(wg).sum()),
    }
df = pd.DataFrame(rows).T
print(df.round(0).to_string())
print("\nshare of total (columns should match if all EV parts follow the same share):")
print((df / df.sum()).round(3).to_string())