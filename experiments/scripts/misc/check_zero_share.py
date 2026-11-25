"""List everything attached to heat buses of (system, profile) pairs whose share is 0.

Usage (PowerShell, one line):
  python check_zero_share.py results\\<run>\\networks\\base_s_3_lv1.2_Co2L0-3H_2030.nc

Default zero pairs match the current config:
  rural:urban  rural:suburban  urban decentral:rural  urban central:rural
Override with --zero "urban central:rural" "rural:urban" ...

A clean split prints "no components on zero-share buses" and exits 0.
"""
import argparse
import sys

import numpy as np
import pandas as pd
import pypsa

SYSTEMS = ("urban central", "urban decentral", "rural")
PROFILES = ("urban", "suburban", "rural")
DEFAULT_ZERO = ["rural:urban", "rural:suburban", "urban decentral:rural", "urban central:rural"]


def system_of(carrier):
    c = str(carrier).lower()
    for s in SYSTEMS:
        if c.startswith(s):
            return s
    return None


def profile_of(name):
    tok = str(name).split()[-1]
    return tok if tok in PROFILES else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("network")
    ap.add_argument("--zero", nargs="*", default=DEFAULT_ZERO)
    a = ap.parse_args()
    n = pypsa.Network(a.network)
    w = n.snapshot_weightings.generators
    zero = {tuple(z.rsplit(":", 1)) for z in a.zero}

    chain = {}
    for b in n.buses.index:
        car = n.buses.at[b, "carrier"]
        s, p = system_of(car), profile_of(b)
        if s and p and (s, p) in zero and any(k in car for k in ("heat", "water tank", "water pit")):
            chain[b] = (s, p)
    if not chain:
        print("no heat buses on zero-share (system, profile) pairs")
        return 0

    rows = []
    for l in n.loads.index[n.loads.bus.isin(chain)]:
        e = float((n.loads_t.p[l] * w).sum()) if l in n.loads_t.p else np.nan
        rows.append(("Load", l, n.loads.at[l, "carrier"], chain[n.loads.at[l, "bus"]], np.nan, e))
    for g in n.generators.index[n.generators.bus.isin(chain)]:
        e = float((n.generators_t.p[g] * w).sum()) if g in n.generators_t.p else np.nan
        rows.append(("Generator", g, n.generators.at[g, "carrier"], chain[n.generators.at[g, "bus"]],
                     n.generators.at[g, "p_nom_opt"], e))
    for s_ in n.stores.index[n.stores.bus.isin(chain)]:
        e = float((n.stores_t.p[s_].abs() * w).sum()) if s_ in n.stores_t.p else np.nan
        rows.append(("Store", s_, n.stores.at[s_, "carrier"], chain[n.stores.at[s_, "bus"]],
                     n.stores.at[s_, "e_nom_opt"], e))
    ports = [c for c in n.links.columns if c.startswith("bus") and c[3:].isdigit()]
    for l in n.links.index:
        hit = [n.links.at[l, c] for c in ports if n.links.at[l, c] in chain]
        if not hit:
            continue
        e = float((n.links_t.p0[l].abs() * w).sum()) if l in n.links_t.p0 else np.nan
        rows.append(("Link", l, n.links.at[l, "carrier"], chain[hit[0]], n.links.at[l, "p_nom_opt"], e))

    df = pd.DataFrame(rows, columns=["type", "name", "carrier", "(system, profile)", "capacity_opt", "energy_MWh"])
    used = df[(df.energy_MWh.fillna(0) > 1e-3) | (df.capacity_opt.fillna(0) > 1e-3)]
    print(f"{len(chain)} zero-share heat buses, {len(df)} attached components, {len(used)} active\n")
    if df.empty:
        print("no components on zero-share buses")
        return 0
    if used.empty:
        print("components EXIST on zero-share buses but none is used (clones were not skipped):")
        print(df.to_string(index=False))
        return 1
    pd.set_option("display.width", 220, "display.max_colwidth", 60)
    print(used.sort_values("energy_MWh", ascending=False).to_string(index=False))
    print("\nBy carrier (GWh):")
    print((used.groupby(["(system, profile)", "carrier"]).energy_MWh.sum() / 1e3).round(1).to_string())
    return 1


if __name__ == "__main__":
    sys.exit(main())