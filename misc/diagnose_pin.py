"""Explain, for every DAC link, why the pin / zero-share cleanup does or does not touch it.

Usage:  python diagnose_pin.py results\\<run>\\networks\\base_s_3_lv1.2_Co2L0-3H_2030.nc
Works on a solved network or a prenetwork. Paste the whole output.
"""
import sys
import pandas as pd
import pypsa

PROFILES = ("urban", "suburban", "rural")
SYSTEMS = ("urban central", "urban decentral", "rural")
KEYS = ("heat", "water tank", "water pit")

n = pypsa.Network(sys.argv[1])
pd.set_option("display.width", 250, "display.max_colwidth", 60, "display.max_rows", 200)

def system(c):
    c = str(c).lower()
    return next((s for s in SYSTEMS if c.startswith(s)), None)

print("\n== heat buses (carrier, system, profile token)")
hb = [(b, n.buses.at[b, "carrier"]) for b in n.buses.index if any(k in str(n.buses.at[b, "carrier"]) for k in KEYS)]
df = pd.DataFrame([(b.split()[0] + " " + b.split()[1], c, system(c), b.split()[-1]) for b, c in hb],
                  columns=["node", "carrier", "system", "profile"])
print(df.groupby(["carrier", "system", "profile"]).size().unstack(fill_value=0).to_string())

print("\n== DAC links")
ports = [c for c in n.links.columns if c.startswith("bus") and c[3:].isdigit()]
chain = {b for b, c in hb if system(c) and b.split()[-1] in PROFILES}
rows = []
for l in n.links.index[n.links.carrier == "DAC"]:
    pc = [(c, n.links.at[l, c]) for c in ports if n.links.at[l, c] in chain]
    rows.append(dict(link=l, carrier=n.links.at[l, "carrier"], last_token=l.split()[-1],
                     token_is_profile=l.split()[-1] in PROFILES,
                     heat_ports=";".join(f"{c}={b}" for c, b in pc) or "NONE",
                     p_nom_opt=n.links.at[l, "p_nom_opt"] if "p_nom_opt" in n.links else float("nan")))
print(pd.DataFrame(rows).to_string(index=False))

print("\n== columns holding heat-bus names in DAC links:", ports)
print("== DAC carrier values:", sorted(n.links.carrier[n.links.carrier.str.contains("DAC")].unique()))
print("== all link carriers containing 'DAC' or 'direct':", sorted(c for c in n.links.carrier.unique() if "DAC" in c or "direct" in c.lower()))