"""Inventory and layered drawing of the buses and links around one node.

Usage:  python draw_node_topology.py split.nc "GR0 0"
        (second argument = bus-name prefix of the node; default: first AC bus)
Needs:  pip install pypsa networkx matplotlib pandas

1. Prints a table: how many buses / links / loads / generators / stores of each
   carrier sit in each profile (urban / suburban / rural) and in the shared layer,
   and saves it as node_inventory.csv (one row per component).
2. Draws buses only (links are the edges, labelled by carrier), arranged in
   columns per profile and layers by distance from the AC bus. Saved as node_topology.png.
Components are keyed by (type, name), so a store or load sharing its bus's name
no longer overwrites it.
"""
import sys
from collections import defaultdict
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pypsa

PROFILES = ["urban", "suburban", "rural"]
COLORS = {"urban": "#1D9E75", "suburban": "#D85A30", "rural": "#534AB7", "shared": "#888780"}


def profile_of(name):
    s = str(name).lower()
    for p in ("suburban", "urban", "rural"):      # suburban before urban
        if s.endswith(" " + p):
            return p
    return "shared"


n = pypsa.Network(sys.argv[1])
prefix = sys.argv[2] if len(sys.argv) > 2 else n.buses.index[n.buses.carrier == "AC"][0]
buses = [b for b in n.buses.index if b.startswith(prefix)]
bset = set(buses)
bprof = {b: profile_of(b) for b in buses}
ports = [c for c in n.links.columns if c.startswith("bus") and c[3:].isdigit()]

# ---- inventory ---------------------------------------------------------------
rows = [("Bus", bprof[b], n.buses.at[b, "carrier"], b) for b in buses]
link_ends = {}
for l, row in n.links.iterrows():
    ends = [row[p] for p in ports if isinstance(row[p], str) and row[p] in bset]
    if not ends:
        continue
    link_ends[l] = ends
    prof = next((bprof[b] for b in ends if bprof[b] != "shared"), "shared")
    rows.append(("Link", prof, row.carrier, l))
for comp, label in [("loads", "Load"), ("generators", "Generator"), ("stores", "Store"),
                    ("storage_units", "StorageUnit")]:
    df = getattr(n, comp)
    for name, row in df[df.bus.isin(bset)].iterrows():
        rows.append((label, bprof[row.bus], row.get("carrier", ""), name))

inv = pd.DataFrame(rows, columns=["component", "profile", "carrier", "name"])
inv.to_csv("node_inventory.csv", index=False)
table = inv.pivot_table(index=["component", "carrier"], columns="profile", values="name",
                        aggfunc="count", fill_value=0)
print(f"{len(buses)} buses and {len(link_ends)} links around '{prefix}'\n")
print(table.to_string())
print("\nTotals per profile:\n", inv.groupby(["profile", "component"]).size().unstack(fill_value=0))

# ---- drawing: buses as nodes, links grouped by carrier as edges ----------------
edges = defaultdict(set)
for l, ends in link_ends.items():
    for b in ends[1:]:
        edges[(ends[0], b)].add(n.links.at[l, "carrier"])
g = nx.Graph()
g.add_nodes_from(buses)
g.add_edges_from(edges)

root = prefix if prefix in bset else next(b for b in buses if n.buses.at[b, "carrier"] == "AC")
depth = nx.single_source_shortest_path_length(g, root)

attached = {c: getattr(n, c).bus.value_counts() for c in ("loads", "generators", "stores")}
pos, labels = {}, {}
cols = {"shared": -1, "urban": 0, "suburban": 1, "rural": 2}
layers = defaultdict(list)
for b in buses:
    layers[(bprof[b], depth.get(b, 9))].append(b)
for (prof, d), members in layers.items():
    for k, b in enumerate(sorted(members)):
        pos[b] = (cols[prof] * 7 + (k - (len(members) - 1) / 2) * 1.5, -2 * d)
for b in buses:
    extra = ", ".join(f"{int(attached[c].get(b, 0))} {c[:-1] if c != 'storage_units' else c}"
                      for c in attached if attached[c].get(b, 0))
    labels[b] = n.buses.at[b, "carrier"] + (f"\n({extra})" if extra else "")

fig, ax = plt.subplots(figsize=(22, 12))
nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.4, width=1)
nx.draw_networkx_edge_labels(g, pos, ax=ax, font_size=5, rotate=False,
                             edge_labels={k: "\n".join(sorted(v)) for k, v in edges.items()})
nx.draw_networkx_nodes(g, pos, ax=ax, node_size=900, node_color=[COLORS[bprof[b]] for b in g])
nx.draw_networkx_labels(g, pos, labels, font_size=6, ax=ax)
for name, c in cols.items():
    ax.text(c * 7, 1.2, name, ha="center", fontsize=12, color=COLORS[name])
ax.set_title(f"Buses and links around '{prefix}' (edges labelled by link carrier)")
ax.axis("off")
fig.tight_layout()
fig.savefig("node_topology.png", dpi=160)
plt.show()