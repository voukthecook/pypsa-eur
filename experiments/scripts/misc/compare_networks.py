"""
Compare solved networks on the three whiteboard questions.

    python compare_networks.py net_A.nc net_B.nc [--labels baseline profiles] [--out out_dir]

1. Distribution grid: energy and peak through the 'electricity distribution grid'
   links (import AC->LV, export LV->AC), capacity, per profile (urban/suburban/rural,
   or a/b, or 'all' for an unsplit network) and in total.
2. System costs: capex + opex by carrier, side by side, plus a focus table for the
   groups on the board (heat storage, rooftop PV, EV, home battery, heating).
3. Operation: EV charge (BEV charger p0), EV discharge to LV (V2G -p1) and EV state of
   charge (stores_t.e), per profile; CSV + figure per network.

Costs are rebuilt from p_nom_opt / e_nom_opt and marginal costs, so they can differ
slightly from n.objective (constants, global-constraint terms); the script prints both.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

PROFILE_SUFFIXES = ("urban", "suburban", "rural", "a", "b")
DIST = "electricity distribution grid"

COST_GROUPS = {
    "heat storage": ("water tank", "water pit"),
    "rooftop PV": ("solar rooftop",),
    "EV (charger/V2G/battery)": ("BEV charger", "V2G", "EV battery"),
    "home battery": ("home battery",),
    "heating": ("heat pump", "resistive heater", "boiler", "micro gas", "CHP"),
    "distribution grid": (DIST,),
}


def profile_of(name):
    last = str(name).split(" ")[-1]
    return last if last in PROFILE_SUFFIXES else "all"


def weights(n):
    return n.snapshot_weightings.generators


def is_reversed(n, idx):
    if "reversed" in n.links.columns:
        return n.links.loc[idx, "reversed"].fillna(False).astype(bool)
    return pd.Series(idx.str.contains("reversed"), index=idx)


# ---------------------------------------------------------------- 1. dist. grid
def dist_grid(n):
    w = weights(n)
    links = n.links.index[n.links.carrier == DIST]
    rev = is_reversed(n, links)
    p0 = n.links_t.p0[links]
    rows = []
    groups = {}
    for l in links:
        groups.setdefault(profile_of(l.replace("-reversed", "")), []).append(l)
    for prof, names in list(groups.items()) + [("TOTAL", list(links))]:
        fwd = [l for l in names if not rev[l]]
        bwd = [l for l in names if rev[l]]
        if bwd:  # lossy bidirectional: separate forward and reversed links
            imp = p0[fwd].sum(axis=1)
            exp = p0[bwd].sum(axis=1)
        else:  # single link with p_min_pu = -1
            net = p0[fwd].sum(axis=1)
            imp, exp = net.clip(lower=0), (-net).clip(lower=0)
        cap = n.links.loc[fwd, "p_nom_opt"].sum()
        rows.append(
            {
                "profile": prof,
                "import_GWh": (imp * w).sum() / 1e3,
                "export_GWh": (exp * w).sum() / 1e3,
                "throughput_GWh": ((imp + exp) * w).sum() / 1e3,
                "peak_import_MW": imp.max(),
                "peak_export_MW": exp.max(),
                "installed_MW": cap,
                "import_peak_to_mean": imp.max() / imp.mean() if imp.mean() > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("profile")


# ---------------------------------------------------------------- 2. costs
def costs_by_carrier(n):
    w = weights(n)
    parts = []
    for comp, nom, nom_ext in (
        ("generators", "p_nom_opt", "p_nom_extendable"),
        ("links", "p_nom_opt", "p_nom_extendable"),
        ("stores", "e_nom_opt", "e_nom_extendable"),
    ):
        df = getattr(n, comp)
        capex = (df[nom] * df["capital_cost"]).where(df[nom_ext], 0.0)
        dyn = getattr(n, comp + "_t")
        flow = dyn.p0 if comp == "links" else dyn.p
        flow = flow.reindex(columns=df.index, fill_value=0.0)
        opex = (flow.mul(w, axis=0) * df["marginal_cost"]).sum()
        parts.append(pd.DataFrame({"carrier": df.carrier, "capex": capex, "opex": opex}))
    out = pd.concat(parts).groupby("carrier").sum()
    out["total"] = out.capex + out.opex
    return out


def group_of(carrier):
    for g, keys in COST_GROUPS.items():
        if any(k.lower() in carrier.lower() for k in keys):
            return g
    return "other"


# ---------------------------------------------------------------- 3. operation
def ev_operation(n):
    chg = n.links.index[n.links.carrier == "BEV charger"]
    v2g = n.links.index[n.links.carrier == "V2G"]
    soc = n.stores.index[n.stores.carrier == "EV battery"]
    cols = {}
    for prof in sorted({profile_of(i) for i in list(chg) + list(v2g) + list(soc)}):
        sel = lambda idx: [i for i in idx if profile_of(i) == prof]
        cols[f"{prof} charge MW"] = n.links_t.p0[sel(chg)].sum(axis=1)
        cols[f"{prof} V2G MW"] = -n.links_t.p1[sel(v2g)].sum(axis=1)
        cols[f"{prof} SOC MWh"] = n.stores_t.e[sel(soc)].sum(axis=1)
    return pd.DataFrame(cols)


def plot_ev(df, title, path):
    profs = sorted({c.split(" ")[0] for c in df.columns})
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    colors = {"urban": "tab:purple", "suburban": "tab:green", "rural": "tab:blue",
              "a": "tab:orange", "b": "tab:red", "all": "tab:gray"}
    for p in profs:
        c = colors.get(p)
        ax[0].plot(df.index, df[f"{p} charge MW"], color=c, label=f"{p} charge")
        ax[0].plot(df.index, -df[f"{p} V2G MW"], "--", color=c, label=f"{p} V2G (neg.)")
        ax[1].plot(df.index, df[f"{p} SOC MWh"], color=c, label=p)
    ax[0].set_ylabel("MW")
    ax[1].set_ylabel("EV state of charge, MWh")
    ax[0].legend(ncol=3, fontsize=8)
    ax[1].legend(fontsize=8)
    ax[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- 4. peak timing
def lv_buses_by_profile(n):
    out = {}
    for b in n.buses.index[n.buses.carrier == "low voltage"]:
        out.setdefault(profile_of(b), []).append(b)
    return out


def flow_group(kind, carrier):
    c = str(carrier)
    if kind == "Load":
        return "electricity load"
    if kind == "Generator":
        return "rooftop PV" if "solar rooftop" in c else "other generation"
    if DIST in c:
        return "_dist"
    if "BEV charger" in c:
        return "EV charge"
    if c == "V2G":
        return "V2G"
    if "heat pump" in c:
        return "heat pump"
    if "resistive" in c:
        return "resistive heater"
    if "home battery" in c:
        return "home battery (net)"
    return "other"


def lv_withdrawals(n, buses):
    """Power taken from the given LV buses per snapshot and group.
    Positive = consumption, negative = injection. '_dist' is the distribution link
    (negative = net import from the HV side)."""
    buses = set(buses)
    parts = {}

    def add(group, series):
        parts[group] = parts[group] + series if group in parts else series

    ld = list(n.loads.index[n.loads.bus.isin(buses)])
    if ld:
        add("electricity load", n.loads_t.p[ld].sum(axis=1))
    gens = n.generators[n.generators.bus.isin(buses)]
    for carrier, idx in gens.groupby("carrier").groups.items():
        add(flow_group("Generator", carrier), -n.generators_t.p[list(idx)].sum(axis=1))
    for col in [c for c in n.links.columns if c.startswith("bus") and c[3:].isdigit()]:
        p = getattr(n.links_t, "p" + col[3:], None)
        if p is None or p.empty:
            continue
        sel = [l for l in n.links.index[n.links[col].isin(buses)] if l in p.columns]
        for carrier, idx in n.links.loc[sel].groupby("carrier").groups.items():
            add(flow_group("Link", carrier), p[list(idx)].sum(axis=1))
    return pd.DataFrame(parts)


def peak_timing(n):
    tables = {p: lv_withdrawals(n, b) for p, b in lv_buses_by_profile(n).items()}
    if len(tables) > 1:
        tables["TOTAL"] = pd.concat(list(tables.values())).groupby(level=0).sum()
    rows = {}
    for label, wd in tables.items():
        if "_dist" not in wd:
            continue
        imp = -wd["_dist"]  # net import into the LV level
        t = imp.idxmax()
        comp = wd.drop(columns="_dist")
        r = {"peak time": t, "net import MW": imp.loc[t]}
        r.update(comp.loc[t].to_dict())
        r["residual"] = imp.loc[t] - comp.loc[t].sum()  # should be ~0
        r["hours within 1% of peak"] = int((imp >= 0.99 * imp.max()).sum())  # >1 = peak time is a tie
        if "EV charge" in comp:
            r["EV charge peak time"] = comp["EV charge"].idxmax()
            r["EV share of import peak"] = (
                comp.loc[t, "EV charge"] / imp.loc[t] if imp.loc[t] > 1e-9 else np.nan
            )
        rows[label] = r
    return pd.DataFrame(rows).T


def peak_hours(n, tol=0.01, max_per_profile=8):
    """All snapshots within `tol` of each profile's import peak, with their components.
    A capacity-expansion LP often sits at the capacity limit in several hours, so a
    single 'peak time' can be one of several near-ties."""
    tables = {p: lv_withdrawals(n, b) for p, b in lv_buses_by_profile(n).items()}
    if len(tables) > 1:
        tables["TOTAL"] = pd.concat(list(tables.values())).groupby(level=0).sum()
    out = []
    for label, wd in tables.items():
        if "_dist" not in wd:
            continue
        imp = -wd["_dist"]
        near = imp[imp >= (1 - tol) * imp.max()].sort_values(ascending=False).index[:max_per_profile]
        near = sorted(near)
        comp = wd.drop(columns="_dist").loc[near].copy()
        comp.insert(0, "net import MW", imp.loc[near])
        comp.index = pd.MultiIndex.from_product([[label], comp.index], names=["profile", "time"])
        out.append(comp)
    return pd.concat(out) if out else pd.DataFrame()


# ---------------------------------------------------------------- 5. heat supply
HEAT_BUS_CARRIERS = ("urban central heat", "urban decentral heat", "rural heat")


def heat_supply(n):
    """Heat delivered to the heat buses by carrier, per heat system and profile (GWh over
    the period). Negative = heat taken out of the heat bus (e.g. tank charging).
    'balance' = supply - demand and should be ~0."""
    w = weights(n)
    heat_buses = n.buses.index[n.buses.carrier.isin(HEAT_BUS_CARRIERS)]
    key = {b: (n.buses.at[b, "carrier"].replace(" heat", ""), profile_of(b)) for b in heat_buses}
    rows = {}

    def add(label, bus, series):
        d = rows.setdefault(label, {})
        d[key[bus]] = d.get(key[bus], 0.0) + float(series.mul(w).sum())

    for l in n.loads.index[n.loads.bus.isin(heat_buses)]:
        add("HEAT DEMAND", n.loads.at[l, "bus"], -n.loads_t.p[l])  # shown as negative supply
    for g in n.generators.index[n.generators.bus.isin(heat_buses)]:
        add(n.generators.at[g, "carrier"], n.generators.at[g, "bus"], n.generators_t.p[g])
    for col in [c for c in n.links.columns if c.startswith("bus") and c[3:].isdigit()]:
        p = getattr(n.links_t, "p" + col[3:], None)
        if p is None or p.empty:
            continue
        for l in n.links.index[n.links[col].isin(heat_buses)]:
            if l in p.columns:
                add(n.links.at[l, "carrier"], n.links.at[l, col], -p[l])
    df = pd.DataFrame(rows).T.fillna(0.0) / 1e3
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["system", "profile"])
    df = df.sort_index(axis=1)
    df.loc["balance (supply - demand)"] = df.sum()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("networks", nargs="+")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--out", default="experiments/results/network_comparison")
    a = ap.parse_args()
    labels = a.labels or [os.path.splitext(os.path.basename(p))[0] for p in a.networks]
    os.makedirs(a.out, exist_ok=True)

    nets = {l: pypsa.Network(p) for l, p in zip(labels, a.networks)}
    for l, n in nets.items():
        sw = n.snapshot_weightings
        print(f"{l}: {len(n.snapshots)} snapshots, {sw.generators.sum():.0f} h covered "
              f"(objective weights sum {sw.objective.sum():.0f}); energies below are for this period only")

    dist = pd.concat({l: dist_grid(n) for l, n in nets.items()})
    dist.to_csv(f"{a.out}/dist_grid.csv")
    print("== 1. distribution grid\n", dist.round(2).to_string(), "\n")

    costs = pd.concat({l: costs_by_carrier(n) for l, n in nets.items()}, axis=1)
    costs.columns = [f"{l} {c}" for l, c in costs.columns]
    costs = costs.fillna(0.0)
    if len(labels) == 2:
        costs["delta total"] = costs[f"{labels[1]} total"] - costs[f"{labels[0]} total"]
    costs = costs.sort_values(costs.columns[2], ascending=False)
    costs.to_csv(f"{a.out}/costs_by_carrier.csv")

    focus = costs.copy()
    focus["group"] = focus.index.map(group_of)
    focus = focus.groupby("group")[[c for c in costs.columns if c.endswith("total")]].sum()
    focus.to_csv(f"{a.out}/costs_focus.csv")
    print("== 2. system costs (focus groups)\n", focus.round(0).to_string(), "\n")
    for l, n in nets.items():
        print(f"   {l}: rebuilt total {costs[f'{l} total'].sum():,.0f} vs objective {n.objective:,.0f}")

    print("\n== 3. EV operation")
    for l, n in nets.items():
        df = ev_operation(n)
        df.to_csv(f"{a.out}/ev_operation_{l}.csv")
        plot_ev(df, f"EV operation, {l}", f"{a.out}/ev_operation_{l}.png")
        w = weights(n)
        tot = {
            "charge GWh": sum(df[c].mul(w).sum() for c in df if c.endswith("charge MW")) / 1e3,
            "V2G GWh": sum(df[c].mul(w).sum() for c in df if c.endswith("V2G MW")) / 1e3,
        }
        print(f"   {l}: {tot}")
    print("\n== 4. what sets the distribution peak (components at each profile's import-peak snapshot, MW)")
    for l, n in nets.items():
        try:
            pk = peak_timing(n)
            pk.to_csv(f"{a.out}/peak_timing_{l}.csv")
            print(f"   {l}\n" + pk.to_string(float_format=lambda x: f"{x:,.2f}") + "\n")
        except Exception as e:  # keep the other results even if this section fails
            print(f"   {l}: peak timing failed ({type(e).__name__}: {e})")
        try:
            ph = peak_hours(n)
            ph.to_csv(f"{a.out}/peak_hours_{l}.csv")
            print(f"   {l}: snapshots within 1% of each profile's import peak (max 8 per profile)\n"
                  + ph.to_string(float_format=lambda x: f"{x:,.0f}") + "\n")
        except Exception as e:
            print(f"   {l}: peak hours failed ({type(e).__name__}: {e})")
    print("\n== 5. heat supply by technology (GWh over the period; demand shown negative)")
    for l, n in nets.items():
        try:
            hs = heat_supply(n)
            hs.to_csv(f"{a.out}/heat_supply_{l}.csv")
            print(f"   {l}\n" + hs.to_string(float_format=lambda x: f"{x:,.1f}") + "\n")
        except Exception as e:
            print(f"   {l}: heat supply failed ({type(e).__name__}: {e})")
    print(f"\nFiles written to {a.out}/")


if __name__ == "__main__":
    main()