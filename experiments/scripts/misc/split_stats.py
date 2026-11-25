"""Statistics of a solved 3-profile split network (lv_split_profiles).

Usage:  python split_stats.py split.nc [unsplit.nc] [node-prefix]
        node-prefix is optional ("GR0 0"); default = whole network.
        With unsplit.nc, section 7 compares totals split vs unsplit in %.
Needs:  pip install pypsa pandas matplotlib

Writes tables and figures to experiments/results/findings/split_stats/.

Sections
  1. Capacities by carrier x profile (and each profile's % of the carrier total)
  2. Distribution links: import/export capacity, peaks, loading, hours at limit
  3. Energy balance of each profile's LV bus (supply / demand by carrier, GWh)
  4. Heat balance of each profile's heat buses
  5. Cost by profile and carrier (capex + opex, sanity-checked against objective)
  6. EV: charging, V2G, peak per profile and coincident peak
Conventions: profile = last word of a component/bus name (urban, suburban, rural),
anything else = "shared". Positive flow in balances = supply into the bus.
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import pypsa

PROFILES = ["urban", "suburban", "rural"]
DIST = "electricity distribution grid"
EV_CHARGER, EV_V2G = "BEV charger", "V2G"
OUT = "experiments/results/findings/split_stats"


def capex_of(comp, df, attr):
    """Capex as the objective pays it: extendable assets only, and (except stores)
    only the capacity above the installed minimum -- that part is objective_constant."""
    ext = df[attr.replace("_opt", "_extendable")].astype(bool)
    opt = df[attr].replace([float("inf"), -float("inf")], 0)
    lo = 0 if comp == "stores" else df[attr.replace("_opt", "_min")].replace([float("inf")], 0)
    return ((opt - lo).clip(lower=0) * df.capital_cost * ext).fillna(0)


def profile_of(name):
    s = str(name).strip().lower()
    for p in PROFILES:
        if s.endswith(" " + p):
            return p
    return "shared"


def port_cols(n):
    return [c for c in n.links.columns if c.startswith("bus") and c[3:].isdigit()]


def link_profile(n, links):
    ports = port_cols(n)
    out = []
    for _, r in links.iterrows():
        p = next((profile_of(r[c]) for c in ports
                  if isinstance(r[c], str) and r[c] and profile_of(r[c]) != "shared"), None)
        out.append(p or profile_of(r.name))
    return pd.Series(out, index=links.index)


def injections(n, bus):
    """Time series of power injected INTO `bus` by every attached component,
    one column per (component type, carrier). Loads are negative."""
    cols = {}

    def add(key, s):
        cols[key] = cols[key] + s if key in cols else s

    g = n.generators[n.generators.bus == bus]
    for name, r in g.iterrows():
        add(("generator", r.carrier), n.generators_t.p[name])
    for name, r in n.loads[n.loads.bus == bus].iterrows():
        add(("load", r.carrier), -n.loads_t.p[name])
    for name, r in n.stores[n.stores.bus == bus].iterrows():
        add(("store", r.carrier), n.stores_t.p[name])
    for name, r in n.storage_units[n.storage_units.bus == bus].iterrows():
        add(("storage unit", r.carrier), n.storage_units_t.p[name])
    for port in port_cols(n):
        i = port[3:]
        sel = n.links[n.links[port] == bus]
        if i == "0":
            flow = n.links_t.p0
        else:
            flow = getattr(n.links_t, "p" + i)
        for name, r in sel.iterrows():
            add(("link", r.carrier), -flow[name])
    return pd.DataFrame(cols)


def balance_table(n, buses, w):
    rows = {}
    for b in buses:
        inj = injections(n, b)
        if inj.empty:
            continue
        gwh = inj.mul(w, axis=0).sum() / 1e3
        gwh.index = [f"{c} | {k}" for c, k in gwh.index]
        rows[b] = gwh
    return pd.DataFrame(rows).fillna(0)


def totals(n, prefix):
    """Profile-free totals so split and unsplit are directly comparable."""
    w = n.snapshot_weightings.generators
    keep = lambda df: df.loc[[i for i in df.index if str(i).startswith(prefix)]]
    cap = {}
    for comp, attr in [("generators", "p_nom_opt"), ("links", "p_nom_opt"), ("stores", "e_nom_opt")]:
        df = keep(getattr(n, comp))
        for car, v in df.groupby("carrier")[attr].sum().items():
            cap[(comp, car)] = v
    dl = keep(n.links[n.links.carrier == DIST])
    into = 0 * w
    for name, r in dl.iterrows():
        lv_is_1 = n.buses.at[r.bus1, "carrier"] == "low voltage"
        into = into - (n.links_t.p1[name] if lv_is_1 else n.links_t.p0[name])
    imp = into.clip(lower=0)
    chg = keep(n.links[n.links.carrier == EV_CHARGER]).index
    v2g = keep(n.links[n.links.carrier == EV_V2G]).index
    c = n.links_t.p0[chg].sum(axis=1)
    key = {
        "distribution cap (MW)": dl.p_nom_opt.sum(),
        "coincident peak import (MW)": imp.max(),
        "import (GWh)": (imp * w).sum() / 1e3,
        "EV charge (GWh)": (c * w).sum() / 1e3,
        "EV V2G (GWh)": (n.links_t.p0[v2g].sum(axis=1) * w).sum() / 1e3,
        "EV coincident peak (MW)": c.max(),
        "objective (EUR)": n.objective,
    }
    cost = {}
    for comp, attr in [("generators", "p_nom_opt"), ("links", "p_nom_opt"), ("stores", "e_nom_opt"),
                       ("storage_units", "p_nom_opt")]:
        df = keep(getattr(n, comp))
        capex = capex_of(comp, df, attr)
        flow = {"generators": n.generators_t.p, "links": n.links_t.p0,
                "storage_units": n.storage_units_t.p_dispatch,
                "stores": n.stores_t.p}.get(comp)
        opex = pd.Series(0.0, index=df.index)
        if flow is not None:
            cols = [x for x in df.index if x in flow.columns]
            opex[cols] = flow[cols].mul(w, axis=0).mul(df.loc[cols, "marginal_cost"], axis=1).sum()
        for car, v in (capex + opex).groupby(df.carrier).sum().items():
            cost[car] = cost.get(car, 0) + v
    return key, pd.Series(cap), pd.Series(cost)


def compare(split, unsplit, prefix):
    pct = lambda a, b: 100 * (b - a) / a if a else float("nan")
    ks, cs, ds = totals(split, prefix)
    ku, cu, du = totals(unsplit, prefix)
    t = pd.DataFrame({"unsplit": ku, "split": ks})
    t["diff_%"] = [pct(a, b) for a, b in zip(t.unsplit, t.split)]
    print("\n== 7a. Key totals: split vs unsplit ==")
    print(t.round(2).to_string())
    c = pd.concat([cu, cs], axis=1, keys=["unsplit", "split"]).fillna(0)
    c = c[c.abs().max(axis=1) > 1e-6]
    c["diff_%"] = [pct(a, b) for a, b in zip(c.unsplit, c.split)]
    c["diff_abs"] = c.split - c.unsplit
    c = c.reindex(c.diff_abs.abs().sort_values(ascending=False).index)
    print("\n== 7b. Installed capacity by carrier (MW / MWh), largest absolute change first ==")
    print(c.head(25).round(2).to_string())
    k = pd.concat([du, ds], axis=1, keys=["unsplit", "split"]).fillna(0)
    k = k[k.abs().max(axis=1) > 0.005 * k.abs().sum().max()]
    k["diff_%"] = [pct(a, b) for a, b in zip(k.unsplit, k.split)]
    k["diff_EUR"] = k.split - k.unsplit
    k = k.reindex(k.diff_EUR.abs().sort_values(ascending=False).index)
    print("\n== 7c. Cost by carrier (EUR) ==")
    print(k.round(1).to_string())
    t.round(3).to_csv(f"{OUT}/7a_totals_vs_unsplit.csv")
    c.round(3).to_csv(f"{OUT}/7b_capacity_vs_unsplit.csv")
    k.round(1).to_csv(f"{OUT}/7c_cost_vs_unsplit.csv")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    t["diff_%"].drop("objective (EUR)").plot.bar(ax=ax[0], rot=30, title="Key totals: split vs unsplit, %")
    k["diff_%"].plot.bar(ax=ax[1], title="Cost by carrier: split vs unsplit, %")
    for a in ax:
        a.axhline(0, c="k", lw=.8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/split_vs_unsplit.png", dpi=150)


def main():
    args = sys.argv[1:]
    ncs = [a for a in args if a.endswith(".nc")]
    rest = [a for a in args if not a.endswith(".nc")]
    path = ncs[0]
    unsplit_path = ncs[1] if len(ncs) > 1 else None
    prefix = rest[0] if rest else ""
    os.makedirs(OUT, exist_ok=True)
    n = pypsa.Network(path)
    w = n.snapshot_weightings.generators
    keep = lambda names: [x for x in names if str(x).startswith(prefix)]

    # ---- 1. capacities ---------------------------------------------------------
    rows = []
    for comp, attr, unit in [("generators", "p_nom_opt", "MW"), ("links", "p_nom_opt", "MW"),
                             ("stores", "e_nom_opt", "MWh")]:
        df = getattr(n, comp)
        df = df.loc[keep(df.index)]
        prof = link_profile(n, df) if comp == "links" else df.bus.map(profile_of)
        t = pd.DataFrame({"carrier": df.carrier, "profile": prof, "value": df[attr], "unit": unit,
                          "component": comp})
        rows.append(t)
    cap = pd.concat(rows)
    cap_t = cap.pivot_table(index=["component", "carrier", "unit"], columns="profile",
                            values="value", aggfunc="sum", fill_value=0)
    for p in PROFILES + ["shared"]:
        if p not in cap_t:
            cap_t[p] = 0.0
    cap_t = cap_t[PROFILES + ["shared"]]
    cap_t["total"] = cap_t.sum(axis=1)
    pct = cap_t[PROFILES].div(cap_t[PROFILES].sum(axis=1).replace(0, float("nan")), axis=0) * 100
    pct.columns = [f"{c} %" for c in PROFILES]
    cap_out = pd.concat([cap_t, pct], axis=1)
    cap_out = cap_out[cap_out["total"].abs() > 1e-6]
    cap_out.round(2).to_csv(f"{OUT}/1_capacities.csv")
    print("\n== 1. Capacities by carrier x profile (% = profile's share of the profile total) ==")
    print(cap_out.round(1).to_string())

    # ---- 2. distribution links --------------------------------------------------
    dl = n.links[(n.links.carrier == DIST)]
    dl = dl.loc[keep(dl.index)]
    dprof = link_profile(n, dl)
    ts = {}
    drows = {}
    for p in PROFILES:
        lv_in_cap = lv_out_cap = 0.0
        into_lv = 0 * w
        for name in dl.index[dprof == p]:
            r = dl.loc[name]
            if profile_of(r.bus1) == p:      # AC -> LV (import)
                lv_in_cap += r.p_nom_opt * r.efficiency      # delivered MW at LV
                into_lv = into_lv - n.links_t.p1[name]
            else:                            # LV -> AC (export link, LV at bus0)
                lv_out_cap += r.p_nom_opt
                into_lv = into_lv - n.links_t.p0[name]
        ts[p] = into_lv
        imp = into_lv.clip(lower=0)
        exp = (-into_lv).clip(lower=0)
        drows[p] = {
            "import_cap_MW(at LV)": lv_in_cap, "export_cap_MW": lv_out_cap,
            "peak_import_MW": imp.max(), "peak_export_MW": exp.max(),
            "import_GWh": (imp * w).sum() / 1e3, "export_GWh": (exp * w).sum() / 1e3,
            "load_factor_import_%": 100 * (imp * w).sum() / max(lv_in_cap * w.sum(), 1e-9),
            "peak/cap_%": 100 * imp.max() / max(lv_in_cap, 1e-9),
            "hours_>95%_of_cap": float(((imp >= 0.95 * lv_in_cap) & (lv_in_cap > 0)).mul(w).sum()),
        }
    dist = pd.DataFrame(drows).T
    ts_df = pd.DataFrame(ts)
    tot = ts_df.sum(axis=1)
    dist.loc["sum of profiles"] = dist.sum()
    dist.loc["sum of profiles", ["peak_import_MW", "peak_export_MW"]] = [
        tot.clip(lower=0).max(), (-tot).clip(lower=0).max()]
    dist.loc["sum of profiles", ["load_factor_import_%", "hours_>95%_of_cap", "peak/cap_%"]] = float("nan")
    dist.round(2).to_csv(f"{OUT}/2_distribution.csv")
    ts_df.to_csv(f"{OUT}/2_net_import_to_LV_timeseries.csv")
    print("\n== 2. Distribution links (peak of the 'sum' row is the COINCIDENT peak) ==")
    print(dist.round(1).to_string())
    print("   sum of individual peaks:", round(dist.loc[PROFILES, "peak_import_MW"].sum(), 1),
          "MW  vs coincident:", round(dist.loc["sum of profiles", "peak_import_MW"], 1), "MW")

    # ---- 3. LV balance ----------------------------------------------------------
    lv = [b for b in n.buses.index if n.buses.at[b, "carrier"] == "low voltage" and b.startswith(prefix)]
    bal = balance_table(n, lv, w)
    bal.columns = [profile_of(c) if profile_of(c) != "shared" else c for c in bal.columns]
    bal = bal.T.groupby(level=0).sum().T
    bal.round(2).to_csv(f"{OUT}/3_LV_balance_GWh.csv")
    print("\n== 3. LV bus energy balance, GWh (+ supply into LV, - demand/withdrawal) ==")
    print(bal.round(1).to_string())

    # ---- 4. heat balance --------------------------------------------------------
    hb = [b for b in n.buses.index if "heat" in n.buses.at[b, "carrier"] and b.startswith(prefix)]
    heat = balance_table(n, hb, w)
    heat.columns = [str(c) for c in heat.columns]
    heat.round(2).to_csv(f"{OUT}/4_heat_balance_GWh.csv")
    by_p = heat.T.groupby(lambda c: profile_of(c)).sum().T
    print("\n== 4. Heat balance (all heat buses summed per profile), GWh ==")
    print(by_p.round(1).to_string())

    # ---- 5. cost by profile -----------------------------------------------------
    crow = []
    for comp, attr in [("generators", "p_nom_opt"), ("links", "p_nom_opt"),
                       ("stores", "e_nom_opt"), ("storage_units", "p_nom_opt"), ("lines", "s_nom_opt")]:
        df = getattr(n, comp)
        df = df.loc[keep(df.index)]
        if df.empty:
            continue
        capex = capex_of(comp, df, attr)
        flow = {"generators": n.generators_t.p, "links": n.links_t.p0,
                "storage_units": n.storage_units_t.p_dispatch,
                "stores": n.stores_t.p}.get(comp)
        opex = pd.Series(0.0, index=df.index)
        if flow is not None:
            cols = [c for c in df.index if c in flow.columns]
            opex[cols] = flow[cols].mul(w, axis=0).mul(df.loc[cols, "marginal_cost"], axis=1).sum()
        if comp == "links":
            prof = link_profile(n, df)
        elif comp == "lines":
            prof = pd.Series("shared", index=df.index)
        else:
            prof = df.bus.map(profile_of)
        crow.append(pd.DataFrame({"carrier": df.carrier, "profile": prof, "capex": capex, "opex": opex}))
    cost = pd.concat(crow)
    cost["total"] = cost.capex + cost.opex
    cp = cost.pivot_table(index="carrier", columns="profile", values="total", aggfunc="sum", fill_value=0)
    for p in PROFILES + ["shared"]:
        if p not in cp:
            cp[p] = 0.0
    cp = cp[PROFILES + ["shared"]]
    cp["total"] = cp.sum(axis=1)
    cp = cp.sort_values("total", ascending=False)
    cp.round(0).to_csv(f"{OUT}/5_cost_by_profile.csv")
    top = cp[cp.total.abs() > 0.005 * cp.total.abs().sum()]
    print("\n== 5. Cost by carrier x profile (EUR, capex + opex), carriers > 0.5% ==")
    print(top.round(0).to_string())
    print(f"   component-sum {cost.total.sum():,.0f} vs objective {n.objective:,.0f} "
          "(should now match to ~0)")
    pc = cost.groupby("profile").total.sum()
    print("   per profile:", {k: f"{v / pc.sum() * 100:.1f}%" for k, v in pc.items()})

    # ---- 6. EV -------------------------------------------------------------------
    ev = n.links[n.links.carrier.isin([EV_CHARGER, EV_V2G])]
    ev = ev.loc[keep(ev.index)]
    eprof = link_profile(n, ev)
    erow, ets = {}, {}
    for p in PROFILES:
        ch = ev.index[(eprof == p) & (ev.carrier == EV_CHARGER)]
        vg = ev.index[(eprof == p) & (ev.carrier == EV_V2G)]
        c = n.links_t.p0[ch].sum(axis=1)
        v = n.links_t.p0[vg].sum(axis=1)       # V2G: bus0 = EV battery, p0 = battery output
        ets[p] = c
        erow[p] = {"charge_GWh": (c * w).sum() / 1e3, "v2g_GWh": (v * w).sum() / 1e3,
                   "charger_cap_MW": n.links.loc[ch, "p_nom_opt"].sum(),
                   "peak_charge_MW": c.max(),
                   "charge_at_peak_LV_%": 100 * c[ts_df[p].idxmax()] / max(c.max(), 1e-9)}
    evt = pd.DataFrame(erow).T
    ev_ts = pd.DataFrame(ets)
    evt.loc["sum of profiles"] = evt.sum()
    evt.loc["sum of profiles", "peak_charge_MW"] = ev_ts.sum(axis=1).max()
    evt.loc["sum of profiles", "charge_at_peak_LV_%"] = float("nan")
    evt.round(2).to_csv(f"{OUT}/6_ev.csv")
    ev_ts.to_csv(f"{OUT}/6_ev_charging_timeseries.csv")
    print("\n== 6. EV per profile ==")
    print(evt.round(1).to_string())

    # ---- figure -----------------------------------------------------------------
    colors = {"urban": "#1D9E75", "suburban": "#D85A30", "rural": "#534AB7"}
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    ts_df.plot(ax=ax[0, 0], color=[colors[c] for c in ts_df.columns], lw=1)
    ax[0, 0].axhline(0, c="k", lw=.6)
    ax[0, 0].set_title("Net import into each LV bus (MW, negative = export)")
    dist.loc[PROFILES, ["import_cap_MW(at LV)", "peak_import_MW"]].plot.bar(ax=ax[0, 1], rot=0)
    ax[0, 1].set_title("Distribution: import capacity vs peak import (MW)")
    top.drop(columns=["total", "shared"]).head(12).plot.bar(
        ax=ax[1, 0], stacked=True, color=[colors[c] for c in PROFILES], width=.8)
    ax[1, 0].set_title("Cost by carrier and profile (EUR)")
    ax[1, 0].tick_params(axis="x", labelsize=7)
    ev_ts.plot(ax=ax[1, 1], color=[colors[c] for c in ev_ts.columns], lw=1)
    ax[1, 1].set_title("EV charging per profile (MW)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/split_stats.png", dpi=150)
    if unsplit_path:
        compare(n, pypsa.Network(unsplit_path), prefix)
    print(f"\nTables and figures in {OUT}/")


if __name__ == "__main__":
    main()