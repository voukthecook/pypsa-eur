"""Focused SPLIT vs UNSPLIT comparison: distribution capacity, system costs, EV operation.

Usage:  python compare_focus.py split.nc unsplit.nc [node-prefix]
Needs:  split_deep.py in the same folder (it provides the shared helpers).
Writes: tables to ./compare_focus/*.csv, figures focus_distribution.png, focus_costs.png, focus_ev.png
Sections: A distribution capacity (per node and profile) | B system costs | C EV operation
A failing section never stops the others.
"""
import os
import re
import sys
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pypsa

from misc.split_deep import (Ctx, PROFILES, DIST, COLORS, profile_of, injections, lv_group, capex_of,
                        dist_for, stat_row)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
OUT = "compare_focus"
U, S = "unsplit", "split"


def hdr(t):
    print(f"\n{'=' * 4} {t} {'=' * 4}")


def save(df, name):
    try:
        df.to_csv(f"{OUT}/{name}.csv")
    except Exception:       # noqa: BLE001
        pass


def node_of(name):
    m = re.match(r"^(\S+ \d+)", str(name))
    return m.group(1) if m else "other"


def prep(n, prefix):
    c = Ctx()
    c.n, c.prefix, c.w = n, prefix, n.snapshot_weightings.generators
    c.dt = isinstance(n.snapshots, pd.DatetimeIndex)
    c.keep = lambda idx: [i for i in idx if str(i).startswith(prefix)]  # noqa: E731
    dl = n.links[n.links.carrier == DIST]
    c.dl = dl.loc[c.keep(dl.index)]
    rows = []
    for b in n.buses.index:
        if n.buses.at[b, "carrier"] == "low voltage" and b.startswith(prefix):
            p = profile_of(b)
            rows.append((b, node_of(b), "all" if p == "shared" else p))
    c.lvi = pd.DataFrame(rows, columns=["bus", "node", "profile"]).set_index("bus")
    return c


def grp_for(c, buses):
    tot = None
    for b in buses:
        i = injections(c.n, b)
        tot = i if tot is None else tot.add(i, fill_value=0)
    return tot.T.groupby(lambda k: lv_group(*k)).sum().T


def pc(a, b):
    return 100 * (b - a) / a if a else np.nan


# ================================================================ A. distribution
def rent_ratio(c, buses):
    mp = c.n.buses_t.marginal_price
    if mp.empty:
        return np.nan
    f = c.dl[c.dl.bus1.isin(buses)]
    num = den = 0.0
    for _, r in f.iterrows():
        mu = (r.efficiency * mp[r.bus1] - mp[r.bus0] - r.marginal_cost).clip(lower=0)
        num += (c.w * mu).sum() * r.p_nom_opt
        den += r.capital_cost * r.p_nom_opt
    return num / den if den else np.nan


def sec_a(cs, cu):
    nodes = sorted(set(cu.lvi.node) | set(cs.lvi.node))
    profs = [p for p in PROFILES if (cs.lvi.profile == p).any()]
    cap_rows, time_rows, hour_rows, pct_rows, misc_rows = {}, {}, {}, {}, {}
    comp = {}
    D = {}
    for node in nodes:
        bu = list(cu.lvi.index[cu.lvi.node == node])
        bs = list(cs.lvi.index[cs.lvi.node == node])
        du, ds = dist_for(cu, bu), dist_for(cs, bs)
        ru, rs = stat_row(cu, du), stat_row(cs, ds)
        D[node] = (du, ds)
        row = {"unsplit MW": ru["cap_in_MW(at LV)"], "split sum MW": rs["cap_in_MW(at LV)"]}
        row["diff MW"] = row["split sum MW"] - row["unsplit MW"]
        row["diff %"] = pc(row["unsplit MW"], row["split sum MW"])
        peaks = {}
        for p in profs:
            bp = [b for b in bs if cs.lvi.at[b, "profile"] == p]
            dp = dist_for(cs, bp)
            rp = stat_row(cs, dp)
            row[f"{p} MW"] = rp["cap_in_MW(at LV)"]
            row[f"{p} share %"] = 100 * rp["cap_in_MW(at LV)"] / max(rs["cap_in_MW(at LV)"], 1e-9)
            peaks[p] = (rp["peak_net_import_MW"], dp["net"].idxmax())
        row["sum of profile peaks / coincident"] = sum(v[0] for v in peaks.values()) / max(rs["peak_net_import_MW"], 1e-9)
        cap_rows[node] = row
        time_rows[node] = {"unsplit peak": str(du["net"].idxmax()), "split peak": str(ds["net"].idxmax()),
                           **{f"{p} peak": str(v[1]) for p, v in peaks.items()}}
        hour_rows[node] = {}
        for lab, r_, d_ in [(U, ru, du), (S, rs, ds)]:
            ci = r_["cap_in_MW(at LV)"]
            imp = d_["net"].clip(lower=0)
            for thr in (0.95, 0.8, 0.5):
                hour_rows[node][f"{lab} h>{int(thr * 100)}%"] = cu.w[imp >= thr * ci].sum() if ci > 0 else 0.0
        pct_rows[node] = {}
        for lab, d_ in [(U, du), (S, ds)]:
            for q in (0.5, 0.9, 0.95, 0.99):
                pct_rows[node][f"{lab} p{int(q * 100)}"] = d_["net"].quantile(q)
        misc_rows[node] = {"unsplit gross export GWh": ru["gross_export_GWh"], "split gross export GWh": rs["gross_export_GWh"],
                           "unsplit net import GWh": ru["net_import_GWh"], "split net import GWh": rs["net_import_GWh"],
                           "unsplit h both dirs": ru["h_both_directions"], "split h both dirs": rs["h_both_directions"],
                           "unsplit rent/capital": rent_ratio(cu, bu), "split rent/capital": rent_ratio(cs, bs)}
        gu, gs = grp_for(cu, bu), grp_for(cs, bs)
        tu, ts = du["net"].idxmax(), ds["net"].idxmax()
        topu = du["net"].nlargest(max(1, int(0.02 * len(du["net"])))).index
        tops = ds["net"].nlargest(max(1, int(0.02 * len(ds["net"])))).index
        comp[node] = pd.DataFrame({"unsplit @peak": gu.loc[tu], "split @peak": gs.loc[ts],
                                   "unsplit top-2% mean": gu.loc[topu].mean(), "split top-2% mean": gs.loc[tops].mean()})
    hdr("A1. Distribution capacity at LV side (MW) per node: unsplit vs split (by profile)")
    t = pd.DataFrame(cap_rows).T
    print(t.round(2).to_string())
    save(t, "A1_capacity")
    hdr("A2. Timing of each node's peak (net import)")
    print(pd.DataFrame(time_rows).T.to_string())
    hdr("A3. Hours near the cap, per node")
    print(pd.DataFrame(hour_rows).T.round(0).to_string())
    hdr("A4. Net import percentiles per node (MW)")
    pt = pd.DataFrame(pct_rows).T
    print(pt.round(0).to_string())
    save(pt, "A4_percentiles")
    hdr("A5. Exports, imports and rent per node")
    print(pd.DataFrame(misc_rows).T.round(2).to_string())
    hdr("A6. What sets each node's peak (MW; + supply to LV, - withdrawal)")
    for node, d in comp.items():
        print(f"\n[{node}]")
        print(d.round(0).to_string())
    save(pd.concat(comp, axis=1), "A6_peak_composition")
    # profile level caps vs shares (split)
    hdr("A7. Split profile capacities vs the profile shares (all nodes)")
    tot = t[["split sum MW"]].sum().iloc[0]
    prow = {p: {"MW": t[f"{p} MW"].sum(), "% of total": 100 * t[f"{p} MW"].sum() / tot} for p in profs}
    print(pd.DataFrame(prow).T.round(1).to_string())
    return D, t, comp


def fig_a(cs, cu, D, t):
    nodes = list(D)
    fig, ax = plt.subplots(2, max(2, len(nodes)), figsize=(6 * max(2, len(nodes)), 9))
    for j, node in enumerate(nodes):
        du, ds = D[node]
        for lab, d_, col in [(U, du, "k"), (S, ds, "#D85A30")]:
            s = d_["net"].clip(lower=0).sort_values(ascending=False).values
            ax[0, j].plot(np.linspace(0, 100, len(s)), s, color=col, label=lab)
        ax[0, j].set_title(f"{node}: import duration curve (MW)")
        ax[0, j].legend()
    cols = [c for c in t.columns if c.endswith(" MW") and c not in ("diff MW",)]
    t[cols].plot.bar(ax=ax[1, 0], rot=0)
    ax[1, 0].set_title("Capacity at LV side by node (MW)")
    ax[1, 0].legend(fontsize=7)
    tot = pd.DataFrame({U: pd.concat([D[n_][0]["net"] for n_ in nodes], axis=1).sum(axis=1),
                        S: pd.concat([D[n_][1]["net"] for n_ in nodes], axis=1).sum(axis=1)})
    (tot[S] - tot[U]).plot(ax=ax[1, 1], lw=.6)
    ax[1, 1].axhline(0, c="k", lw=.5)
    ax[1, 1].set_title("Aggregate net import: split minus unsplit (MW)")
    fig.tight_layout()
    fig.savefig("focus_distribution.png", dpi=140)


# ================================================================ B. costs
def cat(car):
    c = str(car).lower()
    if "distribution" in c:
        return "distribution grid"
    if c in ("bev charger", "v2g", "ev battery"):
        return "EV (chargers, V2G, battery)"
    if "home battery" in c:
        return "home battery"
    if "heat pump" in c or "resistive" in c:
        return "electric heating"
    if "boiler" in c:
        return "gas boilers"
    if "chp" in c:
        return "CHP"
    if c == "dac":
        return "DAC"
    if "co2" in c:
        return "CO2 handling"
    if "solar thermal" in c or "vent" in c or "water" in c or "tank" in c or "pit" in c:
        return "heat storage / other heat"
    if c == "solar rooftop":
        return "rooftop PV"
    if c.startswith("solar"):
        return "utility PV"
    if "wind" in c:
        return "wind"
    if c in ("hydro", "ror", "phs"):
        return "hydro"
    if c in ("battery", "battery charger", "battery discharger"):
        return "HV battery"
    if c in ("gas", "gas pipeline", "gas pipeline new", "oil primary", "oil refining", "land transport oil", "oil"):
        return "fuel supply and networks"
    if c == "ac":
        return "AC lines"
    return "other"


def cost_rows(n, prefix):
    w = n.snapshot_weightings.generators
    rows = []
    for comp, a in [("generators", "p_nom_opt"), ("links", "p_nom_opt"), ("stores", "e_nom_opt"),
                    ("storage_units", "p_nom_opt"), ("lines", "s_nom_opt")]:
        df = getattr(n, comp)
        df = df.loc[[i for i in df.index if str(i).startswith(prefix)]]
        if df.empty:
            continue
        capex = capex_of(comp, df, a)
        flow = {"generators": n.generators_t.p, "links": n.links_t.p0,
                "storage_units": n.storage_units_t.p_dispatch, "stores": n.stores_t.p}.get(comp)
        opex = pd.Series(0.0, index=df.index)
        if flow is not None:
            cols = [x for x in df.index if x in flow.columns]
            opex[cols] = flow[cols].mul(w, axis=0).mul(df.loc[cols, "marginal_cost"], axis=1).sum()
        rows.append(pd.DataFrame({"comp": comp, "carrier": df.carrier.values, "node": [node_of(i) for i in df.index],
                                  "capex": capex.values, "opex": opex.values}, index=df.index))
    r = pd.concat(rows)
    r["category"] = r.carrier.map(cat)
    return r


def sec_b(cs, cu):
    ru, rs = cost_rows(cu.n, cu.prefix), cost_rows(cs.n, cs.prefix)
    hdr("B1. Objective and its parts (EUR)")
    ob = pd.DataFrame({U: [cu.n.objective, ru.capex.sum(), ru.opex.sum()], S: [cs.n.objective, rs.capex.sum(), rs.opex.sum()]},
                      index=["objective", "capex (above installed min)", "opex"])
    ob["diff"] = ob[S] - ob[U]
    ob["diff %"] = 100 * ob["diff"] / ob[U]
    print(ob.round(3).to_string(float_format=lambda x: f"{x:,.3f}" if abs(x) < 100 else f"{x:,.0f}"))
    hdr("B2. By category (EUR): capex, opex, total")
    g = pd.concat({U: ru.groupby("category")[["capex", "opex"]].sum(), S: rs.groupby("category")[["capex", "opex"]].sum()}, axis=1).fillna(0)
    out = pd.DataFrame({"unsplit total": g[(U, "capex")] + g[(U, "opex")], "split total": g[(S, "capex")] + g[(S, "opex")]})
    out["diff"] = out["split total"] - out["unsplit total"]
    out["diff %"] = 100 * out["diff"] / out["unsplit total"].where(out["unsplit total"].abs() > 1)
    out["capex diff"] = g[(S, "capex")] - g[(U, "capex")]
    out["opex diff"] = g[(S, "opex")] - g[(U, "opex")]
    out["share of objective %"] = 100 * out["unsplit total"] / cu.n.objective
    out = out.sort_values("share of objective %", ascending=False)
    print(out.round(2).to_string(float_format=lambda x: f"{x:,.2f}" if abs(x) < 1000 else f"{x:,.0f}"))
    save(out, "B2_category")
    hdr("B3. Carriers with the largest absolute change (EUR)")
    cr = pd.concat({U: ru.groupby("carrier")[["capex", "opex"]].sum().sum(axis=1),
                    S: rs.groupby("carrier")[["capex", "opex"]].sum().sum(axis=1)}, axis=1).fillna(0)
    cr["diff"] = cr[S] - cr[U]
    cr["diff %"] = 100 * cr["diff"] / cr[U].where(cr[U].abs() > 1)
    top = cr.reindex(cr["diff"].abs().sort_values(ascending=False).index).head(15)
    print(top.round(1).to_string(float_format=lambda x: f"{x:,.1f}" if abs(x) < 1000 else f"{x:,.0f}"))
    tot_abs = cr["diff"].abs().sum()
    print(f"\nsum of |carrier changes| = {tot_abs:,.0f} EUR = {100 * tot_abs / cu.n.objective:.4f}% of the objective; "
          f"gains {cr['diff'][cr['diff'] < 0].sum():,.0f}, losses {cr['diff'][cr['diff'] > 0].sum():,.0f}")
    save(cr, "B3_carrier")
    hdr("B4. Change by node and category (EUR, split - unsplit)")
    gn = pd.concat({U: ru.assign(t=ru.capex + ru.opex).groupby(["node", "category"]).t.sum(),
                    S: rs.assign(t=rs.capex + rs.opex).groupby(["node", "category"]).t.sum()}, axis=1).fillna(0)
    gn["diff"] = gn[S] - gn[U]
    print(gn["diff"].unstack().fillna(0).round(0).to_string(float_format=lambda x: f"{x:,.0f}"))
    nd = gn.groupby(level=0).sum()
    print("\nper node totals:")
    print(nd.assign(**{"diff %": 100 * nd["diff"] / nd[U]}).round(1).to_string(float_format=lambda x: f"{x:,.1f}" if abs(x) < 1000 else f"{x:,.0f}"))
    save(gn, "B4_node_category")
    hdr("B5. Distribution grid cost and what it delivers")
    rows = {}
    for lab, c, r in [(U, cu, ru), (S, cs, rs)]:
        d = dist_for(c, list(c.lvi.index))
        gwh = d["net"].clip(lower=0).mul(c.w).sum() / 1e3
        cost = r[r.category == "distribution grid"][["capex", "opex"]].sum().sum()
        rows[lab] = {"cost EUR": cost, "net import GWh": gwh, "EUR per MWh": cost / max(gwh * 1e3, 1e-9),
                     "cap MW (LV side)": d["cap_in"], "EUR per MW installed": cost / max(d["cap_in"], 1e-9)}
    print(pd.DataFrame(rows).round(2).to_string(float_format=lambda x: f"{x:,.2f}"))
    return out, cr


def fig_b(out, cr):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    out["diff"].sort_values().plot.barh(ax=ax[0])
    ax[0].set_title("Cost change by category, split - unsplit (EUR)")
    cr["diff"].reindex(cr["diff"].abs().sort_values().tail(15).index).plot.barh(ax=ax[1])
    ax[1].set_title("Largest carrier changes (EUR)")
    ax[1].tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig("focus_costs.png", dpi=140)


# ================================================================ C. EV
def ev_data(c):
    n, w = c.n, c.w
    ch = n.links[n.links.carrier == "BEV charger"]
    v2 = n.links[n.links.carrier == "V2G"]
    ch, v2 = ch.loc[c.keep(ch.index)], v2.loc[c.keep(v2.index)]
    st = n.stores[n.stores.carrier == "EV battery"]
    st = st.loc[c.keep(st.index)]
    ld = n.loads[n.loads.carrier == "land transport EV"]
    ld = ld.loc[c.keep(ld.index)]
    D = {"ch": ch, "v2": v2, "st": st}
    D["charge"] = n.links_t.p0[ch.index].sum(axis=1)
    D["v2g"] = -n.links_t.p1[v2.index].sum(axis=1)
    D["v2g_batt"] = n.links_t.p0[v2.index].sum(axis=1)
    D["soc"] = n.stores_t.e[st.index].sum(axis=1)
    D["enom"] = st.e_nom_opt.sum()
    dem = n.loads_t.p[ld.index].sum(axis=1)
    eff = ch.efficiency.mean() if len(ch) else 1.0
    D["baseline"] = dem / eff            # unmanaged charging that would just cover demand
    D["demand"] = dem
    mp = n.buses_t.marginal_price
    if not mp.empty:
        paid = sum(n.links_t.p0[l] * mp[ch.at[l, "bus0"]] for l in ch.index)
        recv = sum(-n.links_t.p1[l] * mp[v2.at[l, "bus1"]] for l in v2.index) if len(v2) else 0 * w
        lvp = pd.concat([mp[b] for b in ch.bus0.unique()], axis=1).mean(axis=1)
        D["paid"], D["recv"], D["lvp"] = paid, recv, lvp
    return D


def sec_c(cs, cu):
    E = {U: ev_data(cu), S: ev_data(cs)}
    w = cu.w
    net = {U: dist_for(cu, list(cu.lvi.index))["net"], S: dist_for(cs, list(cs.lvi.index))["net"]}
    top = {k: net[k].nlargest(max(1, int(0.02 * len(net[k])))).index for k in E}
    rows = {}
    for k, d in E.items():
        ch, v = d["charge"], d["v2g"]
        r = {"charge GWh": (ch * w).sum() / 1e3, "V2G delivered GWh": (v * w).sum() / 1e3,
             "V2G / charge %": 100 * (v * w).sum() / max((ch * w).sum(), 1e-9),
             "charger MW": d["ch"].p_nom_opt.sum(), "peak charge MW": ch.max(),
             "mean utilisation %": 100 * ch.mean() / max(d["ch"].p_nom_opt.sum(), 1e-9),
             "battery GWh": d["enom"] / 1e3, "equiv cycles": (ch * w).sum() / max(d["enom"], 1e-9),
             "mean SOC %": 100 * d["soc"].mean() / max(d["enom"], 1e-9),
             "shifted vs unmanaged %": 100 * ((ch - d["baseline"]).clip(lower=0) * w).sum() / max((ch * w).sum(), 1e-9),
             "charge in top-2% import hours MW": ch[top[k]].mean(), "V2G in top-2% import hours MW": v[top[k]].mean(),
             "share of charge in top-2% hours %": 100 * (ch[top[k]] * w[top[k]]).sum() / max((ch * w).sum(), 1e-9)}
        if "paid" in d:
            r["mean price paid EUR/MWh"] = (d["paid"] * w).sum() / max((ch * w).sum(), 1e-9)
            r["time-avg LV price EUR/MWh"] = d["lvp"].mean()
            r["charging saving vs avg price %"] = 100 * (1 - r["mean price paid EUR/MWh"] / d["lvp"].mean())
            r["V2G mean sell price EUR/MWh"] = (d["recv"] * w).sum() / max((v * w).sum(), 1e-9)
            r["V2G revenue MEUR"] = (d["recv"] * w).sum() / 1e6
            r["charging cost MEUR"] = (d["paid"] * w).sum() / 1e6
        rows[k] = r
    t = pd.DataFrame(rows)
    t["diff"] = t[S] - t[U]
    t["diff %"] = 100 * t["diff"] / t[U].where(t[U].abs() > 1e-9)
    hdr("C1. EV operation, annual metrics")
    print(t.round(2).to_string())
    save(t, "C1_ev_metrics")

    a, b = E[U]["charge"], E[S]["charge"]
    d = b - a
    hdr("C2. Similarity of the charging time series (aggregate)")
    print(pd.Series({"correlation": a.corr(b), "RMS diff MW": float(np.sqrt((d ** 2).mean())),
                     "energy displaced GWh (sum|diff|/2)": (d.abs() * w).sum() / 2e3,
                     "as % of charging": 100 * (d.abs() * w).sum() / 2 / max((a * w).sum(), 1e-9),
                     "V2G correlation": E[U]["v2g"].corr(E[S]["v2g"])}).round(3).to_string())
    prof = {}
    if cu.dt:
        n_ = cu.n
        hod = pd.DataFrame({f"{k} {m}": E[k][m].groupby(n_.snapshots.hour).mean()
                            for k in (U, S) for m in ("charge", "v2g", "baseline")})
        hod["charge diff"] = hod[f"{S} charge"] - hod[f"{U} charge"]
        hdr("C3. Average day (MW by hour of day)")
        print(hod.round(0).to_string())
        save(hod, "C3_avg_day")
        mon = pd.DataFrame({f"{k} {m}": (E[k][m] * w).groupby(n_.snapshots.month).sum() / 1e3
                            for k in (U, S) for m in ("charge", "v2g")})
        mon["charge diff GWh"] = mon[f"{S} charge"] - mon[f"{U} charge"]
        mon["V2G diff GWh"] = mon[f"{S} v2g"] - mon[f"{U} v2g"]
        hdr("C4. By month (GWh)")
        print(mon.round(1).to_string())
        save(mon, "C4_monthly")
        seas = {}
        for lab, months in [("winter (Dec-Feb)", [12, 1, 2]), ("summer (Jun-Aug)", [6, 7, 8])]:
            msk = n_.snapshots.month.isin(months)
            for k in (U, S):
                seas[(lab, k)] = E[k]["charge"][msk].groupby(n_.snapshots.hour[msk]).mean()
        hdr("C5. Average-day charging by season (MW)")
        print(pd.DataFrame(seas).round(0).to_string())
        pk = net[U].idxmax()
        win = (net[U].index >= pk - pd.Timedelta(days=3)) & (net[U].index <= pk + pd.Timedelta(days=3))
        wk = pd.DataFrame({f"{k} {m}": E[k][m][win] for k in (U, S) for m in ("charge", "v2g")})
        wk["unsplit net import"], wk["split net import"] = net[U][win], net[S][win]
        hdr(f"C6. Around the grid peak ({pk}): +/- 3 days, daily means (MW)")
        print(wk.resample("D").mean().round(0).to_string())
        prof["wk"] = wk
        prof["hod"] = hod
    hdr("C7. Split profiles: do the profiles charge differently? (split network only)")
    ch = cs.n.links[cs.n.links.carrier == "BEV charger"]
    ch = ch.loc[cs.keep(ch.index)]
    pr = {}
    series = {}
    for p in PROFILES:
        names = [x for x in ch.index if profile_of(ch.at[x, "bus0"]) == p]
        if not names:
            continue
        s = cs.n.links_t.p0[names].sum(axis=1)
        series[p] = s
        cap = ch.loc[names, "p_nom_opt"].sum()
        pr[p] = {"charge GWh": (s * w).sum() / 1e3, "charger MW": cap, "peak MW": s.max(),
                 "peak / charger %": 100 * s.max() / max(cap, 1e-9), "mean util %": 100 * s.mean() / max(cap, 1e-9)}
    print(pd.DataFrame(pr).T.round(1).to_string())
    if len(series) > 1:
        sdf = pd.DataFrame(series)
        print("\ncorrelation between profiles' charging:\n", sdf.corr().round(4).to_string())
        tot_s = sdf.sum(axis=1)
        sh = sdf.div(tot_s.where(tot_s > 1), axis=0)
        print("\nprofile share of aggregate charging over time (min / mean / max):")
        print(pd.DataFrame({"min": sh.min(), "mean": sh.mean(), "max": sh.max()}).round(3).to_string())
    return E, prof, t


def fig_c(E, prof, cu):
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    if "hod" in prof:
        h = prof["hod"]
        for k, col in [(U, "k"), (S, "#D85A30")]:
            ax[0, 0].plot(h.index, h[f"{k} charge"], color=col, label=f"{k} charge")
            ax[0, 0].plot(h.index, h[f"{k} v2g"], "--", color=col, label=f"{k} V2G")
        ax[0, 0].plot(h.index, h[f"{U} baseline"], ":", color="grey", label="unmanaged baseline")
        ax[0, 0].set_title("Average day (MW)")
        ax[0, 0].legend(fontsize=7)
        ax[0, 1].plot(h.index, h["charge diff"])
        ax[0, 1].axhline(0, c="k", lw=.5)
        ax[0, 1].set_title("Average-day charging: split - unsplit (MW)")
    if "wk" in prof:
        wk = prof["wk"]
        for k, col in [(U, "k"), (S, "#D85A30")]:
            ax[0, 2].plot(wk.index, wk[f"{k} charge"], color=col, label=f"{k} charge")
            ax[1, 0].plot(wk.index, wk[f"{k} net import"], color=col, label=k)
        ax[0, 2].set_title("EV charging around the grid peak (MW)")
        ax[0, 2].legend()
        ax[1, 0].set_title("Net import around the grid peak (MW)")
        ax[1, 0].legend()
    for k, col in [(U, "k"), (S, "#D85A30")]:
        s = E[k]["charge"].sort_values(ascending=False).values
        ax[1, 1].plot(np.linspace(0, 100, len(s)), s, color=col, label=k)
    ax[1, 1].set_title("EV charging duration curve (MW)")
    ax[1, 1].legend()
    for k, col in [(U, "k"), (S, "#D85A30")]:
        soc = 100 * E[k]["soc"] / max(E[k]["enom"], 1e-9)
        ax[1, 2].plot(soc.values, color=col, lw=.4, label=k)
    ax[1, 2].set_title("EV state of charge (% of battery)")
    ax[1, 2].legend()
    fig.tight_layout()
    fig.savefig("focus_ev.png", dpi=140)


# ================================================================ main
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ncs = [a for a in args if a.endswith(".nc")]
    if len(ncs) < 2:
        sys.exit(__doc__)
    prefix = [a for a in args if not a.endswith(".nc")]
    prefix = prefix[0] if prefix else ""
    os.makedirs(OUT, exist_ok=True)
    ns, nu = pypsa.Network(ncs[0]), pypsa.Network(ncs[1])
    cs, cu = prep(ns, prefix), prep(nu, prefix)
    if not ns.snapshots.equals(nu.snapshots):
        print("WARNING: the two networks have different snapshots; time-series comparisons will be misaligned")
    for name, fn in [("A", lambda: fig_a(cs, cu, *sec_a(cs, cu)[:2])),
                     ("B", lambda: fig_b(*sec_b(cs, cu))),
                     ("C", lambda: fig_c(*sec_c(cs, cu)[:2], cu))]:
        try:
            fn()
        except Exception as e:                      # noqa: BLE001
            print(f"\n[section {name}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
    print(f"\nCSVs in ./{OUT}/, figures focus_distribution.png, focus_costs.png, focus_ev.png")


if __name__ == "__main__":
    main()