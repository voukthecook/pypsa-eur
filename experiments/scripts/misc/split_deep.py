"""Deep dive into ONE solved 3-profile split network.

Usage:  python -m experiments.scripts.misc.split_deep split.nc [unsplit.nc] [node-prefix] [--only-compare]
        With unsplit.nc: also prints SPLIT vs UNSPLIT comparisons (section C below).
Needs:  pip install pypsa pandas numpy matplotlib

Prints 14 sections, writes CSVs and figures to experiments/results/findings/split_deep/.

 1 Run basics + global constraints         8 Heat per system and profile (+ COP)
 2 Distribution: gross/net, losses, stats  9 DAC
 3 Per-node distribution                  10 EV
 4 What sets each profile's peak          11 Home battery
 5 LV annual balance per profile          12 Rooftop PV
 6 Months and shape of the profiles       13 Shared HV (AC) bus balance
 7 Prices and the value of capacity       14 Distribution cost per delivered MWh
Positive flow in balances = supply into the bus. Profile = last word of a name.
Loads are negative. Energies are weighted by the snapshot weightings.
"""
import os
import sys
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pypsa

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

PROFILES = ["urban", "suburban", "rural"]
COLORS = {"urban": "#1D9E75", "suburban": "#D85A30", "rural": "#534AB7"}
DIST = "electricity distribution grid"
OUT = "experiments/results/findings/split_deep"


# ---------------------------------------------------------------- helpers
def profile_of(name):
    s = str(name).strip().lower()
    for p in PROFILES:
        if s.endswith(" " + p):
            return p
    return "shared"


def port_cols(n):
    return [c for c in n.links.columns if c.startswith("bus") and c[3:].isdigit()]


def link_profile(n, df):
    ports = port_cols(n)
    out = []
    for _, r in df.iterrows():
        p = next((profile_of(r[c]) for c in ports
                  if isinstance(r[c], str) and r[c] and profile_of(r[c]) != "shared"), None)
        out.append(p or profile_of(r.name))
    return pd.Series(out, index=df.index, dtype=object)


def injections(n, bus):
    """Power injected INTO `bus` by each attached component; one column per
    (component type, carrier). Loads are negative."""
    cols = {}

    def add(key, s):
        cols[key] = cols[key] + s if key in cols else s

    for name, r in n.generators[n.generators.bus == bus].iterrows():
        add(("generator", r.carrier), n.generators_t.p[name])
    for name, r in n.loads[n.loads.bus == bus].iterrows():
        add(("load", r.carrier), -n.loads_t.p[name])
    for name, r in n.stores[n.stores.bus == bus].iterrows():
        add(("store", r.carrier), n.stores_t.p[name])
    for name, r in n.storage_units[n.storage_units.bus == bus].iterrows():
        add(("storage unit", r.carrier), n.storage_units_t.p[name])
    for port in port_cols(n):
        flow = n.links_t.p0 if port == "bus0" else getattr(n.links_t, "p" + port[3:])
        for name, r in n.links[n.links[port] == bus].iterrows():
            add(("link", r.carrier), -flow[name])
    return pd.DataFrame(cols)


def balance_table(n, buses, w):
    rows = {}
    for b in buses:
        inj = injections(n, b)
        if inj.empty:
            continue
        gwh = inj.mul(w, axis=0).sum() / 1e3
        gwh.index = [f"{k} | {c}" for k, c in gwh.index]
        rows[b] = gwh
    return pd.DataFrame(rows).fillna(0)


def lv_group(kind, carrier):
    c = str(carrier).lower()
    if kind == "generator" and "rooftop" in c:
        return "rooftop PV"
    if kind == "load":
        return "base electricity load" if "electricity" in c else f"load: {carrier}"
    if c == "bev charger":
        return "EV charging"
    if c == "v2g":
        return "V2G"
    if "home battery" in c:
        return "home battery (net)"
    if "resistive" in c:
        return "resistive heating"
    if "heat pump" in c:
        return "heat pumps"
    if "distribution" in c:
        return "distribution (net import)"
    return f"other: {carrier}"


def capex_of(comp, df, attr):
    """Capex as the objective pays it (extendable only; capacity above the installed
    minimum, except stores which are charged in full)."""
    ext = df[attr.replace("_opt", "_extendable")].astype(bool)
    opt = df[attr].replace([float("inf"), -float("inf")], 0)
    lo = 0 if comp == "stores" else df[attr.replace("_opt", "_min")].replace([float("inf")], 0)
    return ((opt - lo).clip(lower=0) * df.capital_cost * ext).fillna(0)


def save(df, name):
    try:
        df.to_csv(f"{OUT}/{name}.csv")
    except Exception:           # noqa: BLE001
        pass


def hdr(t):
    print(f"\n{'=' * 4} {t} {'=' * 4}")


# ---------------------------------------------------------------- context
class Ctx:
    pass


def build_ctx(n, prefix):
    c = Ctx()
    c.n, c.prefix = n, prefix
    c.w = n.snapshot_weightings.generators
    c.dt = isinstance(n.snapshots, pd.DatetimeIndex)
    keep = lambda idx: [i for i in idx if str(i).startswith(prefix)]  # noqa: E731
    c.keep = keep
    c.lv = {p: [b for b in n.buses.index if n.buses.at[b, "carrier"] == "low voltage"
                and profile_of(b) == p and b.startswith(prefix)] for p in PROFILES}
    c.inj = {}
    for p in PROFILES:
        tot = None
        for b in c.lv[p]:
            i = injections(n, b)
            tot = i if tot is None else tot.add(i, fill_value=0)
        c.inj[p] = tot if tot is not None else pd.DataFrame(index=n.snapshots)
    c.grp = {p: c.inj[p].T.groupby(lambda k: lv_group(*k)).sum().T for p in PROFILES}
    dl = n.links[n.links.carrier == DIST]
    c.dl = dl.loc[keep(dl.index)]
    return c


def dist_for(c, buses):
    n, w = c.n, c.w
    f = c.dl[c.dl.bus1.isin(buses)]
    r = c.dl[c.dl.bus0.isin(buses)]
    imp_hv = n.links_t.p0[f.index].sum(axis=1)
    imp_d = -n.links_t.p1[f.index].sum(axis=1)
    exp_lv = n.links_t.p0[r.index].sum(axis=1)
    exp_d = -n.links_t.p1[r.index].sum(axis=1)
    return dict(imp_hv=imp_hv, imp_d=imp_d, exp_lv=exp_lv, exp_d=exp_d, net=imp_d - exp_lv,
                cap_in=(f.p_nom_opt * f.efficiency).sum(), cap_out=r.p_nom_opt.sum(), f=f, r=r)


def stat_row(c, d):
    w = c.w
    net, ci = d["net"], d["cap_in"]
    imp = net.clip(lower=0)
    exp = (-net).clip(lower=0)
    both = (d["imp_d"] > 1) & (d["exp_lv"] > 1)
    q = net.quantile([0.5, 0.95, 0.99])
    return {
        "cap_in_MW(at LV)": ci, "cap_out_MW": d["cap_out"],
        "peak_net_import_MW": imp.max(), "peak/cap_%": 100 * imp.max() / max(ci, 1e-9),
        "p99_MW": q[0.99], "p95_MW": q[0.95], "median_MW": q[0.5], "mean_MW": net.mean(),
        "load_factor_%": 100 * (imp * w).sum() / max(ci * w.sum(), 1e-9),
        "h_>95%cap": w[imp >= 0.95 * ci].sum() if ci > 0 else 0.0,
        "h_>80%cap": w[imp >= 0.80 * ci].sum() if ci > 0 else 0.0,
        "net_import_GWh": (imp * w).sum() / 1e3, "net_export_GWh": (exp * w).sum() / 1e3,
        "gross_import_GWh": (d["imp_d"] * w).sum() / 1e3, "gross_export_GWh": (d["exp_lv"] * w).sum() / 1e3,
        "h_both_directions": w[both].sum(),
        "export_in_both_GWh": (d["exp_lv"][both] * w[both]).sum() / 1e3,
        "losses_GWh": ((d["imp_hv"] - d["imp_d"]) * w).sum() / 1e3 + ((d["exp_lv"] - d["exp_d"]) * w).sum() / 1e3,
    }


def binding_mask(c, p):
    d = c.D[p]
    imp = d["net"].clip(lower=0)
    return (imp >= 0.95 * d["cap_in"]) if d["cap_in"] > 0 else pd.Series(False, index=imp.index)


# ---------------------------------------------------------------- sections
def s1(c):
    n, w = c.n, c.w
    hdr("1. Run basics")
    print(f"snapshots: {len(n.snapshots)}  {n.snapshots[0]} -> {n.snapshots[-1]}")
    print(f"weight sum (generators): {w.sum():.0f} h   objective: {n.objective:,.0f}")
    gc = n.global_constraints
    print(gc[[x for x in ["type", "carrier_attribute", "sense", "constant", "mu"] if x in gc.columns]].to_string())
    print("buses:", len(n.buses), " links:", len(n.links), " LV buses per profile:",
          {p: len(c.lv[p]) for p in PROFILES})


def s2(c):
    hdr("2. Distribution links per profile (LV-side figures; gross vs net)")
    c.D = {p: dist_for(c, c.lv[p]) for p in PROFILES}
    rows = {p: stat_row(c, c.D[p]) for p in PROFILES}
    tot_net = sum(c.D[p]["net"] for p in PROFILES)
    cap = sum(c.D[p]["cap_in"] for p in PROFILES)
    tot = {k: sum(rows[p][k] for p in PROFILES) for k in
           ["cap_in_MW(at LV)", "cap_out_MW", "net_import_GWh", "net_export_GWh", "gross_import_GWh",
            "gross_export_GWh", "h_both_directions", "export_in_both_GWh", "losses_GWh"]}
    tot["peak_net_import_MW"] = tot_net.clip(lower=0).max()
    tot["peak/cap_%"] = 100 * tot["peak_net_import_MW"] / max(cap, 1e-9)
    tot["mean_MW"] = tot_net.mean()
    t = pd.DataFrame(rows).T
    t.loc["ALL (coincident)"] = pd.Series(tot)
    print(t.round(1).T.to_string())
    ind = sum(rows[p]["peak_net_import_MW"] for p in PROFILES)
    print(f"\nsum of individual peaks {ind:,.1f} MW  vs coincident {tot['peak_net_import_MW']:,.1f} MW "
          f"-> diversity factor {ind / max(tot['peak_net_import_MW'], 1e-9):.4f}")
    print("hours with BOTH import and export flowing = distribution losses used as an energy sink "
          "(see 'h_both_directions', 'export_in_both_GWh').")
    save(t, "2_distribution")
    c.tot_net = tot_net


def s3(c):
    hdr("3. Per node (LV bus)")
    rows = {}
    for p in PROFILES:
        for b in c.lv[p]:
            d = dist_for(c, [b])
            r = stat_row(c, d)
            rows[b] = {k: r[k] for k in ["cap_in_MW(at LV)", "peak_net_import_MW", "peak/cap_%",
                                         "load_factor_%", "h_>95%cap", "net_import_GWh", "net_export_GWh"]}
    t = pd.DataFrame(rows).T
    print(t.round(1).to_string())
    save(t, "3_per_node")


def s4(c):
    hdr("4. What sets each profile's import peak (MW; + = supply to LV, - = withdrawal)")
    allt = {}
    for p in PROFILES:
        g = c.grp[p]
        net = g["distribution (net import)"] if "distribution (net import)" in g else c.D[p]["net"]
        tpk = net.idxmax()
        bind = binding_mask(c, p)
        cols = {"at peak": g.loc[tpk], "mean in >95%-cap hours": g[bind].mean() if bind.any() else g.mean() * np.nan,
                "annual mean": g.mean()}
        allt[p] = pd.DataFrame(cols)
        print(f"\n[{p}] peak at {tpk}, {c.w[bind].sum():.0f} h within 5% of the capacity")
        print(allt[p].round(1).to_string())
        chk = g.loc[tpk].sum()
        if abs(chk) > 1:
            print(f"  (balance residual at peak: {chk:.1f} MW - other components not captured)")
    save(pd.concat(allt, axis=1), "4_peak_composition")
    c.peak_comp = allt


def s5(c):
    hdr("5. LV annual balance per profile (GWh)")
    t = pd.DataFrame({p: c.grp[p].mul(c.w, axis=0).sum() / 1e3 for p in PROFILES}).fillna(0)
    t.loc["TOTAL (should be ~0)"] = t.sum()
    print(t.round(1).to_string())
    wd = t.drop(index="TOTAL (should be ~0)")
    wd = wd.where(wd < 0, 0).abs()
    share = 100 * wd / wd.sum()
    print("\nshare of gross LV withdrawals (%):")
    print(share.round(1).to_string())
    save(t, "5_lv_balance")
    c.lv_gwh = t


def s6(c):
    hdr("6. Months and shape of the profiles")
    n = c.n
    if c.dt:
        mon = pd.DataFrame({p: c.D[p]["net"].clip(lower=0).groupby(n.snapshots.month).max() for p in PROFILES})
        en = pd.DataFrame({p: (c.D[p]["net"].clip(lower=0) * c.w).groupby(n.snapshots.month).sum() / 1e3 for p in PROFILES})
        near = pd.DataFrame({p: c.w[binding_mask(c, p)].groupby(n.snapshots.month[binding_mask(c, p).values]).sum()
                             for p in PROFILES}).reindex(range(1, 13)).fillna(0)
        print("monthly peak net import (MW):\n", mon.round(0).to_string())
        print("\nmonthly net import (GWh):\n", en.round(0).to_string())
        print("\nhours within 5% of capacity per month:\n", near.round(0).to_string())
        save(mon, "6_monthly_peak")
        c.mon = mon
    nets = pd.DataFrame({p: c.D[p]["net"] for p in PROFILES})
    print("\ncorrelation of net import between profiles:\n", nets.corr().round(4).to_string())
    tot = nets.sum(axis=1)
    ok = tot > 0.05 * tot.max()
    sh = nets[ok].div(tot[ok], axis=0)
    st = pd.DataFrame({"mean_share": sh.mean(), "min": sh.min(), "max": sh.max(), "std": sh.std()})
    print("\nshare of total net import over time (constant share = same shape, only scale differs):")
    print(st.round(4).to_string())
    print("\ntime of each profile's peak:", {p: str(nets[p].idxmax()) for p in PROFILES})
    save(st, "6_import_share_stats")
    c.share_ts = sh


def s7(c):
    hdr("7. Prices and the value of distribution capacity")
    n, w = c.n, c.w
    mp = n.buses_t.marginal_price
    if mp.empty:
        print("no marginal prices in this network (duals not stored) - skipped")
        return
    rows = {}
    rent_rows = {}
    for p in PROFILES:
        f = c.dl[c.dl.bus1.isin(c.lv[p])]
        if f.empty:
            continue
        lvp = pd.concat([mp[b] for b in c.lv[p]], axis=1).mean(axis=1)
        acp = pd.concat([mp[b] for b in f.bus0.unique()], axis=1).mean(axis=1)
        imp = c.D[p]["net"].clip(lower=0)
        spread = ((lvp - acp) * imp).sum() / max(imp.sum(), 1e-9)
        rows[p] = {"LV mean": lvp.mean(), "LV p95": lvp.quantile(.95), "LV max": lvp.max(),
                   "AC mean": acp.mean(), "import-weighted LV-AC spread": spread,
                   "h LV>AC+5": w[(lvp - acp) > 5].sum()}
        num = den = 0.0
        for nm, r in f.iterrows():
            mu = (r.efficiency * mp[r.bus1] - mp[r.bus0] - r.marginal_cost).clip(lower=0)
            rent = (w * mu).sum()
            num += rent * r.p_nom_opt
            den += r.capital_cost * r.p_nom_opt
        rent_rows[p] = {"capacity rent per MW (EUR/MW/yr)": num / max(f.p_nom_opt.sum(), 1e-9),
                        "capital cost per MW": den / max(f.p_nom_opt.sum(), 1e-9),
                        "rent / capital": num / max(den, 1e-9)}
    print(pd.DataFrame(rows).T.round(2).to_string())
    print("\nrent / capital close to 1 = capacity is priced at its cost (no external limit); "
          "above 1 = a limit is binding.")
    print(pd.DataFrame(rent_rows).T.round(2).to_string())
    c.prices = (mp, rows)
    save(pd.DataFrame(rows).T, "7_prices")


def heat_tables(c):
    n, w = c.n, c.w
    hb = [b for b in n.buses.index if "heat" in str(n.buses.at[b, "carrier"]) and b.startswith(c.prefix)]
    tab = balance_table(n, hb, w)
    keys = [(n.buses.at[b, "carrier"], profile_of(b)) for b in tab.columns]
    tab.columns = pd.MultiIndex.from_tuples(keys, names=["system", "profile"])
    return tab.T.groupby(level=[0, 1]).sum().T


def s8(c):
    hdr("8. Heat per system and profile (GWh)")
    h = heat_tables(c)
    c.heat = h
    show = h.loc[(h.abs() > 0.05).any(axis=1)]
    print(show.round(1).to_string())
    save(h, "8_heat_balance")

    def pick(df, pred):
        rows = [i for i in df.index if pred(i)]
        return df.loc[rows].sum() if rows else df.iloc[0] * 0

    summ = pd.DataFrame({
        "household demand": -pick(h, lambda i: i.startswith("load")),
        "DAC heat": -pick(h, lambda i: i.endswith("| DAC")),
        "heat pumps": pick(h, lambda i: "heat pump" in i),
        "resistive": pick(h, lambda i: "resistive" in i),
        "gas boilers": pick(h, lambda i: "boiler" in i),
        "CHP": pick(h, lambda i: "CHP" in i),
        "storage net": pick(h, lambda i: "water" in i),
        "solar thermal": pick(h, lambda i: "solar thermal" in i),
        "vent": pick(h, lambda i: "vent" in i),
    })
    summ["DAC / household %"] = 100 * summ["DAC heat"] / summ["household demand"].replace(0, np.nan)
    print("\nsummary by system x profile:")
    print(summ.round(1).to_string())
    save(summ, "8_heat_summary")
    c.heat_summ = summ
    # COP = heat supplied / LV electricity withdrawn, per heat-pump carrier
    out = {}
    for p in PROFILES:
        inj = c.inj[p]
        for (kind, car) in inj.columns:
            if kind == "link" and "heat pump" in str(car):
                elec = -(inj[(kind, car)] * c.w).sum() / 1e3
                heat = sum(h.loc[f"link | {car}", (s, p)] for s in h.columns.get_level_values(0).unique()
                           if (s, p) in h.columns and f"link | {car}" in h.index)
                if abs(elec) > 1e-6:
                    out[(car, p)] = {"LV electricity GWh": elec, "heat GWh": heat, "COP": heat / elec}
    if out:
        print("\nheat pump COP (seasonal):")
        print(pd.DataFrame(out).T.round(2).to_string())


def s9(c):
    hdr("9. DAC")
    n, w = c.n, c.w
    d = n.links[n.links.carrier == "DAC"]
    d = d.loc[c.keep(d.index)]
    if d.empty:
        print("no DAC links")
        return
    prof = link_profile(n, d)
    rows = {}
    for p in PROFILES:
        nm = d.index[prof == p]
        if not len(nm):
            continue
        cap = d.loc[nm, "p_nom_opt"].sum()
        e = (n.links_t.p0[nm].sum(axis=1) * w).sum() / 1e3
        co2 = 0.0
        for port in port_cols(n):
            for l in nm:
                b = d.at[l, port]
                if isinstance(b, str) and "atmosphere" in b:
                    df = n.links_t.p0 if port == "bus0" else getattr(n.links_t, "p" + port[3:])
                    if l in df.columns:
                        co2 += abs((df[l] * w).sum()) / 1e3
        rows[p] = {"capacity MW": cap, "electricity from AC GWh": e,
                   "full-load hours": (n.links_t.p0[nm].sum(axis=1) * w).sum() / max(cap, 1e-9),
                   "CO2 captured (kt, if t/h)": co2}
    t = pd.DataFrame(rows).T
    if hasattr(c, "heat_summ"):
        dh = c.heat_summ["DAC heat"].groupby(level=1).sum()
        t["heat withdrawn GWh"] = dh.reindex(t.index)
    print(t.round(1).to_string())
    save(t, "9_dac")


def s10(c):
    hdr("10. EV")
    n, w = c.n, c.w
    ch = n.links[n.links.carrier == "BEV charger"]
    v2 = n.links[n.links.carrier == "V2G"]
    ch, v2 = ch.loc[c.keep(ch.index)], v2.loc[c.keep(v2.index)]
    pch, pv2 = link_profile(n, ch), link_profile(n, v2)
    st = n.stores[n.stores.carrier == "EV battery"]
    st = st.loc[c.keep(st.index)]
    pst = st.bus.map(profile_of)
    rows, avgday = {}, {}
    c.ev = {}
    for p in PROFILES:
        cn, vn = ch.index[pch == p], v2.index[pv2 == p]
        charge = n.links_t.p0[cn].sum(axis=1)
        v2g_b = n.links_t.p0[vn].sum(axis=1)
        v2g_d = -n.links_t.p1[vn].sum(axis=1)
        cap = ch.loc[cn, "p_nom_opt"].sum()
        bind = binding_mask(c, p)
        enom = st.loc[st.index[pst == p], "e_nom_opt"].sum()
        soc = n.stores_t.e[st.index[pst == p]].sum(axis=1) / max(enom, 1e-9) * 100 if enom > 0 else None
        rows[p] = {
            "charge GWh": (charge * w).sum() / 1e3, "V2G from battery GWh": (v2g_b * w).sum() / 1e3,
            "V2G delivered GWh": (v2g_d * w).sum() / 1e3, "charger MW": cap,
            "peak charge MW": charge.max(), "mean utilisation %": 100 * charge.mean() / max(cap, 1e-9),
            "battery GWh": enom / 1e3, "equiv. full cycles/period": (charge * w).sum() / max(enom, 1e-9),
            "mean SOC %": soc.mean() if soc is not None else np.nan,
            "% of charge in binding hours": 100 * (charge[bind] * w[bind]).sum() / max((charge * w).sum(), 1e-9),
            "% of hours binding": 100 * w[bind].sum() / w.sum(),
            "V2G MW in binding hours (mean)": v2g_d[bind].mean() if bind.any() else np.nan,
        }
        c.ev[p] = (charge, v2g_d)
        if c.dt:
            avgday[(p, "charge MW")] = charge.groupby(n.snapshots.hour).mean()
            avgday[(p, "V2G MW")] = v2g_d.groupby(n.snapshots.hour).mean()
    t = pd.DataFrame(rows).T
    print(t.round(1).T.to_string())
    save(t, "10_ev")
    if avgday:
        ad = pd.DataFrame(avgday)
        print("\naverage day (MW by hour of day):")
        print(ad.round(0).to_string())
        save(ad, "10_ev_avg_day")


def s11(c):
    hdr("11. Home battery")
    n, w = c.n, c.w
    cl = n.links[n.links.carrier == "home battery charger"]
    dl_ = n.links[n.links.carrier == "home battery discharger"]
    cl, dl_ = cl.loc[c.keep(cl.index)], dl_.loc[c.keep(dl_.index)]
    st = n.stores[n.stores.carrier == "home battery"]
    st = st.loc[c.keep(st.index)]
    pc, pd_, ps = link_profile(n, cl), link_profile(n, dl_), st.bus.map(profile_of)
    rows, avgday = {}, {}
    for p in PROFILES:
        cn, dn = cl.index[pc == p], dl_.index[pd_ == p]
        chg = n.links_t.p0[cn].sum(axis=1)
        dis = -n.links_t.p1[dn].sum(axis=1)
        enom = st.loc[st.index[ps == p], "e_nom_opt"].sum()
        bind = binding_mask(c, p)
        rows[p] = {"energy GWh": enom / 1e3, "charger MW": cl.loc[cn, "p_nom_opt"].sum(),
                   "discharger MW": dl_.loc[dn, "p_nom_opt"].sum(),
                   "discharge GWh": (dis * w).sum() / 1e3,
                   "equiv. full cycles/period": (dis * w).sum() / max(enom, 1e-9),
                   "duration h (energy/discharger)": enom / max(dl_.loc[dn, "p_nom_opt"].sum(), 1e-9),
                   "discharge MW in binding hours (mean)": dis[bind].mean() if bind.any() else np.nan,
                   "% of discharge in binding hours": 100 * (dis[bind] * w[bind]).sum() / max((dis * w).sum(), 1e-9)}
        if c.dt:
            avgday[(p, "net discharge MW")] = (dis - chg).groupby(n.snapshots.hour).mean()
    print(pd.DataFrame(rows).T.round(1).T.to_string())
    save(pd.DataFrame(rows).T, "11_home_battery")
    if avgday:
        print("\naverage day, net discharge MW (+ = discharging):")
        print(pd.DataFrame(avgday).round(0).to_string())


def s12(c):
    hdr("12. Rooftop PV")
    n, w = c.n, c.w
    g = n.generators[n.generators.carrier == "solar rooftop"]
    g = g.loc[c.keep(g.index)]
    pg = g.bus.map(profile_of)
    rows = {}
    for p in PROFILES:
        names = g.index[pg == p]
        if not len(names):
            continue
        cap = g.loc[names, "p_nom_opt"].sum()
        gen = n.generators_t.p[names].sum(axis=1)
        pot = pd.Series(0.0, index=n.snapshots)
        for nm in names:
            pm = n.generators_t.p_max_pu[nm] if nm in n.generators_t.p_max_pu.columns else g.at[nm, "p_max_pu"]
            pot = pot + pm * g.at[nm, "p_nom_opt"]
        net = c.D[p]["net"]
        exp = (-net).clip(lower=0)
        nmax = g.loc[names, "p_nom_max"].replace(np.inf, np.nan).sum() if "p_nom_max" in g else np.nan
        rows[p] = {"capacity MW": cap, "p_nom_max MW": nmax, "at potential limit %": 100 * cap / nmax if nmax else np.nan,
                   "generation GWh": (gen * w).sum() / 1e3,
                   "capacity factor %": 100 * (gen * w).sum() / max(cap * w.sum(), 1e-9),
                   "curtailed GWh": ((pot - gen).clip(lower=0) * w).sum() / 1e3,
                   "net export GWh": (exp * w).sum() / 1e3,
                   "export / generation %": 100 * (exp * w).sum() / max((gen * w).sum(), 1e-9)}
    t = pd.DataFrame(rows).T
    print(t.round(1).T.to_string())
    save(t, "12_rooftop_pv")


def s13(c):
    hdr("13. Shared HV (AC) bus balance, GWh")
    n, w = c.n, c.w
    ac = [b for b in n.buses.index if n.buses.at[b, "carrier"] == "AC" and b.startswith(c.prefix)]
    t = balance_table(n, ac, w).sum(axis=1)
    t = t[t.abs() > 0.001 * t.abs().max()].sort_values()
    supply, demand = t[t > 0], t[t < 0]
    print("SUPPLY into HV:\n", supply.sort_values(ascending=False).round(0).to_string())
    print("\nDEMAND from HV:\n", demand.round(0).to_string())
    tot_d = demand.abs().sum()
    dist = demand[[i for i in demand.index if "distribution" in i]].abs().sum()
    print(f"\ndistribution links take {100 * dist / max(tot_d, 1e-9):.1f}% of all HV demand "
          f"({dist:,.0f} of {tot_d:,.0f} GWh)")
    save(t.to_frame("GWh"), "13_ac_balance")
    c.ac = t
    fuels = {}
    for car in ["gas", "oil primary"]:
        gg = n.generators[(n.generators.carrier == car)]
        gg = gg.loc[c.keep(gg.index)]
        if len(gg):
            fuels[car] = (n.generators_t.p[gg.index].sum(axis=1) * w).sum() / 1e3
    print("\nfuel supply GWh:", {k: round(v) for k, v in fuels.items()})
    seq = n.links[n.links.carrier == "co2 sequestered"]
    seq = seq.loc[c.keep(seq.index)]
    if len(seq):
        print("CO2 sequestered (kt, if t/h):", round((n.links_t.p0[seq.index].sum(axis=1) * w).sum() / 1e3))


def s14(c):
    hdr("14. Distribution cost per delivered MWh")
    n = c.n
    rows = {}
    for p in PROFILES:
        f = c.dl[c.dl.bus1.isin(c.lv[p]) | c.dl.bus0.isin(c.lv[p])]
        capex = capex_of("links", f, "p_nom_opt").sum()
        opex = (n.links_t.p0[f.index].mul(c.w, axis=0).mul(f.marginal_cost, axis=1)).sum().sum()
        gwh = (c.D[p]["net"].clip(lower=0) * c.w).sum() / 1e3
        rows[p] = {"capex+opex EUR": capex + opex, "net import GWh": gwh,
                   "EUR per MWh imported": (capex + opex) / max(gwh * 1e3, 1e-9)}
    t = pd.DataFrame(rows).T
    print(t.round(2).to_string())
    save(t, "14_dist_cost")


# ---------------------------------------------------------------- figures
def figures(c):
    n, w = c.n, c.w
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    for p in PROFILES:
        d = c.D[p]
        s = (d["net"].clip(lower=0) / max(d["cap_in"], 1e-9) * 100).sort_values(ascending=False).values
        ax[0, 0].plot(np.linspace(0, 100, len(s)), s, color=COLORS[p], label=p)
    ax[0, 0].set_title("Import duration curve (% of LV-side capacity)")
    ax[0, 0].set_xlabel("% of time")
    ax[0, 0].legend()
    if c.dt:
        for p in PROFILES:
            m = c.D[p]["net"].clip(lower=0).groupby(n.snapshots.month).max()
            ax[0, 1].plot(m.index, m.values, "o-", color=COLORS[p], label=p)
            h = c.D[p]["net"].groupby(n.snapshots.hour).mean()
            ax[0, 2].plot(h.index, h.values, color=COLORS[p], label=p)
        ax[0, 1].set_title("Monthly peak net import (MW)")
        ax[0, 2].set_title("Average day: net import (MW)")
        for p in PROFILES:
            ch, v2 = c.ev[p] if hasattr(c, "ev") else (None, None)
            if ch is not None:
                ax[1, 0].plot(ch.groupby(n.snapshots.hour).mean(), color=COLORS[p], label=f"{p} charge")
                ax[1, 0].plot(v2.groupby(n.snapshots.hour).mean(), "--", color=COLORS[p])
        ax[1, 0].set_title("EV average day: charge (solid), V2G (dashed), MW")
        ax[1, 0].legend(fontsize=7)
    if hasattr(c, "share_ts"):
        for p in PROFILES:
            ax[1, 1].plot(c.share_ts[p].rolling(24, min_periods=1).mean().values, color=COLORS[p], label=p, lw=.8)
        ax[1, 1].set_title("Each profile's share of total net import over time")
        ax[1, 1].legend()
    if hasattr(c, "lv_gwh"):
        t = c.lv_gwh.drop(index="TOTAL (should be ~0)")
        t.T.plot.bar(stacked=True, ax=ax[1, 2], width=.7, legend=False)
        ax[1, 2].set_title("LV balance by group (GWh, + supply / - withdrawal)")
        ax[1, 2].legend(fontsize=6, loc="lower left")
        ax[1, 2].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/split_deep_1.png", dpi=140)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    if hasattr(c, "heat_summ"):
        cols = ["heat pumps", "resistive", "gas boilers", "CHP"]
        s = c.heat_summ.groupby(level=0)[cols + ["household demand", "DAC heat"]].sum()
        s[cols].plot.bar(stacked=True, ax=ax[0], width=.6)
        ax[0].plot(range(len(s)), (s["household demand"] + s["DAC heat"]).values, "k_", ms=40, mew=2,
                   label="household + DAC heat")
        ax[0].set_title("Heat supply by technology vs demand (GWh), all profiles")
        ax[0].legend(fontsize=7)
        ax[0].tick_params(axis="x", rotation=15)
    if hasattr(c, "ac"):
        d = c.ac[c.ac < 0].abs().sort_values().tail(10)
        d.plot.barh(ax=ax[1])
        ax[1].set_title("Largest HV electricity consumers (GWh)")
        ax[1].tick_params(axis="y", labelsize=7)
    if hasattr(c, "prices"):
        mp, _ = c.prices
        for p in PROFILES:
            if c.lv[p]:
                v = pd.concat([mp[b] for b in c.lv[p]], axis=1).mean(axis=1).sort_values(ascending=False).values
                ax[2].plot(np.linspace(0, 100, len(v)), v, color=COLORS[p], label=f"{p} LV")
        ac = [b for b in n.buses.index if n.buses.at[b, "carrier"] == "AC" and b.startswith(c.prefix)]
        if ac:
            v = pd.concat([mp[b] for b in ac], axis=1).mean(axis=1).sort_values(ascending=False).values
            ax[2].plot(np.linspace(0, 100, len(v)), v, "k--", label="AC")
        ax[2].set_title("Price duration (EUR/MWh)")
        ax[2].legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/split_deep_2.png", dpi=140)



# ---------------------------------------------------------------- split vs unsplit
def heat_summary(h):
    """h: heat balance (rows 'kind | carrier', columns = anything). Returns one row per column."""
    def pick(pred):
        rows = [i for i in h.index if pred(i)]
        return h.loc[rows].sum() if rows else h.iloc[0] * 0
    out = pd.DataFrame({
        "household demand": -pick(lambda i: i.startswith("load")),
        "DAC heat": -pick(lambda i: i.endswith("| DAC")),
        "heat pumps": pick(lambda i: "heat pump" in i),
        "resistive": pick(lambda i: "resistive" in i),
        "gas boilers": pick(lambda i: "boiler" in i),
        "CHP": pick(lambda i: "CHP" in i),
        "storage net": pick(lambda i: "water" in i),
        "solar thermal": pick(lambda i: "solar thermal" in i),
        "vent": pick(lambda i: "vent" in i),
    })
    out["DAC / household %"] = 100 * out["DAC heat"] / out["household demand"].replace(0, np.nan)
    return out


def cost_cap(n, prefix):
    w = n.snapshot_weightings.generators
    keep = lambda idx: [i for i in idx if str(i).startswith(prefix)]  # noqa: E731
    cap, cost = {}, {}
    for comp, a in [("generators", "p_nom_opt"), ("links", "p_nom_opt"), ("stores", "e_nom_opt"),
                    ("storage_units", "p_nom_opt"), ("lines", "s_nom_opt")]:
        df = getattr(n, comp)
        df = df.loc[keep(df.index)]
        if df.empty:
            continue
        for car, v in df[a].replace([np.inf, -np.inf], 0).groupby(df.carrier).sum().items():
            cap[f"{comp} {car}"] = v
        capex = capex_of(comp, df, a)
        flow = {"generators": n.generators_t.p, "links": n.links_t.p0,
                "storage_units": n.storage_units_t.p_dispatch, "stores": n.stores_t.p}.get(comp)
        opex = pd.Series(0.0, index=df.index)
        if flow is not None:
            cols = [x for x in df.index if x in flow.columns]
            opex[cols] = flow[cols].mul(w, axis=0).mul(df.loc[cols, "marginal_cost"], axis=1).sum()
        for car, v in (capex + opex).groupby(df.carrier).sum().items():
            cost[car] = cost.get(car, 0) + v
    return pd.Series(cap, dtype=float), pd.Series(cost, dtype=float)


def collect(n, prefix):
    """System-level metrics, independent of how many LV buses/profiles exist."""
    c = Ctx()
    c.n, c.prefix, c.w = n, prefix, n.snapshot_weightings.generators
    c.dt = isinstance(n.snapshots, pd.DatetimeIndex)
    c.keep = lambda idx: [i for i in idx if str(i).startswith(prefix)]  # noqa: E731
    w = c.w
    dl = n.links[n.links.carrier == DIST]
    c.dl = dl.loc[c.keep(dl.index)]
    lv = [b for b in n.buses.index if n.buses.at[b, "carrier"] == "low voltage" and b.startswith(prefix)]
    M = {"c": c, "lv": lv}
    S = {}
    d = dist_for(c, lv)
    M["d"], M["net"] = d, d["net"]
    for k, v in stat_row(c, d).items():
        S[f"1 distribution | {k}"] = v
    S["1 distribution | LV buses"] = len(lv)
    S["0 total | objective EUR"] = n.objective
    net = d["net"]
    top = net.nlargest(max(1, int(0.02 * len(net)))).index
    M["top"] = top
    tot = None
    for b in lv:
        i = injections(n, b)
        tot = i if tot is None else tot.add(i, fill_value=0)
    grp = tot.T.groupby(lambda k: lv_group(*k)).sum().T
    M["grp"] = grp
    for k, v in (grp.mul(w, axis=0).sum() / 1e3).items():
        S[f"2 LV balance GWh | {k}"] = v
    M["peak_comp"] = pd.DataFrame({"at peak": grp.loc[net.idxmax()], "mean in top 2% import hours": grp.loc[top].mean(),
                                   "annual mean": grp.mean()})
    nodes = {}
    for ac, f in c.dl[c.dl.bus1.isin(lv)].groupby("bus0"):
        r = stat_row(c, dist_for(c, list(f.bus1.unique())))
        nodes[ac] = {k: r[k] for k in ["cap_in_MW(at LV)", "peak_net_import_MW", "load_factor_%", "h_>95%cap",
                                       "net_import_GWh", "gross_export_GWh"]}
    M["nodes"] = pd.DataFrame(nodes).T
    # heat
    h = heat_tables(c)
    hs = h.T.groupby(level=0).sum().T
    M["heat_sum"] = heat_summary(hs)
    for system, row in M["heat_sum"].iterrows():
        for k, v in row.items():
            S[f"3 heat GWh | {system} | {k}"] = v
    # DAC
    dac = n.links[n.links.carrier == "DAC"]
    dac = dac.loc[c.keep(dac.index)]
    if len(dac):
        cap = dac.p_nom_opt.sum()
        e = n.links_t.p0[dac.index].sum(axis=1)
        S["4 DAC | capacity MW"] = cap
        S["4 DAC | electricity from AC GWh"] = (e * w).sum() / 1e3
        S["4 DAC | full-load hours"] = (e * w).sum() / max(cap, 1e-9)
    # EV
    ch = n.links[n.links.carrier == "BEV charger"]
    v2 = n.links[n.links.carrier == "V2G"]
    ch, v2 = ch.loc[c.keep(ch.index)], v2.loc[c.keep(v2.index)]
    st = n.stores[n.stores.carrier == "EV battery"]
    st = st.loc[c.keep(st.index)]
    charge = n.links_t.p0[ch.index].sum(axis=1)
    v2g = -n.links_t.p1[v2.index].sum(axis=1)
    enom = st.e_nom_opt.sum()
    S["5 EV | charge GWh"] = (charge * w).sum() / 1e3
    S["5 EV | V2G delivered GWh"] = (v2g * w).sum() / 1e3
    S["5 EV | charger MW"] = ch.p_nom_opt.sum()
    S["5 EV | peak charge MW (coincident)"] = charge.max()
    S["5 EV | battery GWh"] = enom / 1e3
    S["5 EV | charge in top-2% import hours MW (mean)"] = charge[top].mean()
    S["5 EV | V2G in top-2% import hours MW (mean)"] = v2g[top].mean()
    M["charge"], M["v2g"] = charge, v2g
    # home battery
    hc = n.links[n.links.carrier == "home battery charger"]
    hd = n.links[n.links.carrier == "home battery discharger"]
    hc, hd = hc.loc[c.keep(hc.index)], hd.loc[c.keep(hd.index)]
    hst = n.stores[n.stores.carrier == "home battery"]
    hst = hst.loc[c.keep(hst.index)]
    dis = -n.links_t.p1[hd.index].sum(axis=1)
    he = hst.e_nom_opt.sum()
    S["6 home battery | energy GWh"] = he / 1e3
    S["6 home battery | discharger MW"] = hd.p_nom_opt.sum()
    S["6 home battery | discharge GWh"] = (dis * w).sum() / 1e3
    S["6 home battery | cycles per period"] = (dis * w).sum() / max(he, 1e-9)
    S["6 home battery | discharge in top-2% import hours MW (mean)"] = dis[top].mean()
    # PV
    g = n.generators[n.generators.carrier == "solar rooftop"]
    g = g.loc[c.keep(g.index)]
    if len(g):
        gen = n.generators_t.p[g.index].sum(axis=1)
        pot = pd.Series(0.0, index=n.snapshots)
        for nm in g.index:
            pm = n.generators_t.p_max_pu[nm] if nm in n.generators_t.p_max_pu.columns else g.at[nm, "p_max_pu"]
            pot = pot + pm * g.at[nm, "p_nom_opt"]
        S["7 rooftop PV | capacity MW"] = g.p_nom_opt.sum()
        S["7 rooftop PV | generation GWh"] = (gen * w).sum() / 1e3
        S["7 rooftop PV | curtailed GWh"] = ((pot - gen).clip(lower=0) * w).sum() / 1e3
        S["7 rooftop PV | net export GWh"] = (((-net).clip(lower=0)) * w).sum() / 1e3
    # HV bus
    ac = [b for b in n.buses.index if n.buses.at[b, "carrier"] == "AC" and b.startswith(prefix)]
    acb = balance_table(n, ac, w).sum(axis=1)
    acb = acb[acb.abs() > 0.001 * acb.abs().max()]
    M["ac"] = acb
    for k, v in acb.items():
        S[f"8 HV bus GWh | {k}"] = v
    # prices
    mp = n.buses_t.marginal_price
    if not mp.empty and lv:
        f = c.dl[c.dl.bus1.isin(lv)]
        lvp = pd.concat([mp[b] for b in lv], axis=1).mean(axis=1)
        acp = pd.concat([mp[b] for b in f.bus0.unique()], axis=1).mean(axis=1)
        imp = net.clip(lower=0)
        num = den = 0.0
        for nm, r in f.iterrows():
            mu = (r.efficiency * mp[r.bus1] - mp[r.bus0] - r.marginal_cost).clip(lower=0)
            num += (w * mu).sum() * r.p_nom_opt
            den += r.capital_cost * r.p_nom_opt
        S["9 prices | LV mean EUR/MWh"] = lvp.mean()
        S["9 prices | AC mean EUR/MWh"] = acp.mean()
        S["9 prices | LV p95"] = lvp.quantile(.95)
        S["9 prices | import-weighted LV-AC spread"] = ((lvp - acp) * imp).sum() / max(imp.sum(), 1e-9)
        S["9 prices | capacity rent / capital"] = num / max(den, 1e-9)
        M["lvp"], M["acp"] = lvp, acp
    cap_s, cost_s = cost_cap(n, prefix)
    M["cap"], M["cost"] = cap_s, cost_s
    M["S"] = pd.Series(S, dtype=float)
    return M


def compare(Ms, Mu):
    hdr("SPLIT vs UNSPLIT - headline metrics (system level; split profiles summed)")
    su, ss = Mu["S"], Ms["S"]
    keys = list(dict.fromkeys(list(su.index) + list(ss.index)))
    t = pd.DataFrame({"unsplit": su.reindex(keys), "split": ss.reindex(keys)})
    t["diff"] = t.split - t.unsplit
    t["diff_%"] = 100 * t["diff"] / t.unsplit.where(t.unsplit.abs() > 1e-9)
    t = t[(t.unsplit.abs().fillna(0) > 1e-6) | (t.split.abs().fillna(0) > 1e-6)]
    t.index = pd.MultiIndex.from_tuples([tuple(k.split(" | ", 1)) if " | " in k else (k, "") for k in t.index])
    for sec, sub in t.groupby(level=0, sort=True):
        print(f"\n[{sec[2:] if sec[:1].isdigit() else sec}]")
        print(sub.droplevel(0).round(2).to_string())
    save(t, "compare_headline")

    hdr("Peak composition (MW): unsplit vs split, at each network's own peak")
    print("unsplit peak:", Mu["net"].idxmax(), "   split peak:", Ms["net"].idxmax())
    pc = pd.concat({"unsplit": Mu["peak_comp"], "split": Ms["peak_comp"]}, axis=1)
    print(pc.round(1).to_string())
    save(pc, "compare_peak_composition")

    hdr("Per node (AC bus): unsplit vs split")
    nd = pd.concat({"unsplit": Mu["nodes"], "split": Ms["nodes"]}, axis=1)
    print(nd.round(1).to_string())
    save(nd, "compare_nodes")

    hdr("Net import time series: unsplit vs split (aggregate over all LV buses)")
    a, b = Mu["net"], Ms["net"]
    if a.index.equals(b.index):
        diff = b - a
        pk = max(a.abs().max(), 1e-9)
        st = {"correlation": a.corr(b), "mean abs diff MW": diff.abs().mean(), "RMS diff MW": float(np.sqrt((diff ** 2).mean())),
              "max abs diff MW": diff.abs().max(), "time of max diff": str(diff.abs().idxmax()),
              "hours with |diff| > 1% of peak": Mu["c"].w[diff.abs() > 0.01 * pk].sum(),
              "hours split higher by >1% of peak": Mu["c"].w[diff > 0.01 * pk].sum(),
              "hours split lower by >1% of peak": Mu["c"].w[diff < -0.01 * pk].sum()}
        print(pd.Series(st).to_string())
        q = [1.0, .99, .95, .9, .75, .5, .25]
        dc = pd.DataFrame({"unsplit": [a.quantile(x) for x in q], "split": [b.quantile(x) for x in q]},
                          index=[f"p{int(x * 100)}" for x in q])
        dc["diff_%"] = 100 * (dc.split - dc.unsplit) / dc.unsplit.where(dc.unsplit.abs() > 1e-9)
        print("\nnet import distribution (MW):")
        print(dc.round(1).to_string())
        Mu["diff"] = diff
    else:
        print("snapshots differ between the two networks - time-series comparison skipped")

    hdr("Installed capacity by carrier: largest changes")
    cap = pd.concat([Mu["cap"], Ms["cap"]], axis=1, keys=["unsplit", "split"]).fillna(0)
    cap["diff"] = cap.split - cap.unsplit
    cap["diff_%"] = 100 * cap["diff"] / cap.unsplit.where(cap.unsplit.abs() > 1e-9)
    cap = cap[(cap.abs().max(axis=1) > 1e-3)]
    print(cap.reindex(cap["diff"].abs().sort_values(ascending=False).index).head(20).round(2).to_string())
    save(cap, "compare_capacity")

    hdr("Cost by carrier (EUR): largest changes")
    cs = pd.concat([Mu["cost"], Ms["cost"]], axis=1, keys=["unsplit", "split"]).fillna(0)
    cs["diff"] = cs.split - cs.unsplit
    cs["diff_%"] = 100 * cs["diff"] / cs.unsplit.where(cs.unsplit.abs() > 1e-9)
    print(cs.reindex(cs["diff"].abs().sort_values(ascending=False).index).head(15).round(1).to_string())
    print(f"\nobjective: unsplit {Mu['S']['0 total | objective EUR']:,.0f}   split {Ms['S']['0 total | objective EUR']:,.0f}"
          f"   diff {Ms['S']['0 total | objective EUR'] - Mu['S']['0 total | objective EUR']:,.0f}")
    save(cs, "compare_cost")
    Mu["cs"] = cs


def compare_figure(Ms, Mu):
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    for M, lab, col in [(Mu, "unsplit", "k"), (Ms, "split", "#D85A30")]:
        s = M["net"].clip(lower=0).sort_values(ascending=False).values
        ax[0, 0].plot(np.linspace(0, 100, len(s)), s, color=col, label=lab)
        if M["c"].dt:
            h = M["net"].groupby(M["net"].index.hour).mean()
            ax[0, 2].plot(h.index, h.values, color=col, label=lab)
    ax[0, 0].set_title("Aggregate net import duration curve (MW)")
    ax[0, 0].legend()
    ax[0, 2].set_title("Average day: net import (MW)")
    ax[0, 2].legend()
    if "diff" in Mu:
        ax[0, 1].plot(Mu["diff"].values, lw=.6)
        ax[0, 1].axhline(0, c="k", lw=.5)
        ax[0, 1].set_title("Net import: split minus unsplit (MW)")
    pc = pd.concat({"unsplit": Mu["peak_comp"]["mean in top 2% import hours"],
                    "split": Ms["peak_comp"]["mean in top 2% import hours"]}, axis=1)
    pc.drop(index=[i for i in pc.index if i.startswith("distribution")], errors="ignore").plot.barh(ax=ax[1, 0])
    ax[1, 0].set_title("Mean composition in top-2% import hours (MW)")
    hu, hs = Mu["heat_sum"], Ms["heat_sum"]
    cols = ["heat pumps", "resistive", "gas boilers", "CHP"]
    pd.concat({"unsplit": hu[cols].sum(), "split": hs[cols].sum()}, axis=1).plot.bar(ax=ax[1, 1])
    ax[1, 1].set_title("Heat supply by technology (GWh)")
    ax[1, 1].tick_params(axis="x", rotation=0)
    cs = Mu["cs"]
    cs["diff"].reindex(cs["diff"].abs().sort_values().tail(12).index).plot.barh(ax=ax[1, 2])
    ax[1, 2].set_title("Cost difference by carrier, split - unsplit (EUR)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/split_vs_unsplit_deep.png", dpi=140)


# ---------------------------------------------------------------- main
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    ncs = [a for a in args if a.endswith(".nc")]
    rest = [a for a in args if not a.endswith(".nc")]
    if not ncs:
        sys.exit(__doc__)
    prefix = rest[0] if rest else ""
    os.makedirs(OUT, exist_ok=True)
    n = pypsa.Network(ncs[0])
    if "--only-compare" not in flags or len(ncs) < 2:
        c = build_ctx(n, prefix)
        for fn in [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14]:
            try:
                fn(c)
            except Exception as e:          # noqa: BLE001
                print(f"\n[{fn.__name__}] FAILED: {type(e).__name__}: {e}")
                traceback.print_exc(limit=2)
        try:
            figures(c)
        except Exception as e:              # noqa: BLE001
            print("figures failed:", type(e).__name__, e)
    if len(ncs) > 1:
        try:
            nu = pypsa.Network(ncs[1])
            Ms, Mu = collect(n, prefix), collect(nu, prefix)
            compare(Ms, Mu)
            compare_figure(Ms, Mu)
        except Exception as e:              # noqa: BLE001
            print(f"\n[compare] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
    print(f"\nCSVs in ./{OUT}/, figures split_deep_1.png, split_deep_2.png, split_vs_unsplit_deep.png")


if __name__ == "__main__":
    main()