"""Compare solved PyPSA-Eur networks: unsplit baseline vs split runs (v2).

Usage: edit RUNS below (first entry = baseline), then
    python inspect_split.py
Prints per-run tables and a comparison, writes compare_fresh.csv.
"""
import re

import numpy as np
import pandas as pd
import pypsa

NET = "networks\\base_s_3_lv1.2_Co2L0-3H_2030.nc"
ROOT = r"C:\Users\nasos\DTU\pypsa-eur\results" + "\\"
RUNS = {
    "baseline": ROOT + r"G4NoSplit_GR_summer" + "\\" + NET,
    "split": ROOT + r"G4Split_GR_summer" + "\\" + NET,
    # neutrality tests: set the result folder names of your new runs
    # 1) single profile: all four share dicts = urban 1.0, suburban 0.0, rural 0.0
    "split even": ROOT + r"G4Split_GR_summer_even shares" + "\\" + NET,
    # 2) identical vector in all four dicts (0.4 / 0.3 / 0.3), pin_profile=None
   # "neutral_403030": ROOT + r"G4Neutral2_GR_summer" + "\\" + NET,
}
 

DIST = "electricity distribution grid"
ELEC_CARRIERS = {"AC", "low voltage"}
GROUP_TOKENS = {"a", "b", "urban", "suburban", "rural"}
STRIP = {"home", "battery", "charger", "discharger", "reversed"}
 
 
def group(name):
    """Profile/half label from the END of a component name; '-' if none."""
    toks = re.sub(r"-reversed$", "", str(name)).split()
    while toks and toks[-1] in STRIP:
        toks.pop()
    return toks[-1] if toks and toks[-1] in GROUP_TOKENS else "-"
 
 
def ports(links):
    return [c for c in links.columns if c.startswith("bus") and c[3:].isdigit()]
 
 
def elec_flow(n, links):
    """Electric-side flow per link (absolute, MW), found via the bus carrier."""
    out, port_used = {}, {}
    for name in links.index:
        for k in (c[3:] for c in ports(n.links)):
            b = n.links.at[name, f"bus{k}"]
            if isinstance(b, str) and b in n.buses.index and n.buses.at[b, "carrier"] in ELEC_CARRIERS:
                out[name] = n.links_t[f"p{k}"][name].abs()
                port_used[name] = int(k)
                break
    return pd.DataFrame(out), pd.Series(port_used, dtype="float")
 
 
def forward_dist(n):
    d = n.links[n.links.carrier == DIST]
    # the profile suffix comes AFTER "-reversed" ("...grid-reversed urban"), so use contains
    return d[~d.index.str.contains("-reversed")]
 
 
def grid_table(n):
    w = n.snapshot_weightings.generators
    rows = []
    for name, l in forward_dist(n).iterrows():
        p = n.links_t.p0[name]
        cap = l.p_nom_opt
        rows.append(dict(link=name, group=group(name), cap_MW=cap, peak_MW=p.max(),
                         peak_at=str(p.idxmax()),
                         pct_time_at_limit=100 * w[p >= 0.99 * cap].sum() / w.sum() if cap > 0 else np.nan,
                         extendable=bool(l.p_nom_extendable)))
    return pd.DataFrame(rows)
 
 
def cap_by_group(n):
    rows = []
    specs = [("Link", "heat pump"), ("Link", "resistive heater"), ("Link", "BEV charger"),
             ("Link", "V2G"), ("Link", "home battery charger"),
             ("Store", "home battery"), ("Store", "water tank|water pit"),
             ("Generator", "solar rooftop")]
    for comp, pat in specs:
        df = getattr(n, comp.lower() + "s")
        sel = df[df.carrier.str.contains(pat, na=False)]
        col = "e_nom_opt" if comp == "Store" else "p_nom_opt"
        g = sel[col].groupby(sel.index.map(group)).sum()
        for k, v in g.items():
            rows.append(dict(tech=pat, group=k, opt_cap=v))
    return pd.DataFrame(rows).pivot_table(index="tech", columns="group", values="opt_cap", aggfunc="sum")
 
 
def pv_battery_table(n):
    """Per profile: rooftop PV potential vs built, and home battery energy per MW of PV.
 
    Potential must sum to the same total in every run (the split only redistributes it).
    Battery MWh per PV MW far above the baseline ratio shows batteries placed by cost
    alone (nothing ties them to PV or demand)."""
    gen = n.generators[n.generators.carrier.str.contains("solar rooftop", na=False)]
    sto = n.stores[n.stores.carrier.str.contains("home battery", na=False)]
    pot = gen.p_nom_max.where(np.isfinite(gen.p_nom_max), np.nan)
    t = pd.DataFrame({
        "pv_potential_MW": pot.groupby(gen.index.map(group)).sum(min_count=1),
        "pv_built_MW": gen.p_nom_opt.groupby(gen.index.map(group)).sum(),
        "battery_MWh": sto.e_nom_opt.groupby(sto.index.map(group)).sum(),
    }).fillna(0.0)
    t.loc["TOTAL"] = t.sum()
    t["pv_used_%"] = 100 * t.pv_built_MW / t.pv_potential_MW.replace(0, np.nan)
    t["potential_share_%"] = 100 * t.pv_potential_MW / t.loc["TOTAL", "pv_potential_MW"]
    t["battery_share_%"] = 100 * t.battery_MWh / t.loc["TOTAL", "battery_MWh"]
    t["battery_MWh_per_PV_MW"] = t.battery_MWh / t.pv_built_MW.replace(0, np.nan)
    return t.round(3)
 
 
def line_table(n):
    """Per AC line: sizes, bounds and cost, so cost differences can be traced to one line."""
    ln = n.lines
    if ln.empty:
        return pd.DataFrame()
    cols = [c for c in ["bus0", "bus1", "length", "s_nom", "s_nom_extendable", "s_nom_min",
                        "s_nom_max", "s_nom_opt", "capital_cost"] if c in ln.columns]
    t = ln[cols].copy()
    t["cost_EUR"] = t.capital_cost * t.s_nom_opt
    t["cost_per_MVA"] = t.capital_cost
    return t.round(3)
 
 
def bus_label(n, b):
    """'<heat bus carrier> [<profile>]' for heat buses, None otherwise."""
    if not isinstance(b, str) or b not in n.buses.index:
        return None
    c = str(n.buses.at[b, "carrier"])
    return f"{c} [{group(b)}]" if "heat" in c else None
 
 
def lv_peak_decomp(n):
    """MW injected into all LV buses by component carrier at the system import peak (negative = drawn)."""
    f = forward_dist(n)
    t = n.links_t.p0[f.index].sum(axis=1).idxmax()
    lv = set(n.buses.index[n.buses.carrier == "low voltage"])
    rows = {}
    for k in (c[3:] for c in ports(n.links)):
        try:
            pk = n.links_t[f"p{k}"]
        except KeyError:
            continue
        if pk.empty:
            continue
        sel = n.links.loc[pk.columns, f"bus{k}"].isin(lv).values
        for l in pk.columns[sel]:
            key = "Link: " + str(n.links.at[l, "carrier"])
            rows[key] = rows.get(key, 0.0) - float(pk.at[t, l])
    lo = n.loads_t.p
    if not lo.empty:
        for l in lo.columns[n.loads.loc[lo.columns, "bus"].isin(lv).values]:
            key = "Load: " + str(n.loads.at[l, "carrier"])
            rows[key] = rows.get(key, 0.0) - float(lo.at[t, l])
    g = n.generators_t.p
    if not g.empty:
        for l in g.columns[n.generators.loc[g.columns, "bus"].isin(lv).values]:
            key = "Gen: " + str(n.generators.at[l, "carrier"])
            rows[key] = rows.get(key, 0.0) + float(g.at[t, l])
    s = pd.Series(rows).round(1)
    s = s[s.abs() > 0.05]
    s.loc["NET (should be ~0)"] = round(float(s.sum()), 2)
    return t, s.sort_values()
 
 
def heat_balance(n):
    """Net GWh delivered INTO each heat bus carrier by component carrier (negative = consumed)."""
    w = n.snapshot_weightings.generators
    parts = []
    for k in (c[3:] for c in ports(n.links)):
        try:
            pk = n.links_t[f"p{k}"]
        except KeyError:
            continue
        if pk.empty:
            continue
        e = -(pk.mul(w, axis=0).sum()) / 1e3
        parts.append(pd.DataFrame({
            "bus": n.links.loc[e.index, f"bus{k}"].map(lambda b: bus_label(n, b)),
            "comp": "Link: " + n.links.loc[e.index, "carrier"].astype(str),
            "GWh": e}))
    g = n.generators_t.p
    if not g.empty:
        e = g.mul(w, axis=0).sum() / 1e3
        parts.append(pd.DataFrame({
            "bus": n.generators.loc[e.index, "bus"].map(lambda b: bus_label(n, b)),
            "comp": "Gen: " + n.generators.loc[e.index, "carrier"].astype(str),
            "GWh": e}))
    lo = n.loads_t.p
    if not lo.empty:
        e = -(lo.mul(w, axis=0).sum()) / 1e3
        parts.append(pd.DataFrame({
            "bus": n.loads.loc[e.index, "bus"].map(lambda b: bus_label(n, b)),
            "comp": "Load: " + n.loads.loc[e.index, "carrier"].astype(str),
            "GWh": e}))
    df = pd.concat(parts)
    df = df[df.bus.astype(str).str.contains("heat", na=False)]
    return df.pivot_table(index="comp", columns="bus", values="GWh", aggfunc="sum").fillna(0)
 
 
def summary(n):
    r = {}
    w = n.snapshot_weightings.generators
    r["snapshots"] = len(n.snapshots)
    r["hours_sum_gen"] = float(w.sum())
    r["hours_sum_obj"] = float(n.snapshot_weightings.objective.sum())
    r["objective"] = float(n.objective)
    f = forward_dist(n)
    r["dist_cap_MW (fwd only)"] = float(f.p_nom_opt.sum())
    r["dist_capital_cost_EURperMW"] = float(f.capital_cost.mean())
    r["dist_n_fwd"] = len(f)
    r["dist_extendable_n"] = int(f.p_nom_extendable.sum())
    r["peak_dist_flow_MW"] = float(n.links_t.p0[f.index].sum(axis=1).max())
    # binding-time metrics: "any feeder at limit" rises mechanically with the NUMBER of feeders,
    # so also report the per-feeder mean and the capacity-weighted mean (comparable across splits)
    gt = grid_table(n)
    r["bind_pct_mean_feeder"] = float(gt.pct_time_at_limit.mean())
    r["bind_pct_cap_weighted"] = float((gt.pct_time_at_limit * gt.cap_MW).sum() / gt.cap_MW.sum())
    at_lim = pd.DataFrame({k: n.links_t.p0[k] >= 0.99 * f.at[k, "p_nom_opt"] for k in f.index
                           if f.at[k, "p_nom_opt"] > 0})
    r["bind_pct_any_feeder"] = float(100 * w[at_lim.any(axis=1)].sum() / w.sum())
    ln = n.lines
    r["line_s_nom_opt_sum_MVA"] = float(ln.s_nom_opt.sum()) if len(ln) else 0.0
    r["line_capex_ext_EUR"] = float((ln.capital_cost * ln.s_nom_opt)[ln.s_nom_extendable].sum()) if len(ln) else 0.0
    try:
        cx = n.statistics.capex().groupby(level=1).sum()
        ox = n.statistics.opex().groupby(level=1).sum()
        r["capex_total"], r["opex_total"] = float(cx.sum()), float(ox.sum())
        r["capex_dist"] = float(cx.get(DIST, 0))
    except Exception as e:
        print("  statistics failed:", e)
 
    hp = n.links[n.links.carrier.str.contains("heat pump", na=False)]
    flow, used = elec_flow(n, hp)
    r["hp_n_links"] = len(hp)
    r["hp_P_MW"] = float(hp.p_nom_opt.sum())
    r["hp_elec_port_counts"] = str(used.value_counts().to_dict())
    if len(flow):
        e = flow.sum(axis=1)
        r["hp_elec_GWh"] = float((e * w).sum() / 1e3)
        r["hp_peak_MW"] = float(e.max())
        r["hp_peak_to_mean"] = float(e.max() / e.mean()) if e.mean() else np.nan
        heat = -n.links_t.p0[hp.index].sum(axis=1)  # bus0 is the heat side in this build
        r["hp_heat_GWh"] = float((heat * w).sum() / 1e3)
        r["hp_avg_COP"] = r["hp_heat_GWh"] / r["hp_elec_GWh"]
    rh = n.links[n.links.carrier.str.contains("resistive heater", na=False)]
    r["rh_P_MW"] = float(rh.p_nom_opt.sum())
    rflow, _ = elec_flow(n, rh)
    if len(rflow):
        r["rh_elec_GWh"] = float((rflow.sum(axis=1) * w).sum() / 1e3)
    tk = n.stores[n.stores.carrier.str.contains("water tank|water pit", na=False)]
    r["tank_E_MWh"] = float(tk.e_nom_opt.sum())
    r["home_batt_E_MWh"] = float(n.stores[n.stores.carrier == "home battery"].e_nom_opt.sum())
    ev = n.links[n.links.carrier == "BEV charger"]
    r["ev_charger_P_MW"] = float(ev.p_nom.sum())
    if len(ev):
        c = n.links_t.p0[ev.index].sum(axis=1)
        r["ev_charge_GWh"] = float((c * w).sum() / 1e3)
        r["ev_charge_peak_MW"] = float(c.max())
    v = n.links[n.links.carrier == "V2G"]
    if len(v):
        r["ev_v2g_GWh"] = float((n.links_t.p0[v.index].sum(axis=1).abs() * w).sum() / 1e3)
    pv = n.generators[n.generators.carrier == "solar rooftop"]
    r["pv_rooftop_MW"] = float(pv.p_nom_opt.sum())
    r["pv_rooftop_potential_MW"] = float(pv.p_nom_max.sum())
    # demand totals per load carrier: must be conserved by any split
    for car, g in n.loads.groupby("carrier"):
        cols = [c for c in g.index if c in n.loads_t.p_set.columns]
        e = n.loads_t.p_set[cols].mul(w, axis=0).sum().sum() if cols else 0.0
        e += g.loc[~g.index.isin(cols), "p_set"].sum() * w.sum()
        r[f"demand_{car}_GWh"] = float(e / 1e3)
    return r
 
 
if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_rows", 200, "display.max_columns", 30,
                  "display.float_format", "{:,.4g}".format)
    res = {}
    capex_by_run = {}
    for name, path in RUNS.items():
        print(f"\n===== {name}: {path}")
        n = pypsa.Network(path)
        res[name] = summary(n)
        nyears = res[name]["hours_sum_gen"] / 8760
        pv_cc = n.generators.loc[n.generators.carrier == "solar rooftop", "capital_cost"].mean()
        gd_cc = res[name]["dist_capital_cost_EURperMW"]
        print(f"  weights: {res[name]['hours_sum_gen']:.0f} h (nyears={nyears:.4f}) | capital_cost per modelled period: "
              f"rooftop PV {pv_cc:,.0f}, dist grid {gd_cc:,.0f} EUR/MW -> annual equivalent "
              f"{pv_cc / nyears:,.0f} / {gd_cc / nyears:,.0f} EUR/MW/yr")
        print("\n  -- forward distribution links")
        print(grid_table(n).to_string(index=False))
        print("\n  -- AC lines (compare across runs: same MVA total, different cost?)")
        print(line_table(n).to_string())
        try:
            print(f"  objective={n.objective:,.2f}  objective_constant={getattr(n, 'objective_constant', float('nan')):,.2f}")
            gc = n.global_constraints
            if len(gc):
                print(gc[[c for c in ["type", "sense", "constant", "mu"] if c in gc.columns]].to_string())
        except Exception as e:
            print("  objective/global constraints failed:", e)
        print("\n  -- optimal capacity by profile/half (MW or MWh)")
        cg = cap_by_group(n)
        print(cg)
        if cg.shape[1] > 1:
            print("\n  -- same, as % of the total across profiles")
            print((cg.div(cg.sum(axis=1), axis=0) * 100).round(1))
        print("\n  -- rooftop PV potential vs built, and home battery per MW of PV")
        print(pv_battery_table(n).to_string())
        print("\n  -- heat balance: net GWh into each heat bus carrier (negative = consumed)")
        hb = heat_balance(n)
        net = hb.sum()                       # NET from UNROUNDED values
        print(hb.round(2))
        print("  NET per heat bus, unrounded (should be ~0):")
        print(net.apply(lambda v: f"{v:.2e}").to_string())
        t_pk, lvp = lv_peak_decomp(n)
        print(f"\n  -- LV balance at the system import peak ({t_pk}): MW into LV buses (negative = drawn)")
        print(lvp)
        try:
            capex_by_run[name] = n.statistics.capex().groupby(level=1).sum()
        except Exception as e:
            print("  capex by carrier failed:", e)
        hp = n.links[n.links.carrier.str.contains("heat pump", na=False)]
        if len(hp):
            first = hp.index[0]
            print(f"\n  -- ports of first heat pump: {first}")
            for k in (c[3:] for c in ports(n.links)):
                b = n.links.at[first, f"bus{k}"]
                if isinstance(b, str) and b:
                    eff = n.links.at[first, "efficiency" if k == "1" else f"efficiency{k}"] if k != "0" else 1.0
                    print(f"     bus{k}: {b:45s} carrier={n.buses.at[b, 'carrier']:22s} eff={eff}")
            print("     p_min_pu:", n.links.at[first, "p_min_pu"], " names:", list(hp.index[:5]))
 
    df = pd.DataFrame(res)
    num = df.apply(pd.to_numeric, errors="coerce")
    base = num.iloc[:, 0]
    pct = num.sub(base, axis=0).div(base.replace(0, np.nan), axis=0) * 100
    print("\n===== COMPARISON (values)\n", df)
    print("\n===== COMPARISON (% vs baseline)\n", pct.iloc[:, 1:])
 
    # ---- neutrality verdict: a correctly wired neutral split must match the baseline
    EXACT = [i for i in num.index if i.startswith("demand_")] + [
        "pv_rooftop_potential_MW", "ev_charger_P_MW"]
    TOL = {"objective": 1e-4}  # 0.01 %
    LOOSE = ["dist_cap_MW (fwd only)", "peak_dist_flow_MW", "hp_elec_GWh", "rh_elec_GWh",
             "tank_E_MWh", "home_batt_E_MWh", "pv_rooftop_MW", "ev_charge_GWh", "ev_v2g_GWh"]
    print("\n===== NEUTRALITY VERDICT vs baseline")
    print("      conserved totals: |rel diff| < 1e-6 | objective: < 0.01 % | capacities/flows: < 0.5 %")
    for run in num.columns[1:]:
        if run == "split":
            continue  # the real-share run is expected to differ
        print(f"\n  {run}")
        fails = 0
        for k, tol in [(k, 1e-6) for k in EXACT] + [("objective", 1e-4)] + [(k, 5e-3) for k in LOOSE]:
            b, v = base.get(k, np.nan), num.at[k, run] if k in num.index else np.nan
            if pd.isna(b) or pd.isna(v):
                print(f"     MISSING  {k}")
                fails += 1
                continue
            rel = abs(v - b) / abs(b) if b else abs(v - b)
            ok = rel < tol
            fails += not ok
            print(f"     {'PASS' if ok else 'FAIL'}  {k:34s} base={b:,.5g}  run={v:,.5g}  rel={rel:.2e}")
        print(f"     -> {'NEUTRAL: wiring sound' if fails == 0 else f'{fails} check(s) failed: wiring or pinning effect'}")
    if capex_by_run:
        cdf = pd.DataFrame(capex_by_run).fillna(0.0)
        for run in cdf.columns[1:]:
            d = cdf[run] - cdf.iloc[:, 0]
            print(f"\n===== annual capex difference by carrier: {run} - baseline (EUR, top 12 by |diff|)")
            print(d.reindex(d.abs().sort_values(ascending=False).index).head(12).round(0))
    pd.concat({"value": df, "pct_vs_baseline": pct}, axis=1).to_csv("compare_fresh.csv")
    print("\nwritten compare_fresh.csv")
