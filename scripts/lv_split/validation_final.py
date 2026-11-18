"""
Validate SOLVED networks of the LV-split model: rooftop solar, home batteries,
distribution-grid flows and storage operation, per node and per LV half (a / b).

Usage
-----
# symmetric split vs unsplit (or vs another symmetric run): totals should match
python validate_solved.py split.nc --reference unsplit.nc --expect symmetric

# asymmetric run: halves should DIFFER; optionally compare to a symmetric reference
python validate_solved.py asym.nc --reference sym.nc --expect asymmetric

Exit code 1 if any hard check fails.
Tolerance (--tol, default 2 %) applies to relative differences.
"""
import argparse
import sys

import numpy as np
import pandas as pd
import pypsa

ROOFTOP = "solar rooftop"
DIST = "electricity distribution grid"
BATT = "home battery"


# ----------------------------------------------------------------- helpers
def _key(n, bus):
    """(node, half) of a bus. half is 'a'/'b' for split buses, '-' otherwise."""
    loc = n.buses.at[bus, "location"] if "location" in n.buses else None
    node = loc if isinstance(loc, str) and loc else bus
    half = bus[-1] if bus.endswith((" a", " b")) else "-"
    return node, half


def _w(n):
    sw = n.snapshot_weightings
    return sw["generators"] if "generators" in sw else sw.iloc[:, 0]


def _add(rows, key, col, val):
    rows.setdefault(key, {})
    rows[key][col] = rows[key].get(col, 0.0) + float(val)


def _dist_groups(n):
    """Per LV bus: forward links (AC->LV, bus1 = LV) and '-reversed' links (LV->AC, bus0 = LV)."""
    lvb = set(n.buses.index[n.buses.carrier == "low voltage"])
    groups = {}
    for name, r in n.links[n.links.carrier == DIST].iterrows():
        if r.bus1 in lvb:
            groups.setdefault(r.bus1, {"fwd": [], "rev": []})["fwd"].append(name)
        elif r.bus0 in lvb:
            groups.setdefault(r.bus0, {"fwd": [], "rev": []})["rev"].append(name)
    return groups


def _dist_flow(n, g):
    """Net import into the LV bus (+ = AC->LV, - = LV->AC) summing forward and reversed links."""
    p0 = n.links_t.p0
    return p0.reindex(columns=g["fwd"]).sum(axis=1) - p0.reindex(columns=g["rev"]).sum(axis=1)


def summarise(n):
    """DataFrame indexed by (node, half) with the quantities we validate."""
    w = _w(n)
    rows = {}

    # rooftop solar
    g = n.generators[n.generators.carrier == ROOFTOP]
    for name, r in g.iterrows():
        k = _key(n, r.bus)
        _add(rows, k, "rooftop_p_nom_opt_MW", r.p_nom_opt)
        _add(rows, k, "rooftop_p_nom_max_MW", r.p_nom_max)
        if name in n.generators_t.p.columns:
            _add(rows, k, "rooftop_gen_GWh", (n.generators_t.p[name] * w).sum() / 1e3)

    # home battery stores
    st = n.stores[n.stores.carrier.str.contains(BATT, na=False)]
    for name, r in st.iterrows():
        k = _key(n, r.bus)
        _add(rows, k, "batt_e_nom_opt_MWh", r.e_nom_opt)
        if name in n.stores_t.p.columns:
            p = n.stores_t.p[name]
            dis = (p.clip(lower=0) * w).sum() / 1e3
            chg = (-p.clip(upper=0) * w).sum() / 1e3
            _add(rows, k, "batt_discharged_GWh", dis)
            _add(rows, k, "batt_charged_GWh", chg)
            _add(rows, k, "batt_soc_peak_MWh", n.stores_t.e[name].max())

    # battery chargers (power capacity), keyed by LV bus (bus0)
    ch = n.links[n.links.carrier.str.contains(BATT + " charger", na=False)]
    for name, r in ch.iterrows():
        _add(rows, _key(n, r.bus0), "batt_charger_MW", r.p_nom_opt)

    # LV electricity demand (context: confirms demand scaling took effect)
    lv = n.buses.index[n.buses.carrier == "low voltage"]
    ld = n.loads[n.loads.bus.isin(lv) & n.loads.carrier.str.contains("electric", na=False)]
    for name, r in ld.iterrows():
        ts = n.loads_t.p_set[name] if name in n.loads_t.p_set.columns else pd.Series(r.p_set, index=n.snapshots)
        _add(rows, _key(n, r.bus), "lv_demand_GWh", (ts * w).sum() / 1e3)

    # distribution grid: forward + reversed links of each LV bus combined
    for bus, g in _dist_groups(n).items():
        k = _key(n, bus)
        cap_imp = n.links.loc[g["fwd"], "p_nom_opt"].sum()
        cap_exp = n.links.loc[g["rev"], "p_nom_opt"].sum() if g["rev"] else cap_imp
        flow = _dist_flow(n, g)  # + = import AC -> LV
        imp, exp = flow.clip(lower=0), (-flow).clip(lower=0)
        _add(rows, k, "dist_cap_MW", cap_imp)
        _add(rows, k, "dist_export_cap_MW", cap_exp)
        _add(rows, k, "dist_import_GWh", (imp * w).sum() / 1e3)
        _add(rows, k, "dist_export_GWh", (exp * w).sum() / 1e3)
        _add(rows, k, "dist_net_import_GWh", (flow * w).sum() / 1e3)
        _add(rows, k, "dist_peak_import_MW", imp.max())
        _add(rows, k, "dist_peak_export_MW", exp.max())
        at_lim = ((imp >= 0.999 * cap_imp) & (cap_imp > 0)) | ((exp >= 0.999 * cap_exp) & (cap_exp > 0))
        _add(rows, k, "dist_pct_time_at_limit", 100 * (w * at_lim).sum() / w.sum())
        reinf = [l for l in g["fwd"] if "reinforcement" in l]
        _add(rows, k, "dist_reinforcement_MW", n.links.loc[reinf, "p_nom_opt"].sum() if reinf else 0.0)

    df = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)
    df.index = pd.MultiIndex.from_tuples(df.index, names=["node", "half"])
    return df.sort_index()


SUM_COLS = ["rooftop_p_nom_opt_MW", "rooftop_gen_GWh", "batt_e_nom_opt_MWh", "batt_charger_MW",
            "batt_discharged_GWh", "batt_charged_GWh", "lv_demand_GWh",
            "dist_net_import_GWh", "dist_import_GWh", "dist_export_GWh"]
# gross import/export are NOT unique when the link is lossless/bidirectional (free cycling),
# so they are only warnings; net import is the robust comparison.
SOFT_COLS = {"dist_import_GWh", "dist_export_GWh", "batt_discharged_GWh", "batt_charged_GWh"}


def node_totals(df):
    return df.groupby(level="node")[[c for c in SUM_COLS if c in df.columns]].sum()


def _rel(a, b, floor):
    return (a - b).abs() / np.maximum(b.abs(), floor)


# ----------------------------------------------------------------- checks
def run_checks(test, ref, expect, tol, solar=None, demand=None, headroom=None):
    res = []

    def add(name, ok, detail="", hard=True):
        res.append((name, bool(ok), detail, hard))

    n_t = test
    df = summarise(n_t)
    if df.empty:
        add("found rooftop / battery / distribution elements", False,
            "nothing matched: check carrier names " + str(sorted(n_t.links.carrier.unique())[:15]))
        return df, res

    carriers = {"rooftop gens": (n_t.generators.carrier == ROOFTOP).sum(),
                "home battery stores": n_t.stores.carrier.str.contains(BATT, na=False).sum(),
                "distribution links": (n_t.links.carrier == DIST).sum()}
    add("elements found", all(v > 0 for v in carriers.values()), str(carriers))

    # always-valid sanity checks
    if "rooftop_p_nom_max_MW" in df:
        fin = df.rooftop_p_nom_max_MW.replace(np.inf, np.nan).notna()
        viol = df[fin & (df.rooftop_p_nom_opt_MW > df.rooftop_p_nom_max_MW * (1 + 1e-6) + 1e-6)]
        add("rooftop p_nom_opt <= p_nom_max on every half", viol.empty, str(list(viol.index[:5])))
    if "dist_cap_MW" in df:
        bad = []
        for bus, g in _dist_groups(n_t).items():
            flow = _dist_flow(n_t, g)
            cap_imp = n_t.links.loc[g["fwd"], "p_nom_opt"].sum()
            cap_exp = n_t.links.loc[g["rev"], "p_nom_opt"].sum() if g["rev"] else cap_imp
            if flow.max() > cap_imp * (1 + 1e-6) + 1e-6 or (-flow).max() > cap_exp * (1 + 1e-6) + 1e-6:
                bad.append(bus)
        add("distribution flow <= link capacity (import and export) on every half", not bad, str(bad[:5]))
        # a reversed link that exists but has zero capacity would silently block export
        zero_rev = [bus for bus, g in _dist_groups(n_t).items()
                    if g["rev"] and n_t.links.loc[g["rev"], "p_nom_opt"].sum() <= 0
                    and n_t.links.loc[g["fwd"], "p_nom_opt"].sum() > 0]
        add("no half with import capacity but ZERO export capacity (reversed link)", not zero_rev,
            str(zero_rev[:5]) + " -> export blocked; check constrain_lv_links handles '-reversed' links")

    # export capacity must equal import capacity on every half (forward / reversed pair)
    mism = []
    for bus, g in _dist_groups(n_t).items():
        if g["rev"]:
            ci = n_t.links.loc[g["fwd"], "p_nom_opt"].sum()
            ce = n_t.links.loc[g["rev"], "p_nom_opt"].sum()
            if abs(ci - ce) > 1e-6 * max(ci, 1) + 1e-6:
                mism.append((bus, round(ci, 3), round(ce, 3)))
    add("export capacity == import capacity on every half", not mism, str(mism[:3]))

    # inputs of the run must match the configured scalings (catches parameters that never arrive)
    if demand is not None or solar is not None or headroom is not None:
        dfa = df.xs("a", level="half") if "a" in df.index.get_level_values("half") else None
        dfb = df.xs("b", level="half") if "b" in df.index.get_level_values("half") else None
        if dfa is None or dfb is None:
            add("scalings given but network has no a/b halves", False)
        else:
            idx = dfa.index.intersection(dfb.index)
            if demand is not None:
                exp = demand[0] / sum(demand)
                sh = dfa.loc[idx, "lv_demand_GWh"] / (dfa.loc[idx, "lv_demand_GWh"] + dfb.loc[idx, "lv_demand_GWh"])
                bad = sh[(sh - exp).abs() > 1e-4]
                add(f"LV demand share of 'a' == {exp:.3f} (demand_scaling {tuple(demand)})", bad.empty,
                    "; ".join(f"{i}: {v:.4f}" for i, v in bad.items()))
            if solar is not None:
                exp = solar[0] / solar[1]
                pa = dfa.loc[idx, "rooftop_p_nom_max_MW"].replace(np.inf, np.nan)
                pb = dfb.loc[idx, "rooftop_p_nom_max_MW"].replace(np.inf, np.nan)
                ratio = (pa / pb).dropna()
                bad = ratio[(ratio - exp).abs() > 1e-4 * max(1, exp)]
                add(f"rooftop potential ratio a/b == {exp:.3f} (solar_scaling {tuple(solar)})", bad.empty,
                    "; ".join(f"{i}: {v:.4f}" for i, v in bad.items()))
            if headroom is not None:
                bad = []
                for bus, g in _dist_groups(n_t).items():
                    h = headroom[0] if bus.endswith(" a") else headroom[1]
                    if h is None:
                        continue
                    base = [l for l in g["fwd"] if "reinforcement" not in l]
                    cap = n_t.links.loc[base, "p_nom"].sum()
                    loads = n_t.loads.index[(n_t.loads.bus == bus) & n_t.loads.carrier.str.contains("electric", na=False)]
                    peak = n_t.loads_t.p_set.reindex(columns=loads).fillna(0).sum(axis=1).max()
                    if abs(cap - h * peak) > 1e-6 * max(cap, 1) + 1e-6:
                        bad.append((bus, round(cap, 2), round(h * peak, 2)))
                add(f"fixed link capacity == headroom {tuple(headroom)} x peak LV load per half", not bad, str(bad[:3]))

    halves = df.index.get_level_values("half")
    split = set(halves) >= {"a", "b"}

    # reference comparison of node totals
    if ref is not None:
        obj_t, obj_r = float(n_t.objective), float(ref.objective)
        rel = abs(obj_t - obj_r) / max(abs(obj_r), 1e-9)
        tot_t, tot_r = node_totals(df), node_totals(summarise(ref))
        common = tot_t.index.intersection(tot_r.index)
        if expect == "symmetric":
            add("objective equal to reference (rel < 1e-4)", rel < 1e-4, f"{obj_t:.8e} vs {obj_r:.8e} (rel {rel:.2e})")
            print(f"[INFO] objective: test {obj_t:.8e} vs reference {obj_r:.8e} (rel diff {rel:.2e})")
            for col in tot_t.columns:
                floor = max(1e-3, 1e-3 * tot_r[col].abs().max())
                r = _rel(tot_t.loc[common, col], tot_r.loc[common, col], floor)
                bad = r[r > tol]
                add(f"node total {col} ~ reference", bad.empty,
                    ("operation (flows / battery throughput) is not unique in an LP with slack; judge by capacities, net import and objective. " if col in SOFT_COLS else "")
                    + "; ".join(f"{i}: {tot_t.loc[i, col]:.3g} vs {tot_r.loc[i, col]:.3g}" for i in bad.index[:3]),
                    hard=col not in SOFT_COLS)
        else:
            print(f"\n[INFO] objective vs reference: {obj_t:.6e} vs {obj_r:.6e} (rel {rel:+.2e})")
            d = (tot_t.loc[common] - tot_r.loc[common])
            print("[INFO] node totals, test minus reference:\n" + d.round(3).to_string())

    if split:
        a, b = df.xs("a", level="half"), df.xs("b", level="half")
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        cols = [c for c in ["rooftop_p_nom_opt_MW", "batt_e_nom_opt_MWh", "batt_discharged_GWh",
                            "dist_net_import_GWh", "dist_import_GWh", "dist_export_GWh", "lv_demand_GWh"] if c in df.columns]
        if expect == "symmetric":
            # LP can break ties between identical halves, so a/b differences are a WARNING not a failure
            # demand and rooftop POTENTIAL are inputs (no solver freedom): a != b means the scaling is active
            deterministic = {"lv_demand_GWh", "rooftop_p_nom_max_MW"}
            cols = cols + (["rooftop_p_nom_max_MW"] if "rooftop_p_nom_max_MW" in df.columns else [])
            for col in cols:
                x, y = a[col].replace(np.inf, np.nan).fillna(0), b[col].replace(np.inf, np.nan).fillna(0)
                floor = max(1e-3, 1e-3 * max(x.abs().max(), y.abs().max()))
                r = _rel(x, y, floor)
                det = col in deterministic
                add(f"halves a ~ b: {col}" + (" (input)" if det else ""), (r <= tol).all(),
                    ("" if det else "tie-breaking between identical halves is possible; totals matter more. ")
                    + "; ".join(f"{i}: {x.at[i]:.3g} vs {y.at[i]:.3g}" for i in r[r > tol].index[:3]),
                    hard=det)
        else:
            dem = _rel(a["lv_demand_GWh"], b["lv_demand_GWh"], 1e-3) if "lv_demand_GWh" in cols else None
            flow = pd.concat([_rel(a[c], b[c], 1e-3) for c in ["dist_net_import_GWh", "dist_import_GWh", "dist_export_GWh"] if c in cols], axis=1)
            stor = pd.concat([_rel(a[c], b[c], 1e-3) for c in ["batt_discharged_GWh", "batt_e_nom_opt_MWh"] if c in cols], axis=1)
            if dem is not None:
                add("halves differ in LV demand (demand scaling took effect)", (dem > 1e-3).any(),
                    f"max rel diff a/b = {dem.max():.3f}")
            add("halves differ in distribution flows", (flow.max(axis=1) > tol).any(),
                f"max rel diff a/b = {flow.max().max():.3f} (flows can legitimately be equal or zero)", hard=False)
            add("halves differ in storage operation / size", (stor.max(axis=1) > tol).any(),
                f"max rel diff a/b = {stor.max().max():.3f}", hard=False)
    else:
        add("LV split present (halves a and b)", expect == "symmetric" and ref is None,
            "no a/b halves found in the test network", hard=expect != "symmetric")
    return df, res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("network")
    p.add_argument("--reference")
    p.add_argument("--expect", choices=["symmetric", "asymmetric"], default="symmetric")
    p.add_argument("--tol", type=float, default=0.02)
    p.add_argument("--csv")
    p.add_argument("--solar", nargs=2, type=float, help="solar_scaling used by this run")
    p.add_argument("--demand", nargs=2, type=float, help="demand_scaling used by this run")
    p.add_argument("--headroom", nargs=2, type=float, help="headroom used by this run (omit if unconstrained)")
    a = p.parse_args()

    n = pypsa.Network(a.network)
    ref = pypsa.Network(a.reference) if a.reference else None
    sol = tuple(a.solar) if a.solar else None
    dem = tuple(a.demand) if a.demand else None
    # demand_scaling is renormalised to sum 2 in the model; shares are what matter
    df, res = run_checks(n, ref, a.expect, a.tol, sol, dem, tuple(a.headroom) if a.headroom else None)

    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.float_format", "{:.3f}".format):
        print("\nPer node / half summary:\n" + df.to_string())
    if a.csv:
        df.to_csv(a.csv)

    print()
    ok_all = True
    for name, ok, detail, hard in res:
        tag = "PASS" if ok else ("FAIL" if hard else "WARN")
        print(f"[{tag}] {name}" + (f"  -> {detail}" if detail and not ok else (f"  -> {detail}" if detail and name.startswith(("elements", "halves differ")) else "")))
        ok_all &= ok or not hard
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()