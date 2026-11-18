#!/usr/bin/env python3
"""
Post-solve analysis of the transport (EV) split for the split-LV distribution grid.

Sits NEXT TO analyze_split_lv.py and reuses its helpers (network search, LV metrics,
baseline pricing, plots). Same CLI, so the F-series commands mirror the E-series:

    python analyze_split_transport.py --network $net --run $F2 --baseline-run $F1 --expect-equal --out analysis/F2_free_3H
    python analyze_split_transport.py --network $net --run $F3 --baseline-run $F1 --expect-constrained --out analysis/F3_08_3H
    python analyze_split_transport.py --network $net --run $F4 --baseline-run $F1 --expect-constrained --expect-reinforcement --out analysis/F4_08_reinf_3H
    python analyze_split_transport.py --network $net --baseline-run $F1 --compare $F2 $F3 $F4 --out analysis/F_sweep_3H

What it adds on top of the LV analysis
--------------------------------------
T1. EV structural checks   - EV battery buses/stores/loads split, chargers/V2G on matching halves
                             (or a clearly reported SHARED chain for the F1 baseline).
T2. EV solution checks     - charger/V2G flow limits, EV battery state of charge within bounds,
                             little simultaneous charge + V2G.
T3. EV metrics per half    - EV demand, charger / V2G / store capacity, energy charged and
                             returned, energy SHIFTED in time vs the driving profile, peak
                             reduction, V2G full-load hours.
Plots: EV capacity, flexibility used, mean daily EV profile, LV peak decomposition,
plus the LV flow / price / binding plots from analyze_split_lv.py.

Baseline note: with the EV split off (F1) the EV load and store sit on ONE shared bus per node.
Their per-half figures are then attributed by each half's share of charger capacity, so F1 and
the split runs stay comparable.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import analyze_split_lv as base
except ImportError:
    sys.exit("analyze_split_transport.py must sit next to analyze_split_lv.py "
             "(or change the import name at the top of this file).")

EV_LOAD, BEVC, V2G, EVST = base.EV_LOAD, base.BEVC, base.V2G, base.EVST
COLORS = base.COLORS


# --------------------------------------------------------------------------- #
# T1. structural checks
# --------------------------------------------------------------------------- #
def ev_structural_checks(n):
    print("\n=== T1. EV STRUCTURAL CHECKS ===")
    ac = n.buses.index[n.buses.carrier == "AC"]
    lv = set(base.lv_bus_set(n))
    ev_b = n.buses.index[n.buses.carrier == EVST]
    chg = n.links[n.links.carrier == BEVC]
    v2g = n.links[n.links.carrier == V2G]
    evl = n.loads[n.loads.carrier == EV_LOAD]
    evs = n.stores[n.stores.carrier == EVST]

    base.check(len(ev_b) > 0, "EV battery buses exist", f"{len(ev_b)} found")
    base.check(len(chg) > 0, "BEV chargers exist", f"{len(chg)} found")
    base.check(len(evl) > 0, "EV load exists", f"{len(evl)} found")
    if not len(ev_b):
        return
    base.info("V2G links", f"{len(v2g)} found" if len(v2g) else "none (V2G disabled?)")

    split = any(b.endswith((" a", " b")) for b in ev_b)
    lv_split = any(b.endswith((" a", " b")) for b in lv)

    # chargers / V2G sit on LV buses
    base.check(chg.bus0.isin(lv).all(), "every BEV charger draws from an LV bus")
    if len(v2g):
        base.check(v2g.bus1.isin(lv).all(), "every V2G link delivers to an LV bus")

    if not split:
        base.info("EV chain is SHARED per node (EV split off)",
                  "EV load/store on one bus per node; chargers/V2G on the LV halves")
        base.check(len(ev_b) == len(ac), "one EV battery bus per AC node", f"{len(ev_b)} vs {len(ac)}")
        return

    base.check(all(b.endswith((" a", " b")) for b in ev_b), "EV battery buses are split into a/b")
    base.check(len(ev_b) == 2 * len(ac), "2 EV battery buses per AC node", f"{len(ev_b)} EV vs {len(ac)} AC")
    old = [b for b in ev_b if not b.endswith((" a", " b"))]
    base.check(not old, "no leftover unsplit EV buses", str(old[:3]) if old else "")
    base.check(set(evs.bus) <= set(ev_b), "EV stores sit on EV half buses", f"{len(evs)} stores / {len(ev_b)} buses")
    base.check(set(evl.bus) <= set(ev_b), "EV loads sit on EV half buses")
    lk = n.links[n.links.carrier.isin([BEVC, V2G])]
    bad = [l for l, r in lk.iterrows() if base.bus_half(r.bus0) != base.bus_half(r.bus1)]
    base.check(not bad, "chargers/V2G connect same-letter LV and EV halves", str(bad[:3]) if bad else "")
    if lv_split:
        na, nb = sum(b.endswith(" a") for b in ev_b), sum(b.endswith(" b") for b in ev_b)
        base.check(na == nb, "same number of EV buses on a and b", f"a={na}, b={nb}")


# --------------------------------------------------------------------------- #
# T3. EV metrics
# --------------------------------------------------------------------------- #
def ev_metrics(n):
    """Per-half EV metrics and the profiles used for plotting."""
    w = base.weights(n)
    K, L, S = n.links, n.loads, n.stores
    lp = base.load_power(n)
    lh, kh, sh = base.comp_half(L), base.comp_half(K), base.comp_half(S)
    evl = list(L.index[L.carrier == EV_LOAD])
    chg = list(K.index[K.carrier == BEVC])
    v2g = list(K.index[K.carrier == V2G])
    evs = list(S.index[S.carrier == EVST])
    if not chg:
        return pd.DataFrame(), {}

    capP = base.optcap(K.loc[chg], "p_nom")
    capV = base.optcap(K.loc[v2g], "p_nom") if v2g else pd.Series(dtype=float)
    capE = base.optcap(S.loc[evs], "e_nom") if evs else pd.Series(dtype=float)
    totP = float(capP.sum())
    ev_total = lp[evl].sum(axis=1) if evl else pd.Series(0.0, index=n.snapshots)
    p0, p1 = n.links_t.p0, n.links_t.p1
    halves = sorted(set(kh[chg]))
    d, series = {}, {}

    def put(name, h, v):
        d.setdefault(name, {})[h] = float(v)

    for h in halves:
        c_h = [x for x in chg if kh[x] == h]
        v_h = [x for x in v2g if kh[x] == h]
        l_h = [x for x in evl if lh[x] == h]
        s_h = [x for x in evs if sh[x] == h]
        share = float(capP[c_h].sum() / totP) if totP > 0 else 1.0 / len(halves)

        demand = lp[l_h].sum(axis=1) if l_h else ev_total * share      # shared bus: attribute by charger share
        store_E = float(capE[s_h].sum()) if s_h else float(capE.sum()) * share
        eff = float(K.loc[c_h, "efficiency"].mean())

        charge = p0[c_h].sum(axis=1)                                   # drawn from the LV bus
        sup = (-p1[v_h]).clip(lower=0).sum(axis=1) if v_h else 0.0 * charge   # delivered to the LV bus
        net = charge - sup
        inflex = demand / eff                                          # same energy, charged when driven

        e_inflex = float((inflex * w).sum() / 1e3)
        shift = 0.5 * float(((net - inflex).abs() * w).sum() / 1e3)
        sim = float((np.minimum(charge, sup) * w).sum() / 1e3)
        chg_gwh = float((charge * w).sum() / 1e3)
        v2g_gwh = float((sup * w).sum() / 1e3)
        v2g_P = float(capV[v_h].sum()) if v_h else 0.0

        put("ev_load_GWh", h, (demand * w).sum() / 1e3)
        put("ev_charger_P_MW", h, capP[c_h].sum())
        put("ev_v2g_P_MW", h, v2g_P)
        put("ev_store_MWh", h, store_E)
        put("ev_charge_GWh", h, chg_gwh)
        put("ev_v2g_GWh", h, v2g_gwh)
        put("ev_shift_GWh", h, shift)
        put("ev_shift_pct", h, 100 * shift / e_inflex if e_inflex > 0 else np.nan)
        put("ev_peak_net_MW", h, net.max())
        put("ev_peak_inflex_MW", h, inflex.max())
        put("ev_peak_reduction_pct", h,
            100 * (inflex.max() - net.max()) / inflex.max() if inflex.max() > 0 else np.nan)
        put("ev_v2g_full_load_h", h, v2g_gwh * 1e3 / v2g_P if v2g_P > 0 else np.nan)
        put("ev_simultaneous_pct", h, 100 * sim / chg_gwh if chg_gwh > 0 else 0.0)
        series[("EV net draw", h)] = net
        series[("EV inflexible profile", h)] = inflex

    return pd.DataFrame(d), series


EV_SUM_COLS = ["ev_load_GWh", "ev_charger_P_MW", "ev_v2g_P_MW", "ev_store_MWh", "ev_charge_GWh",
               "ev_v2g_GWh", "ev_shift_GWh", "ev_peak_net_MW", "ev_peak_inflex_MW"]


def ev_totals(fm):
    return pd.Series({c: fm[c].sum() for c in EV_SUM_COLS if c in fm})


# --------------------------------------------------------------------------- #
# T2. solution checks
# --------------------------------------------------------------------------- #
def ev_solution_checks(n, fm, tol):
    print("\n=== T2. EV SOLUTION CHECKS ===")
    K, S = n.links, n.stores
    scale = max(1.0, 1e-2 * float(base.load_power(n).sum(axis=1).max()))

    for name in (BEVC, V2G):
        ls = K[K.carrier == name]
        if ls.empty:
            continue
        cap = base.optcap(ls, "p_nom")
        pmax = base.dense(n, "Link", "p_max_pu")[ls.index]
        pmin = base.dense(n, "Link", "p_min_pu")[ls.index]
        p0 = n.links_t.p0[ls.index]
        viol = max((p0 - pmax * cap).max().max(), (pmin * cap - p0).max().max())
        base.check(viol <= tol * scale, f"{name} flows within limits", f"max violation {viol:.3e} MW")

    evs = S[S.carrier == EVST]
    if len(evs):
        e = n.stores_t.e[evs.index]
        e_nom = base.optcap(evs, "e_nom")
        emin = base.dense(n, "Store", "e_min_pu")[evs.index] * e_nom
        emax = base.dense(n, "Store", "e_max_pu")[evs.index] * e_nom
        base.check((e - emin).min().min() >= -tol * scale, "EV battery state of charge above its minimum",
                   f"worst {(e - emin).min().min():.3e} MWh")
        base.check((e - emax).max().max() <= tol * scale, "EV battery state of charge below its maximum",
                   f"worst {(e - emax).max().max():.3e} MWh")
    else:
        base.info("no EV stores in network")

    if not fm.empty and "ev_simultaneous_pct" in fm:
        mx = float(fm.ev_simultaneous_pct.max())
        base.check(mx < 1.0, "little simultaneous charging and V2G", f"max {mx:.2f}% of charged energy",
                   warn_only=True)
        if "ev_v2g_GWh" in fm and fm.ev_v2g_GWh.sum() < 1e-3 * max(fm.ev_load_GWh.sum(), 1e-9):
            base.info("V2G is barely used", f"{fm.ev_v2g_GWh.sum():.3f} GWh vs EV demand {fm.ev_load_GWh.sum():.1f} GWh")


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def plot_ev_capacity(fm, out):
    cols_p = [c for c in ("ev_charger_P_MW", "ev_v2g_P_MW") if c in fm]
    cols_e = [c for c in ("ev_store_MWh",) if c in fm]
    if not cols_p and not cols_e:
        return
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, cs, ttl, yl in ((axs[0], cols_p, "EV power capacity", "MW"),
                            (axs[1], cols_e, "EV battery energy capacity", "MWh")):
        x = np.arange(len(fm))
        k = max(len(cs), 1)
        for i, c in enumerate(cs):
            ax.bar(x + (i - (k - 1) / 2) * 0.8 / k, fm[c].fillna(0), 0.8 / k, label=c)
        ax.set_xticks(x, fm.index)
        ax.set(title=ttl, ylabel=yl)
        if cs:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "t01_ev_capacity.png", dpi=150)
    plt.close(fig)


def plot_ev_activity(fm, out):
    cols = [c for c in ("ev_charge_GWh", "ev_shift_GWh", "ev_v2g_GWh") if c in fm]
    if not cols:
        return
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(fm))
    k = len(cols)
    for i, c in enumerate(cols):
        axs[0].bar(x + (i - (k - 1) / 2) * 0.8 / k, fm[c].fillna(0), 0.8 / k, label=c)
    axs[0].set_xticks(x, fm.index)
    axs[0].set(ylabel="GWh", title="EV energy: charged, shifted, returned (V2G)")
    axs[0].legend(fontsize=8)
    if "ev_peak_reduction_pct" in fm:
        axs[1].bar(x, fm.ev_peak_reduction_pct.fillna(0), color=[COLORS.get(h, "tab:gray") for h in fm.index])
    axs[1].set_xticks(x, fm.index)
    axs[1].set(ylabel="%", title="EV peak reduction vs the driving profile")
    fig.tight_layout()
    fig.savefig(out / "t02_ev_activity.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# comparing a set of runs
# --------------------------------------------------------------------------- #
def run_summary(n, cost_mw=None, obj_base=None):
    r = base.run_summary(n, cost_mw, obj_base)
    fm, _ = ev_metrics(n)
    if not fm.empty:
        ft = ev_totals(fm)
        for k in ("ev_shift_GWh", "ev_v2g_GWh", "ev_v2g_P_MW", "ev_charger_P_MW", "ev_store_MWh"):
            if k in ft:
                r[k] = ft[k]
        r["ev_peak_reduction_pct_max"] = float(fm.ev_peak_reduction_pct.max())
    return r


def run_compare(args):
    cost_mw = obj_base = None
    if args.baseline or args.baseline_run:
        nb, _ = base.load_solved(args.baseline or args.network, args.baseline_run, args.root)
        cost_mw, obj_base = base.baseline_cost_mw(nb), float(nb.objective)
        print(f"baseline objective {obj_base:,.0f}; distribution capex {cost_mw:,.0f} EUR/MW")
    rows, missing = {}, []
    for run in args.compare:
        print(f"\n--- {run}")
        n, _ = base.load_solved(args.network, run, args.root, fatal=False)
        if n is None:
            missing.append(run)
            continue
        rows[run] = run_summary(n, cost_mw, obj_base)
    if missing:
        print("\nNo solved network (likely infeasible or not run yet): " + ", ".join(missing))
    if not rows:
        return 1
    df = pd.DataFrame(rows).T
    if "existing_grid_MW" in df:
        df = df.sort_values("existing_grid_MW")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220, "display.max_columns", 60, "display.float_format", "{:,.2f}".format)
    print("\n=== RUN COMPARISON (transport) ===")
    print(df.T)
    df.to_csv(out / "compare_runs.csv")
    base.plot_compare(df, out)
    print(f"Saved to {out.resolve()}")
    return 0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--network", required=True, help="path to solved .nc OR a name/pattern")
    ap.add_argument("--run", help="run name (folder under results/) to search in")
    ap.add_argument("--baseline", help="baseline .nc path or name/pattern (default: same name as --network)")
    ap.add_argument("--baseline-run", help="run name of the baseline (searched under results/)")
    ap.add_argument("--root", default="results")
    ap.add_argument("--expect-constrained", action="store_true",
                    help="FAIL if no distribution link is fixed (headroom silently not applied)")
    ap.add_argument("--expect-reinforcement", action="store_true", help="FAIL if the run has no reinforcement links")
    ap.add_argument("--compare", nargs="+", metavar="RUN", help="compare several runs in one table and plot")
    ap.add_argument("--out", default="analysis_split_transport")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--expect-equal", action="store_true",
                    help="require objective == baseline (EV split (1,1), no headroom test)")
    args = ap.parse_args()

    if args.compare:
        return run_compare(args)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n, _ = base.load_solved(args.network, args.run, args.root)
    hrs = float(base.weights(n).mean())
    print(f"  {len(n.buses)} buses, {len(n.links)} links, {len(n.snapshots)} snapshots, mean resolution {hrs:.1f} h")
    months = sorted(set(base.time_axis(n).month))
    base.info("snapshot window", f"{base.time_axis(n)[0]:%Y-%m-%d} to {base.time_axis(n)[-1]:%Y-%m-%d}, months {months}")
    if hrs > 6:
        base.check(False, "temporal resolution too coarse for peak-based limits",
                   f"mean resolution {hrs:.0f} h: peaks are averaged out", warn_only=True)

    # LV-level structure (split buses, distribution links, constraint state) + EV chain
    base.structural_checks(n)
    ev_structural_checks(n)
    d_chk, _ = base.dist_interface(n)
    fwd_chk = d_chk[(~d_chk.rev) & ~d_chk.index.map(base.is_reinf)]
    if args.expect_constrained:
        base.check(bool((~fwd_chk.p_nom_extendable).any()), "headroom constraint is applied (fixed distribution links exist)",
                   "no fixed links: check the headroom key in config.yaml and the prepare log")
    if args.expect_reinforcement:
        base.check(bool(d_chk.index.map(base.is_reinf).any()), "reinforcement links exist", "reinforce flag was not applied")
    base.solution_checks(n, args.tol)

    print("\n=== T3. METRICS ===")
    bus_table, half = base.compute_metrics(n)
    pd.set_option("display.width", 200, "display.max_columns", 50, "display.float_format", "{:,.2f}".format)
    print("\nPer LV half:")
    print(half.T)
    bus_table.to_csv(out / "metrics_per_lv_bus.csv")
    half.to_csv(out / "metrics_per_half.csv")

    fm, series = ev_metrics(n)
    ev_solution_checks(n, fm, args.tol)
    if fm.empty:
        base.info("no EV chargers found -> EV metrics skipped")
    else:
        print("\nEV per LV half:")
        print(fm.T)
        fm.to_csv(out / "metrics_ev_per_half.csv")
        if {"a", "b"} <= set(fm.index):
            for c in ("ev_load_GWh", "ev_charger_P_MW", "ev_v2g_P_MW", "ev_shift_GWh", "ev_peak_net_MW"):
                if c in fm:
                    base.info(f"a vs b: {c}", f"a={fm.at['a', c]:.2f}  b={fm.at['b', c]:.2f}")

    tot = pd.concat([base.totals(n, half), ev_totals(fm) if not fm.empty else pd.Series(dtype=float)])
    tot.to_csv(out / "totals.csv")

    if "bind_import_pct" in half and half.bind_import_pct.notna().any():
        mx = max(half.bind_import_pct.max(), half.bind_export_pct.max())
        base.check(mx > 0.5, "headroom limit binds somewhere (>0.5% of time)",
                   f"max time at limit {mx:.1f}%", warn_only=True)

    # baseline comparison
    if args.baseline or args.baseline_run:
        print("\n=== BASELINE COMPARISON ===")
        nb, _ = base.load_solved(args.baseline or args.network, args.baseline_run, args.root)
        _, half_b = base.compute_metrics(nb)
        fm_b, _ = ev_metrics(nb)
        tot_b = pd.concat([base.totals(nb, half_b), ev_totals(fm_b) if not fm_b.empty else pd.Series(dtype=float)])
        cmp_df = pd.DataFrame({"baseline": tot_b, "this_run": tot})
        cmp_df["delta"] = cmp_df.this_run - cmp_df.baseline
        cmp_df["delta_pct"] = 100 * cmp_df.delta / cmp_df.baseline.abs().replace(0, np.nan)
        print(cmp_df)
        cmp_df.to_csv(out / "baseline_comparison.csv")
        rel = (tot["objective"] - tot_b["objective"]) / abs(tot_b["objective"])
        if args.expect_equal:
            base.check(abs(rel) < 1e-4, "objective equals baseline (EV split is neutral)", f"rel. difference {rel:.2e}")
            if not fm.empty and not fm_b.empty:
                e1, e0 = float(fm.ev_charge_GWh.sum()), float(fm_b.ev_charge_GWh.sum())
                base.check(abs(e1 - e0) <= 1e-3 * max(float(fm.ev_load_GWh.sum()), 1e-9),
                           "EV charging energy equals baseline (tolerance scaled by EV demand)", f"{e0:.3f} vs {e1:.3f} GWh")
        else:
            base.info("objective vs baseline", f"rel. difference {rel:+.3e} (raw; fixed grid is free)")
            cost_mw = base.baseline_cost_mw(nb)
            adj, fixed_mw = base.adjusted_objective(n, cost_mw)
            if fixed_mw > 0:
                rel_adj = (adj - tot_b["objective"]) / abs(tot_b["objective"])
                base.info("adjusted objective (fixed grid priced at baseline capex)",
                          f"{adj:,.0f}  ({rel_adj:+.2%} vs baseline; existing grid {fixed_mw:,.0f} MW at {cost_mw:,.0f} EUR/MW)")
                base.check(rel_adj > -1e-4, "adjusted objective >= baseline (constraint cannot make the system cheaper)",
                           f"{rel_adj:+.3%}")
        base.plot_baseline(tot, tot_b, out)

    base.flex_checks(n, args.tol)
    T, times = base.peak_tables(base.lv_decomposition(n))
    if not T.empty:
        print("\nLV peak decomposition (MW, + consumption / - supply):")
        print(T)
        T.to_csv(out / "lv_peak_decomposition.csv")
        for (h, kind), t in times.items():
            base.info(f"{kind} half {h}", f"at {t}")

    print("\n=== PLOTS ===")
    base.plot_flow_duration(n, bus_table, out)
    base.plot_price_duration(n, bus_table, out)
    base.plot_flows_and_binding(half, out)
    if not fm.empty:
        plot_ev_capacity(fm, out)
        plot_ev_activity(fm, out)
        base.plot_mean_day(n, series, out)
    if not T.empty:
        base.plot_peak_decomposition(T, times, out)
    print(f"Saved to {out.resolve()}")

    res = pd.DataFrame(base.RESULTS)
    res.to_csv(out / "checks.csv", index=False)
    n_fail = int((res.status == "FAIL").sum())
    n_warn = int((res.status == "WARN").sum())
    print(f"\nSUMMARY: {(res.status == 'PASS').sum()} pass, {n_warn} warn, {n_fail} fail")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())