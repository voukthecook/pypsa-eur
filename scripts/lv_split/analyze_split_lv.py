#!/usr/bin/env python3
"""
Post-solve analysis for the split-LV distribution grid extension of PyPSA-Eur.

What it does
------------
1. Structural checks   - is the split topology intact (buses, links, batteries)?
2. Solution checks     - energy balance, limits respected, battery sanity.
3. Metrics             - per LV half (a / b) and per LV bus: flows, binding hours,
                         rooftop PV use and curtailment, batteries, LV prices.
4. Plots               - flow duration curves, price duration curves, PV, batteries,
                         example dispatch week, (optional) baseline comparison.
5. Outputs             - CSVs + PNGs + checks.csv in the output folder.

Usage
-----
python analyze_split_lv.py \\
    --network base_s_3_lv1.2_Co2L0-168H_2030 --run <run name> \\
    [--baseline-run <baseline run name>] [--out analysis/<name>] [--expect-equal]

--network accepts a full path to the solved .nc, or just the file name/pattern;
the script then searches results/ (override with --root), skips prenetworks
and files without a solution, and uses the newest solved match.

Extra options
-------------
--flexoff-run RUN   run with flexibility switched off (same grid settings): value of flexibility
--compare R1 R2 ... compare several runs (headroom sweep etc.) in one table and plot; no single-run
                    analysis. Runs without a solved network (infeasible?) are listed as missing.

--expect-equal : use for the "split (1,1), no headroom" test, where the objective
                 must match the baseline. Do NOT use it for constrained runs.

Note on comparing objectives
----------------------------
constrain_lv_links() sets capital_cost = 0 on the fixed existing distribution
links (sunk grid), while the unsplit baseline pays capital cost for its
extendable distribution link. A constrained run's objective can therefore be
LOWER than the baseline without anything being wrong. Compare dispatch,
curtailment and investment rather than raw objectives in that case.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

logging.getLogger("pypsa").setLevel(logging.ERROR)

DIST = "electricity distribution grid"
ROOF = "solar rooftop"
BATT = "home battery"
CHG = "home battery charger"
DCH = "home battery discharger"
COLORS = {"a": "tab:blue", "b": "tab:orange", "all": "tab:gray"}

RESULTS = []


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def check(ok, name, detail="", warn_only=False):
    status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    RESULTS.append({"status": status, "check": name, "detail": detail})
    print(f"[{status}] {name}" + (f"  --  {detail}" if detail else ""))
    return ok


def info(name, detail=""):
    RESULTS.append({"status": "INFO", "check": name, "detail": detail})
    print(f"[INFO] {name}" + (f"  --  {detail}" if detail else ""))


def bus_half(bus):
    return bus[-1] if bus.endswith((" a", " b")) else "all"


def is_reinf(name):
    return " reinforcement" in name


def dist_interface(n):
    """
    Distribution links attached to LV buses, forward and reversed.
    PyPSA-Eur's lossy bidirectional mechanism adds '<name>-reversed' links
    (bus0 = LV, bus1 = AC) that carry the LV -> HV (export) direction, while the
    forward link (bus0 = AC, bus1 = LV) carries import with p_min_pu = 0.
    Returns (attached links with columns lv/rev/cap, stray links).
    """
    lv = n.buses.index[n.buses.carrier == "low voltage"]
    d = n.links[n.links.carrier == DIST].copy()
    d["rev"] = d.bus0.isin(lv)
    d["lv"] = np.where(d.rev, d.bus0, np.where(d.bus1.isin(lv), d.bus1, ""))
    d["cap"] = np.where(d.p_nom_extendable, d.p_nom_opt, d.p_nom)
    return d[d.lv != ""], d[d.lv == ""]


def find_networks(pattern, run=None, root="results"):
    """Resolve a file path, or a name/pattern searched under `root` (recursive)."""
    p = Path(pattern)
    if p.is_file():
        return [p]
    stem = p.stem if p.suffix == ".nc" else pattern
    hits = list(Path(root).rglob(f"*{stem}*.nc"))
    if run:
        hits = [f for f in hits if run in f.parts]
    # solved networks first (not prenetworks), then newest
    hits.sort(key=lambda f: ("prenetworks" in f.parts, -f.stat().st_mtime))
    return hits


def load_solved(pattern, run=None, root="results", fatal=True):
    def fail(msg):
        if fatal:
            sys.exit(msg)
        print(f"   [no solved network] {msg}")
        return None, None

    hits = find_networks(pattern, run, root)
    if not hits:
        return fail(f"No .nc file matching '{pattern}' under '{root}'"
                    + (f" in run '{run}'" if run else ""))
    print(f"Candidates for '{pattern}':")
    for f in hits[:8]:
        print(f"   {f}")
    for f in hits:
        try:
            n = pypsa.Network(str(f))
        except Exception as e:
            print(f"   skip {f.name}: {e}")
            continue
        if not n.generators_t.p.empty:
            print(f"-> using solved network {f}")
            return n, f
        print(f"   skip {f} (no solution inside, probably a prenetwork)")
    return fail("Found files, but none contains a solved network.")


def weights(n):
    return n.snapshot_weightings.generators


def time_axis(n):
    idx = n.snapshots
    return idx.get_level_values(-1) if isinstance(idx, pd.MultiIndex) else idx


def load_power(n):
    """Load power per snapshot (time-varying p_set overrides static p_set)."""
    p = pd.DataFrame(
        np.tile(n.loads.p_set.values, (len(n.snapshots), 1)),
        index=n.snapshots,
        columns=n.loads.index,
    )
    if not n.loads_t.p_set.empty:
        p.update(n.loads_t.p_set)
    return p


def dense(n, component, attr):
    try:
        return n.get_switchable_as_dense(component, attr)
    except Exception:
        df = getattr(n, component.lower() + "s")
        return pd.DataFrame(
            np.tile(df[attr].values, (len(n.snapshots), 1)),
            index=n.snapshots,
            columns=df.index,
        )


def group_cols(df, key):
    """Sum columns of df by the Series `key` (index = column names)."""
    cols = [c for c in df.columns if c in key.index]
    if not cols:
        return pd.DataFrame(index=df.index)
    return df[cols].T.groupby(key[cols]).sum().T


def bus_balance(n):
    """Net injection per bus and snapshot; should be ~0 for a solved network."""
    bal = pd.DataFrame(0.0, index=n.snapshots, columns=n.buses.index)

    def add(df, bus_series, sign=1.0):
        if df is None or df.empty:
            return
        valid = bus_series[bus_series.isin(n.buses.index)]
        g = group_cols(df, valid)
        for b in g.columns:
            bal[b] = bal[b] + sign * g[b]

    add(n.generators_t.p, n.generators.bus)
    add(load_power(n), n.loads.bus, -1.0)
    add(n.stores_t.p, n.stores.bus)
    if len(n.storage_units):
        add(n.storage_units_t.p, n.storage_units.bus)
    for col in [c for c in n.links.columns if c.startswith("bus") and c[3:].isdigit()]:
        i = col[3:]
        if f"p{i}" in n.links_t:
            add(n.links_t[f"p{i}"], n.links[col], -1.0)
    # AC transmission lines (and transformers, if any)
    for comp, t in [("lines", n.lines), ("transformers", n.transformers)]:
        if len(t):
            for i in ("0", "1"):
                add(getattr(n, comp + "_t")[f"p{i}"], t[f"bus{i}"], -1.0)
    return bal


# --------------------------------------------------------------------------- #
# 1. structural checks
# --------------------------------------------------------------------------- #
def structural_checks(n):
    print("\n=== 1. STRUCTURAL CHECKS ===")
    lv = n.buses.index[n.buses.carrier == "low voltage"]
    ac = n.buses.index[n.buses.carrier == "AC"]

    check(len(lv) > 0, "LV buses exist", f"{len(lv)} found")
    unsplit = len(lv) > 0 and not any(b.endswith((" a", " b")) for b in lv)
    if unsplit:
        info("network is UNSPLIT (reference run)", "split-specific checks skipped")
    else:
        check(all(b.endswith((" a", " b")) for b in lv), "all LV buses are split into a/b")
        check(len(lv) == 2 * len(ac), "2 LV buses per AC node", f"{len(lv)} LV vs {len(ac)} AC")
        old = [b for b in n.buses.index
               if b.endswith(" low voltage") or b.endswith(" home battery")]
        check(not old, "no leftover unsplit LV / home battery buses", f"{old[:5]}" if old else "")

    d, stray = dist_interface(n)
    if len(stray):
        check(False, f"{len(stray)} links carry carrier '{DIST}' but are not attached to an LV bus",
              "; ".join(f"{i}: {r.bus0}->{r.bus1}" for i, r in stray.head(6).iterrows()))
    fwd = d[(~d.rev) & ~d.index.map(is_reinf)]
    rev = d[d.rev]
    check(set(fwd.lv) == set(lv) and len(fwd) == len(lv),
          "every LV bus has exactly one forward (import) distribution link",
          f"{len(fwd)} forward links for {len(lv)} LV buses")
    check(set(fwd.bus0) <= set(ac), "forward distribution links start at AC buses")
    if len(rev):
        info("reversed (export) distribution links present (lossy bidirectional)",
             f"{len(rev)} links; forward p_min_pu values: {sorted(set(fwd.p_min_pu))}")
        check(set(rev.lv) == set(lv), "every LV bus has a reversed (export) link")
    else:
        info("no reversed links: export runs through the forward links (p_min_pu < 0)")
    # every LV bus is actually used by something besides the distribution link
    used = set(n.loads.bus) | set(n.generators.bus)
    for c in [c for c in n.links.columns if c.startswith("bus") and c[3:].isdigit()]:
        used |= set(n.links[c])
    check(set(lv) <= used, "no dangling LV bus")

    # batteries
    hb = n.buses.index[n.buses.carrier == BATT]
    st = n.stores[n.stores.carrier == BATT]
    if len(hb):
        if not unsplit:
            check(all(b.endswith((" a", " b")) for b in hb), "home battery buses are split")
        check(set(st.bus) == set(hb), "one home battery store per battery bus",
              f"{len(st)} stores / {len(hb)} buses")
        bl = n.links[n.links.carrier.isin([CHG, DCH])]
        bad = [
            l
            for l, r in bl.iterrows()
            if bus_half(r.bus0) != bus_half(r.bus1)
            and "a" in (bus_half(r.bus0), bus_half(r.bus1))
        ]
        check(not bad, "battery chargers/dischargers stay within their own half",
              f"{bad[:3]}" if bad else "no cross-half battery path")
        # stronger check: no battery link connects LV half x to battery bus of half y
        mism = [l for l, r in bl.iterrows() if bus_half(r.bus0) != bus_half(r.bus1)]
        check(not mism, "no battery link mixes a and b buses", f"{mism[:3]}" if mism else "")
    else:
        info("no home battery buses in network")

    # constraint state
    fixed = fwd[~fwd.p_nom_extendable]
    reinf = d[d.index.map(is_reinf)]
    if len(fixed):
        info("headroom constraint detected",
             f"{len(fixed)}/{len(fwd)} forward links fixed; {len(reinf)} reinforcement links")
        check((fixed.capital_cost == 0).all(), "fixed existing links are sunk (capital_cost = 0)",
              warn_only=True)
        if len(rev):
            rev_fixed = rev[~rev.p_nom_extendable]
            leak = sorted(set(fixed.lv) - set(rev_fixed.lv))
            check(not leak, "export side is capped wherever the import side is",
                  f"LV buses with fixed import but EXTENDABLE export link: {leak[:4]}" if leak
                  else "reversed links are fixed too")
            a = fixed.set_index("lv").cap
            b = rev_fixed.set_index("lv").cap.reindex(a.index)
            check(np.allclose(a, b, equal_nan=True), "import and export capacities are equal",
                  f"max diff {np.nanmax(np.abs(a - b)):.3g} MW", warn_only=True)
        else:
            check((fixed.p_min_pu == -1).all(), "fixed links allow export (p_min_pu = -1)",
                  warn_only=True)
    else:
        info("no fixed distribution links -> run is UNCONSTRAINED (headroom not applied)")


# --------------------------------------------------------------------------- #
# 2. solution checks
# --------------------------------------------------------------------------- #
def solution_checks(n, tol):
    print("\n=== 2. SOLUTION CHECKS ===")
    w = weights(n)

    obj = getattr(n, "objective", None)
    check(obj is not None and np.isfinite(obj), "objective is finite", f"{obj}")

    solved = not n.generators_t.p.empty
    check(solved, "network contains a solution (generators_t.p filled)")
    if not solved:
        return

    # energy balance
    bal = bus_balance(n)
    resid = bal.abs().max()
    worst = resid.idxmax()
    scale = max(1.0, 1e-2 * float(load_power(n).sum(axis=1).max()))
    top = ", ".join(f"{b}: {v:.2e}" for b, v in resid.sort_values(ascending=False).head(4).items())
    check(resid.max() <= tol * scale, "nodal energy balance holds at every bus",
          f"max residual {resid.max():.3e} MW at '{worst}' (tol {tol * scale:.2e}); worst: {top}")

    # distribution link limits
    d = n.links[n.links.carrier == DIST]
    cap = pd.Series(np.where(d.p_nom_extendable, d.p_nom_opt, d.p_nom), index=d.index)
    pmax = dense(n, "Link", "p_max_pu")[d.index]
    pmin = dense(n, "Link", "p_min_pu")[d.index]
    p0 = n.links_t.p0[d.index]
    viol_up = (p0 - pmax * cap).max().max()
    viol_lo = (pmin * cap - p0).max().max()
    check(max(viol_up, viol_lo) <= tol * scale, "distribution link flows within limits",
          f"max violation {max(viol_up, viol_lo):.3e} MW")

    # rooftop PV
    g = n.generators[n.generators.carrier == ROOF]
    if len(g):
        over = (g.p_nom_opt - g.p_nom_max).max()
        check(over <= tol * scale, "rooftop PV capacity <= potential", f"max excess {over:.3e} MW")
        avail = dense(n, "Generator", "p_max_pu")[g.index] * g.p_nom_opt
        excess = (n.generators_t.p[g.index] - avail).max().max()
        check(excess <= tol * scale, "rooftop PV output <= availability", f"max excess {excess:.3e} MW")
        nzero = int((g.p_nom_opt < 1e-6).sum())
        info("rooftop PV generators with ~zero capacity", f"{nzero}/{len(g)}")
    else:
        info("no rooftop PV generators in network")

    # battery sanity
    chg = n.links[n.links.carrier == CHG]
    dch = n.links[n.links.carrier == DCH]
    if len(chg) and len(dch):
        c = n.links_t.p0[chg.index].copy()
        c.columns = chg.bus0.values
        dd = n.links_t.p0[dch.index].copy()
        dd.columns = dch.bus1.values
        c = c.T.groupby(level=0).sum().T
        dd = dd.T.groupby(level=0).sum().T
        demand_e = load_power(n).sum(axis=1).mul(w).sum()
        if c.mul(w, axis=0).sum().sum() < 1e-4 * demand_e:
            info("negligible home battery throughput -> simultaneity check skipped")
            return
        both = np.minimum(c, dd.reindex(columns=c.columns).fillna(0))
        thr = (c.mul(w, axis=0)).sum().sum()
        sim = (both.mul(w, axis=0)).sum().sum()
        share = sim / thr if thr > 1e-9 else 0.0
        check(share < 0.01, "little simultaneous battery charge/discharge",
              f"{100 * share:.2f}% of throughput", warn_only=True)
        st = n.stores[n.stores.carrier == BATT]
        if len(st):
            e = n.stores_t.e[st.index]
            check((e.min() >= -tol * scale).all(), "battery state of charge non-negative")
            e_nom = np.where(st.e_nom_extendable, st.e_nom_opt, st.e_nom)
            check((e.max() <= e_nom + tol * scale).all(), "battery state of charge <= capacity")


# --------------------------------------------------------------------------- #
# 3. metrics
# --------------------------------------------------------------------------- #
def compute_metrics(n):
    w = weights(n)
    d, _ = dist_interface(n)
    fw = d[~d.rev]
    rv = d[d.rev]
    p0 = n.links_t.p0

    lv = pd.Index(sorted(set(fw.lv)))
    imp = group_cols(p0[fw.index], fw.lv).reindex(columns=lv).fillna(0)  # HV -> LV, HV side
    cap_imp = fw.groupby("lv").cap.sum().reindex(lv)
    if len(rv):  # export carried by the reversed links
        exp = group_cols(p0[rv.index], rv.lv).reindex(columns=lv).fillna(0)
        cap_exp = rv.groupby("lv").cap.sum().reindex(lv).fillna(0)
    else:        # export = negative flow on the forward links
        net = imp.copy()
        imp, exp, cap_exp = net.clip(lower=0), (-net).clip(lower=0), cap_imp
    reinf_bus = fw[fw.index.map(is_reinf)].groupby("lv").cap.sum().reindex(lv).fillna(0)
    fixed_bus = fw.groupby("lv").p_nom_extendable.apply(lambda s: bool((~s).any())).reindex(lv)

    t = pd.DataFrame(index=lv)
    t["half"] = [bus_half(b) for b in lv]
    t["node"] = n.buses.location.reindex(lv).values
    t["dist_cap_MW"] = cap_imp
    t["export_cap_MW"] = cap_exp
    t["reinforcement_MW"] = reinf_bus
    t["capacity_fixed"] = fixed_bus

    t["peak_import_MW"] = imp.max()
    t["peak_export_MW"] = exp.max()
    t["import_GWh"] = imp.mul(w, axis=0).sum() / 1e3
    t["export_GWh"] = exp.mul(w, axis=0).sum() / 1e3

    wsum = w.sum()
    eps = 1e-3
    bind_imp = (imp >= (cap_imp - eps) * 0.999) & (cap_imp > eps)
    bind_exp = (exp >= (cap_exp - eps) * 0.999) & (cap_exp > eps)
    t["bind_import_pct"] = 100 * bind_imp.mul(w, axis=0).sum() / wsum
    t["bind_export_pct"] = 100 * bind_exp.mul(w, axis=0).sum() / wsum
    # binding hours are only meaningful where a capacity limit is fixed
    t.loc[~t.capacity_fixed, ["bind_import_pct", "bind_export_pct"]] = np.nan

    # demand on LV bus
    lp = load_power(n)
    lv_loads = n.loads[n.loads.bus.isin(lv)]
    dem = group_cols(lp[lv_loads.index], lv_loads.bus)
    t["demand_GWh"] = dem.mul(w, axis=0).sum().reindex(lv).fillna(0) / 1e3

    # rooftop PV
    g = n.generators[n.generators.carrier == ROOF]
    if len(g):
        gen = group_cols(n.generators_t.p[g.index], g.bus)
        avail = group_cols(dense(n, "Generator", "p_max_pu")[g.index] * g.p_nom_opt, g.bus)
        t["pv_potential_MW"] = g.groupby("bus").p_nom_max.sum().reindex(lv)
        t["pv_opt_MW"] = g.groupby("bus").p_nom_opt.sum().reindex(lv)
        t["pv_gen_GWh"] = gen.mul(w, axis=0).sum().reindex(lv) / 1e3
        t["pv_curtail_GWh"] = (avail - gen).mul(w, axis=0).sum().reindex(lv) / 1e3
    # batteries
    chg = n.links[n.links.carrier == CHG]
    dch = n.links[n.links.carrier == DCH]
    st = n.stores[n.stores.carrier == BATT]
    if len(chg):
        t["batt_P_MW"] = chg.groupby("bus0").p_nom_opt.sum().reindex(lv)
        b2lv = chg.set_index("bus1").bus0
        t["batt_E_MWh"] = (
            st.groupby(st.bus.map(b2lv)).e_nom_opt.sum().reindex(lv) if len(st) else np.nan
        )
        thr = group_cols(n.links_t.p0[chg.index], chg.bus0)
        t["batt_throughput_GWh"] = thr.mul(w, axis=0).sum().reindex(lv) / 1e3

    # prices
    if not n.buses_t.marginal_price.empty:
        mp = n.buses_t.marginal_price
        lvp = mp[lv]
        acp = mp[t.node.values]
        acp.columns = lv
        t["lv_price_mean"] = lvp.mul(w, axis=0).sum() / wsum
        t["lv_price_max"] = lvp.max()
        t["lv_ac_spread_mean"] = (lvp - acp).abs().mul(w, axis=0).sum() / wsum

    t = t.fillna({"pv_curtail_GWh": 0})
    t["pv_use_pct"] = 100 * t.get("pv_opt_MW", 0) / t.get("pv_potential_MW", np.nan)
    t["pv_curtail_pct"] = 100 * t.get("pv_curtail_GWh", 0) / (
        t.get("pv_gen_GWh", 0) + t.get("pv_curtail_GWh", 0)
    ).replace(0, np.nan)

    sum_cols = [
        "dist_cap_MW", "export_cap_MW", "reinforcement_MW", "peak_import_MW", "peak_export_MW",
        "import_GWh", "export_GWh", "demand_GWh", "pv_potential_MW", "pv_opt_MW",
        "pv_gen_GWh", "pv_curtail_GWh", "batt_P_MW", "batt_E_MWh", "batt_throughput_GWh",
    ]
    mean_cols = [
        "bind_import_pct", "bind_export_pct", "lv_price_mean", "lv_price_max", "lv_ac_spread_mean",
    ]
    agg = {c: "sum" for c in sum_cols if c in t}
    agg.update({c: "mean" for c in mean_cols if c in t})
    half = t.groupby("half").agg(agg)
    if {"pv_opt_MW", "pv_potential_MW"} <= set(half.columns):
        half["pv_use_pct"] = 100 * half.pv_opt_MW / half.pv_potential_MW
    if {"pv_curtail_GWh", "pv_gen_GWh"} <= set(half.columns):
        tot = half.pv_gen_GWh + half.pv_curtail_GWh
        half["pv_curtail_pct"] = 100 * half.pv_curtail_GWh / tot.replace(0, np.nan)
    return t, half


def system_costs(n):
    try:
        s = n.statistics
        capex = s.capex().groupby(level=-1).sum()
        opex = s.opex().groupby(level=-1).sum()
        df = pd.DataFrame({"capex": capex, "opex": opex}).fillna(0)
        df["total"] = df.capex + df.opex
        return df.sort_values("total", ascending=False)
    except Exception as e:  # statistics API differs between versions
        print(f"[skip] cost breakdown unavailable ({e})")
        return None


def totals(n, half):
    out = {"objective": getattr(n, "objective", np.nan)}
    for c in ["pv_opt_MW", "pv_curtail_GWh", "batt_E_MWh", "batt_P_MW",
              "import_GWh", "export_GWh", "dist_cap_MW", "reinforcement_MW"]:
        if c in half:
            out[c] = half[c].sum()
    return pd.Series(out)


# --------------------------------------------------------------------------- #
# 4. plots
# --------------------------------------------------------------------------- #
def plot_flow_duration(n, bus_table, out):
    d, _ = dist_interface(n)
    fw, rv = d[~d.rev], d[d.rev]
    imp = group_cols(n.links_t.p0[fw.index], fw.lv)
    net = imp - group_cols(n.links_t.p0[rv.index], rv.lv).reindex(columns=imp.columns).fillna(0) if len(rv) else imp
    cap = bus_table.dist_cap_MW.replace(0, np.nan)
    norm = net[cap.index] / cap
    fig, ax = plt.subplots(figsize=(8, 5))
    for b in norm.columns:
        h = bus_half(b)
        s = norm[b].sort_values(ascending=False).values
        ax.plot(np.linspace(0, 100, len(s)), s, color=COLORS[h], alpha=0.5, lw=1)
    for h in sorted(set(bus_table.half)):
        ax.plot([], [], color=COLORS[h], label=f"LV {h}")
    ax.axhline(1, color="k", ls="--", lw=0.8)
    ax.axhline(-1, color="k", ls="--", lw=0.8)
    ax.set(xlabel="% of time", ylabel="net HV-LV flow / capacity  (+import, -export)",
           title="Distribution link flow duration (1.0 = limit)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "01_flow_duration.png", dpi=150)
    plt.close(fig)


def plot_price_duration(n, bus_table, out):
    if n.buses_t.marginal_price.empty:
        return
    mp = n.buses_t.marginal_price
    fig, ax = plt.subplots(figsize=(8, 5))
    ac = mp[bus_table.node.unique()].mean(axis=1).sort_values(ascending=False).values
    ax.plot(np.linspace(0, 100, len(ac)), ac, color="k", label="AC (mean over nodes)")
    for h in sorted(set(bus_table.half)):
        cols = bus_table.index[bus_table.half == h]
        s = mp[cols].mean(axis=1).sort_values(ascending=False).values
        ax.plot(np.linspace(0, 100, len(s)), s, color=COLORS[h], label=f"LV {h}")
    ax.set(xlabel="% of time", ylabel="Marginal price [EUR/MWh]",
           title="Price duration curves: HV vs LV halves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "02_price_duration.png", dpi=150)
    plt.close(fig)


def plot_pv(half, out):
    if "pv_opt_MW" not in half:
        return
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(half))
    axs[0].bar(x - 0.2, half.pv_potential_MW, 0.4, label="potential", color="lightgray")
    axs[0].bar(x + 0.2, half.pv_opt_MW, 0.4, label="installed", color="gold")
    axs[0].set_xticks(x, half.index)
    axs[0].set(ylabel="MW", title="Rooftop PV: potential vs installed")
    axs[0].legend()
    axs[1].bar(x - 0.2, half.pv_gen_GWh, 0.4, label="used", color="gold")
    axs[1].bar(x + 0.2, half.pv_curtail_GWh, 0.4, label="curtailed", color="tab:red")
    axs[1].set_xticks(x, half.index)
    axs[1].set(ylabel="GWh", title="Rooftop PV energy")
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(out / "03_rooftop_pv.png", dpi=150)
    plt.close(fig)


def plot_battery(half, out):
    if "batt_E_MWh" not in half:
        return
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    x = np.arange(len(half))
    axs[0].bar(x, half.batt_E_MWh, color=[COLORS[h] for h in half.index])
    axs[0].set_xticks(x, half.index)
    axs[0].set(ylabel="MWh", title="Home battery energy capacity")
    axs[1].bar(x, half.batt_P_MW, color=[COLORS[h] for h in half.index])
    axs[1].set_xticks(x, half.index)
    axs[1].set(ylabel="MW", title="Home battery power capacity")
    fig.tight_layout()
    fig.savefig(out / "04_battery.png", dpi=150)
    plt.close(fig)


def plot_flows_and_binding(half, out):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(half))
    axs[0].bar(x - 0.2, half.import_GWh, 0.4, label="import", color="tab:green")
    axs[0].bar(x + 0.2, half.export_GWh, 0.4, label="export", color="tab:purple")
    axs[0].set_xticks(x, half.index)
    axs[0].set(ylabel="GWh", title="HV-LV energy exchange")
    axs[0].legend()
    if "bind_import_pct" in half and half.bind_import_pct.notna().any():
        axs[1].bar(x - 0.2, half.bind_import_pct, 0.4, label="import at limit", color="tab:green")
        axs[1].bar(x + 0.2, half.bind_export_pct, 0.4, label="export at limit", color="tab:purple")
        axs[1].legend()
    else:
        axs[1].text(0.5, 0.5, "capacity not fixed\n(unconstrained run)", ha="center", va="center")
    axs[1].set_xticks(x, half.index)
    axs[1].set(ylabel="% of time", title="Time with distribution link binding")
    fig.tight_layout()
    fig.savefig(out / "05_exchange_and_binding.png", dpi=150)
    plt.close(fig)


def plot_dispatch_week(n, bus_table, out):
    g = n.generators[n.generators.carrier == ROOF]
    if g.empty:
        return
    node_pv = bus_table.groupby("node").pv_opt_MW.sum() if "pv_opt_MW" in bus_table else None
    node = node_pv.idxmax() if node_pv is not None and node_pv.max() > 0 else bus_table.node.iloc[0]
    hrs = float(weights(n).mean())
    npts = max(2, int(168 / hrs))
    d, _ = dist_interface(n)
    fw, rv = d[~d.rev], d[d.rev]
    chg = n.links[n.links.carrier == CHG]
    dch = n.links[n.links.carrier == DCH]
    lp = load_power(n)

    halves = [b for b in bus_table.index[bus_table.node == node]]
    def lv_flows(b):  # LV-side import / export in MW
        imp = -n.links_t.p1[fw.index[fw.lv == b]].sum(axis=1)
        if len(rv):
            return imp.clip(lower=0), n.links_t.p0[rv.index[rv.lv == b]].sum(axis=1).clip(lower=0)
        return imp.clip(lower=0), (-imp).clip(lower=0)

    flows = {b: lv_flows(b) for b in halves}
    coarse = hrs > 6  # e.g. 168H: a "week" is one point, show the full horizon
    if coarse:
        sl = slice(None)
    else:
        pos = int(np.argmax(sum(f[0].values for f in flows.values())))
        start = int(np.clip(pos - npts // 2, 0, max(0, len(n.snapshots) - npts)))
        sl = slice(start, start + npts)
    x = time_axis(n)[sl]

    fig, axs = plt.subplots(len(halves), 1, figsize=(11, 3.6 * len(halves)), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, b in zip(axs, halves):
        pv = n.generators_t.p[g.index[g.bus == b]].sum(axis=1)
        load = lp[n.loads.index[n.loads.bus == b]].sum(axis=1)
        cc = n.links_t.p0[chg.index[chg.bus0 == b]].sum(axis=1)
        dc = -n.links_t.p1[dch.index[dch.bus1 == b]].sum(axis=1)
        fimp, fexp = flows[b]
        ax.stackplot(x, pv.values[sl], dc.values[sl], fimp.values[sl],
                     labels=["rooftop PV", "battery discharge", "import from HV"],
                     colors=["gold", "tab:cyan", "tab:green"], alpha=0.85)
        ax.stackplot(x, -load.values[sl], -cc.values[sl], -fexp.values[sl],
                     labels=["load", "battery charge", "export to HV"],
                     colors=["tab:gray", "tab:blue", "tab:purple"], alpha=0.85)
        ax.axhline(0, color="k", lw=0.6)
        ax.set(ylabel="MW", title=f"{b}")
        ax.legend(ncol=3, fontsize=8, loc="upper right")
    fig.suptitle(f"Dispatch at node {node} ("
                 + ("full horizon, coarse resolution" if coarse else "week around peak LV import") + ")")
    fig.tight_layout()
    fig.savefig(out / "06_dispatch_week.png", dpi=150)
    plt.close(fig)


def plot_baseline(tot_run, tot_base, out):
    keys = [k for k in tot_run.index if k != "objective" and k in tot_base.index]
    keys = [k for k in keys if max(abs(tot_run[k]), abs(tot_base[k])) > 0]
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [tot_base[k] for k in keys], 0.4, label="baseline", color="lightgray")
    ax.bar(x + 0.2, [tot_run[k] for k in keys], 0.4, label="this run", color="tab:blue")
    ax.set_xticks(x, keys, rotation=30, ha="right")
    ax.set(title="Baseline vs this run (system totals)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "07_baseline_comparison.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. flexibility: EV / heat pumps / tanks / home battery
# --------------------------------------------------------------------------- #
EV_LOAD, BEVC, V2G, EVST = "land transport EV", "BEV charger", "V2G", "EV battery"
HVX = "HV exchange"


def lv_bus_set(n):
    return set(n.buses.index[n.buses.carrier == "low voltage"])


def _bus_cols(df):
    return [c for c in df.columns if c == "bus" or (c.startswith("bus") and c[3:].isdigit())]


def comp_half(df):
    """'a' / 'b' from the first bus name ending in ' a' / ' b'; 'all' if none (unsplit)."""
    cols = _bus_cols(df)
    out = pd.Series("all", index=df.index)
    for c in reversed(cols):  # the first column wins
        s = df[c].astype(str)
        m = s.str.endswith((" a", " b"))
        out[m] = s[m].str[-1]
    return out


def mixed_halves(df):
    """Components whose buses sit on different halves (should never happen)."""
    cols = _bus_cols(df)
    if not cols or df.empty:
        return []
    letters = pd.concat(
        [df[c].astype(str).where(df[c].astype(str).str.endswith((" a", " b"))).str[-1] for c in cols],
        axis=1,
    )
    return list(df.index[letters.nunique(axis=1) > 1])


def lv_port_bus(links, lv):
    out = pd.Series("", index=links.index)
    for c in reversed(_bus_cols(links)):
        m = links[c].isin(lv)
        out[m] = links.loc[m, c]
    return out


def port_power(n, links, buses):
    """Bus-side power (MW, + = withdrawn from the bus) of `links` at ports attached to `buses`."""
    links = list(links)
    out = pd.DataFrame(0.0, index=n.snapshots, columns=links)
    if not links:
        return out
    for col in [c for c in n.links.columns if c.startswith("bus") and c[3:].isdigit()]:
        i = col[3:]
        if f"p{i}" not in n.links_t:
            continue
        m = n.links.loc[links, col].isin(buses)
        sel = list(m.index[m])
        if sel:
            out[sel] = out[sel] + n.links_t[f"p{i}"].reindex(columns=sel).fillna(0)
    return out


def optcap(df, nom):
    if df.empty:
        return pd.Series(dtype=float)
    return pd.Series(np.where(df[f"{nom}_extendable"], df[f"{nom}_opt"], df[nom]), index=df.index)


def by_half(vals, half):
    if len(vals) == 0:
        return pd.Series(dtype=float)
    return vals.groupby(half.reindex(vals.index)).sum()


def flex_metrics(n):
    """Per-half metrics of EV charging / V2G, heat pumps, resistive heaters and water tanks."""
    w = weights(n)
    lv = lv_bus_set(n)
    L, K, S = n.loads, n.links, n.stores
    lh, kh, sh = comp_half(L), comp_half(K), comp_half(S)
    lp = load_power(n)
    p0, p1 = n.links_t.p0, n.links_t.p1
    d, series = {}, {}

    def put(name, s):
        for h, v in s.items():
            d.setdefault(name, {})[h] = float(v)

    # ---- EV ----
    evl = list(L.index[L.carrier == EV_LOAD])
    chg = list(K.index[K.carrier == BEVC])
    v2g = list(K.index[K.carrier == V2G])
    evs = S.index[S.carrier == EVST]
    if evl or chg:
        put("ev_load_GWh", by_half(lp[evl].mul(w, axis=0).sum(), lh) / 1e3)
        put("ev_charger_P_MW", by_half(optcap(K.loc[chg], "p_nom"), kh))
        put("ev_v2g_P_MW", by_half(optcap(K.loc[v2g], "p_nom"), kh))
        put("ev_store_MWh", by_half(optcap(S.loc[evs], "e_nom"), sh))
        for h in sorted(set(kh[chg])):
            c_h = [x for x in chg if kh[x] == h]
            v_h = [x for x in v2g if kh[x] == h]
            l_h = [x for x in evl if lh[x] == h]
            charge = p0[c_h].sum(axis=1)
            sup = (-p1[v_h]).clip(lower=0).sum(axis=1) if v_h else 0 * charge
            net = charge - sup                       # net EV draw from the LV bus
            eff = float(K.loc[c_h, "efficiency"].mean())
            inflex = lp[l_h].sum(axis=1) / eff       # same energy, charged when it is driven
            e_inflex = (inflex * w).sum() / 1e3
            shift = 0.5 * ((net - inflex).abs() * w).sum() / 1e3
            d.setdefault("ev_charge_GWh", {})[h] = float((charge * w).sum() / 1e3)
            d.setdefault("ev_v2g_GWh", {})[h] = float((sup * w).sum() / 1e3)
            d.setdefault("ev_shift_GWh", {})[h] = float(shift)
            d.setdefault("ev_shift_pct", {})[h] = float(100 * shift / e_inflex) if e_inflex > 0 else np.nan
            d.setdefault("ev_peak_net_MW", {})[h] = float(net.max())
            d.setdefault("ev_peak_inflex_MW", {})[h] = float(inflex.max())
            series[("EV net draw", h)] = net
            series[("EV inflexible profile", h)] = inflex

    # ---- heat pumps / resistive heaters ----
    for name, pat in (("hp", "heat pump"), ("rh", "resistive heater")):
        idx = list(K.index[K.carrier.str.contains(pat)])
        if not idx:
            continue
        P = port_power(n, idx, lv).clip(lower=0)
        put(f"{name}_P_MW", by_half(optcap(K.loc[idx], "p_nom"), kh))
        for h in sorted(set(kh[idx])):
            s = P[[x for x in idx if kh[x] == h]].sum(axis=1)
            d.setdefault(f"{name}_elec_GWh", {})[h] = float((s * w).sum() / 1e3)
            d.setdefault(f"{name}_peak_MW", {})[h] = float(s.max())
            d.setdefault(f"{name}_peak_to_mean", {})[h] = float(s.max() / s.mean()) if s.mean() > 0 else np.nan
            series[({"hp": "Heat pumps", "rh": "Resistive heaters"}[name], h)] = s

    # ---- decentral heat storage and heat demand ----
    tk = S.index[S.carrier.str.contains("water tank")]
    if len(tk):
        put("tank_E_MWh", by_half(optcap(S.loc[tk], "e_nom"), sh))
        dis = n.stores_t.p.reindex(columns=tk).fillna(0).clip(lower=0).mul(w, axis=0).sum() / 1e3
        put("tank_discharge_GWh", by_half(dis, sh))
    hl = list(L.index[L.carrier.str.contains("heat")])
    if hl:
        put("heat_demand_GWh", by_half(lp[hl].mul(w, axis=0).sum(), lh) / 1e3)

    fm = pd.DataFrame(d)
    return fm, series


FLEX_SUM_COLS = [
    "ev_load_GWh", "ev_charger_P_MW", "ev_v2g_P_MW", "ev_store_MWh", "ev_charge_GWh", "ev_v2g_GWh",
    "ev_shift_GWh", "ev_peak_net_MW", "ev_peak_inflex_MW", "hp_P_MW", "hp_elec_GWh", "hp_peak_MW",
    "rh_P_MW", "rh_elec_GWh", "rh_peak_MW", "tank_E_MWh", "tank_discharge_GWh", "heat_demand_GWh",
]


def flex_totals(fm):
    return pd.Series({c: fm[c].sum() for c in FLEX_SUM_COLS if c in fm})


def lv_decomposition(n):
    """Contribution of every technology at the LV buses (+ = consumes, - = supplies), per half."""
    lv = lv_bus_set(n)
    lp = load_power(n)
    parts = {}

    def add(df, half_s, label_s, sign=1.0):
        if df.shape[1] == 0:
            return
        key = pd.Series(df.columns, index=df.columns)
        for (h, lab), cols in key.groupby([half_s.reindex(df.columns), label_s.reindex(df.columns)]):
            parts[(h, lab)] = parts.get((h, lab), 0) + sign * df[list(cols.values)].sum(axis=1)

    Ld = n.loads[n.loads.bus.isin(lv)]
    add(lp[Ld.index], Ld.bus.map(bus_half), Ld.carrier)
    Gd = n.generators[n.generators.bus.isin(lv)]
    add(n.generators_t.p.reindex(columns=Gd.index).fillna(0), Gd.bus.map(bus_half), Gd.carrier, -1.0)
    touch = n.links[n.links[[c for c in n.links.columns if c.startswith("bus") and c[3:].isdigit()]]
                    .isin(lv).any(axis=1)]
    P = port_power(n, touch.index, lv)
    pb = lv_port_bus(touch, lv)
    lab = touch.carrier.where(touch.carrier != DIST, HVX)
    add(P, pb.map(bus_half), lab)
    D = pd.DataFrame(parts)
    D.columns = pd.MultiIndex.from_tuples(D.columns)
    return D.sort_index(axis=1)


def peak_tables(D):
    """Technology breakdown at the hour of maximum LV import and maximum LV export, per half."""
    cols, times = {}, {}
    for h in sorted(set(D.columns.get_level_values(0))):
        Dh = D[h]
        if HVX not in Dh:
            continue
        imp = -Dh[HVX]
        for kind, t in (("import peak", imp.idxmax()), ("export peak", imp.idxmin())):
            cols[(h, kind)] = Dh.loc[t]
            times[(h, kind)] = t
    T = pd.DataFrame(cols)
    T.columns = pd.MultiIndex.from_tuples(T.columns)
    return T, times


def plot_peak_decomposition(T, times, out):
    if T.empty:
        return
    T = T.loc[(T.abs() > 1e-6).any(axis=1)]
    fig, ax = plt.subplots(figsize=(max(7, 2.2 * T.shape[1]), 5.5))
    x = np.arange(T.shape[1])
    cmap = plt.get_cmap("tab20")
    bp, bn = np.zeros(len(x)), np.zeros(len(x))
    for i, lab in enumerate(T.index):
        v = T.loc[lab].values.astype(float)
        pos, neg = np.clip(v, 0, None), np.clip(v, None, 0)
        ax.bar(x, pos, bottom=bp, color=cmap(i % 20), label=lab, width=0.6)
        ax.bar(x, neg, bottom=bn, color=cmap(i % 20), width=0.6)
        bp, bn = bp + pos, bn + neg
    ax.axhline(0, color="k", lw=0.6)
    ax.margins(y=0.08)
    ax.set_xticks(x, [f"{h}\n{k}" for h, k in T.columns])
    ax.set(ylabel="MW at the LV buses (+ consumption, - supply)",
           title="What drives the LV peaks (summed over nodes of each half)")
    ax.legend(fontsize=7, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(out / "08_lv_peak_decomposition.png", dpi=150)
    plt.close(fig)


def plot_flex_capacity(fm, out):
    cols = [c for c in ["ev_charger_P_MW", "ev_v2g_P_MW", "hp_P_MW", "rh_P_MW"] if c in fm]
    cols2 = [c for c in ["ev_store_MWh", "tank_E_MWh"] if c in fm]
    if not cols and not cols2:
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, cs, ttl, yl in ((axs[0], cols, "Flexible power capacity", "MW"),
                            (axs[1], cols2, "Flexible energy capacity", "MWh")):
        x = np.arange(len(fm))
        k = max(len(cs), 1)
        for i, c in enumerate(cs):
            ax.bar(x + (i - (k - 1) / 2) * 0.8 / k, fm[c].fillna(0), 0.8 / k, label=c)
        ax.set_xticks(x, fm.index)
        ax.set(title=ttl, ylabel=yl)
        if cs:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "09_flex_capacity.png", dpi=150)
    plt.close(fig)


def plot_flex_activity(fm, out):
    cols = [c for c in ["ev_shift_GWh", "ev_v2g_GWh", "tank_discharge_GWh"] if c in fm]
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(fm))
    k = len(cols)
    for i, c in enumerate(cols):
        ax.bar(x + (i - (k - 1) / 2) * 0.8 / k, fm[c].fillna(0), 0.8 / k, label=c)
    ax.set_xticks(x, fm.index)
    ax.set(ylabel="GWh", title="Flexibility actually used (energy shifted in time)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "10_flex_activity.png", dpi=150)
    plt.close(fig)


def plot_mean_day(n, series, out):
    if float(weights(n).mean()) > 6 or not series:
        return
    hours = time_axis(n).hour
    halves = sorted({h for _, h in series})
    fig, axs = plt.subplots(len(halves), 1, figsize=(8, 3.4 * len(halves)), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, h in zip(axs, halves):
        for (lab, hh), s in series.items():
            if hh == h:
                ax.plot(s.groupby(hours).mean().index, s.groupby(hours).mean().values,
                        label=lab, ls="--" if "inflexible" in lab else "-")
        ax.set(ylabel="MW (mean over horizon)", title=f"Mean daily profile, half {h}")
        ax.legend(fontsize=8)
    axs[-1].set_xlabel("hour of day")
    fig.tight_layout()
    fig.savefig(out / "11_mean_day_profiles.png", dpi=150)
    plt.close(fig)


def flex_checks(n, tol):
    print("\n=== 5. FLEXIBILITY CHECKS ===")
    mix = {}
    for name, df in (("links", n.links), ("loads", n.loads), ("stores", n.stores),
                     ("generators", n.generators)):
        bad = mixed_halves(df)
        if bad:
            mix[name] = bad[:3]
    check(not mix, "no component connects buses of different halves (a vs b)",
          "; ".join(f"{k}: {v}" for k, v in mix.items()))
    lv = lv_bus_set(n)
    if lv:
        D = lv_decomposition(n)
        res = pd.Series({h: D[h].sum(axis=1).abs().max() for h in set(D.columns.get_level_values(0))})
        scale = max(1.0, 1e-2 * float(load_power(n).sum(axis=1).max()))
        check(res.max() <= tol * scale, "LV decomposition closes at every LV bus (all technologies accounted for)",
              f"max residual {res.max():.2e} MW")
    # EV / heat present in both halves?
    for lab, mask in (("EV chain", n.links.carrier == BEVC),
                      ("heat pumps", n.links.carrier.str.contains("heat pump"))):
        hs = set(comp_half(n.links[mask]))
        if mask.any() and {"a", "b"} & hs:
            check({"a", "b"} <= hs, f"{lab} present on both halves", f"halves found: {sorted(hs)}")


def adjusted_objective(n, cost_mw):
    """Objective with the sunk (fixed) existing LV grid priced at `cost_mw` EUR/MW."""
    d, _ = dist_interface(n)
    fr = d[(~d.rev) & ~d.index.map(is_reinf) & ~d.p_nom_extendable]
    return float(n.objective + cost_mw * fr.cap.sum()), float(fr.cap.sum())


def baseline_cost_mw(nb):
    d, _ = dist_interface(nb)
    fb = d[(~d.rev) & ~d.index.map(is_reinf)]
    return float((fb.capital_cost * fb.cap).sum() / fb.cap.sum()) if fb.cap.sum() > 0 else 0.0


def value_of_flexibility(n_on, n_off, cost_mw, out):
    print("\n=== VALUE OF FLEXIBILITY (flex-on vs flex-off, same distribution settings) ===")
    a_on, c_on = adjusted_objective(n_on, cost_mw)
    a_off, c_off = adjusted_objective(n_off, cost_mw)
    same = abs(c_on - c_off) <= 1e-3 * max(c_on, c_off, 1.0)
    _, h_on = compute_metrics(n_on)
    _, h_off = compute_metrics(n_off)
    f_on, f_off = flex_totals(flex_metrics(n_on)[0]), flex_totals(flex_metrics(n_off)[0])
    rows = {
        "objective (adjusted)": (a_on, a_off),
        "existing LV grid MW": (c_on, c_off),
        "reinforcement MW": (h_on.reinforcement_MW.sum(), h_off.reinforcement_MW.sum()),
        "rooftop PV curtailed GWh": (h_on.get("pv_curtail_GWh", pd.Series(0)).sum(),
                                     h_off.get("pv_curtail_GWh", pd.Series(0)).sum()),
        "home battery MWh": (h_on.get("batt_E_MWh", pd.Series(0)).sum(), h_off.get("batt_E_MWh", pd.Series(0)).sum()),
    }
    for k in ("ev_shift_GWh", "ev_v2g_GWh", "tank_E_MWh", "tank_discharge_GWh"):
        rows[k] = (f_on.get(k, 0.0), f_off.get(k, 0.0))
    df = pd.DataFrame(rows, index=["flex_on", "flex_off"]).T
    df["off_minus_on"] = df.flex_off - df.flex_on
    print(df)
    value = a_off - a_on
    print(f"\nValue of flexibility here: {value:,.0f} EUR  ({100 * value / abs(a_off):.2f}% of the flex-off cost)")
    if same:
        check(value > -1e-4 * abs(a_off), "flex-off is never cheaper than flex-on (same existing grid)",
              f"value {value:,.0f} EUR")
    else:
        info("existing grids differ between flex-on and flex-off",
             f"{c_on:,.0f} vs {c_off:,.0f} MW: the headroom basis moved with the loads, so compare with care")
    df.to_csv(out / "value_of_flexibility.csv")
    return value


# --------------------------------------------------------------------------- #
# 6. comparing a set of runs (headroom sweep, asymmetric cases, ...)
# --------------------------------------------------------------------------- #
def run_summary(n, cost_mw=None, obj_base=None):
    bus_t, half = compute_metrics(n)
    fm, _ = flex_metrics(n)
    ft = flex_totals(fm)
    tot = totals(n, half)
    r = {"objective": tot["objective"]}
    if cost_mw is not None:
        adj, existing = adjusted_objective(n, cost_mw)
        r["existing_grid_MW"] = existing
        r["objective_adjusted"] = adj
        if obj_base:
            r["adjusted_vs_baseline_pct"] = 100 * (adj - obj_base) / abs(obj_base)
    else:
        r["existing_grid_MW"] = half.dist_cap_MW.sum() - half.reinforcement_MW.sum()
    r["reinforcement_MW"] = half.reinforcement_MW.sum()
    r["total_grid_MW"] = half.dist_cap_MW.sum()
    for c in ("pv_opt_MW", "pv_curtail_GWh", "batt_P_MW", "batt_E_MWh", "peak_import_MW", "peak_export_MW"):
        if c in half:
            r[c] = half[c].sum()
    for c in ("bind_import_pct", "bind_export_pct"):
        if c in half and half[c].notna().any():
            r[c + "_max"] = half[c].max()
    for k in ("ev_shift_GWh", "ev_v2g_GWh", "ev_v2g_P_MW", "hp_peak_MW", "tank_E_MWh", "tank_discharge_GWh"):
        if k in ft:
            r[k] = ft[k]
    return r


def run_compare(args):
    nb = None
    cost_mw = obj_base = None
    if args.baseline or args.baseline_run:
        nb, _ = load_solved(args.baseline or args.network, args.baseline_run, args.root)
        cost_mw, obj_base = baseline_cost_mw(nb), float(nb.objective)
        print(f"baseline objective {obj_base:,.0f}; distribution capex {cost_mw:,.0f} EUR/MW")
    rows, missing = {}, []
    for run in args.compare:
        print(f"\n--- {run}")
        n, _ = load_solved(args.network, run, args.root, fatal=False)
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
    print("\n=== RUN COMPARISON ===")
    print(df.T)
    df.to_csv(out / "compare_runs.csv")
    plot_compare(df, out)
    print(f"Saved to {out.resolve()}")
    return 0


def plot_compare(df, out):
    x = df.existing_grid_MW if "existing_grid_MW" in df else pd.Series(range(len(df)), index=df.index)
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    ax = axs[0, 0]
    if "adjusted_vs_baseline_pct" in df:
        ax.plot(x, df.adjusted_vs_baseline_pct, "o-")
        ax.axhline(0, color="k", lw=0.6)
    ax.set(title="System cost vs baseline (existing grid priced in)", ylabel="%")
    ax = axs[0, 1]
    ax.bar(range(len(df)), df.reinforcement_MW)
    ax.set_xticks(range(len(df)), [f"{v:,.0f}" for v in x], rotation=45)
    ax.set(title="Reinforcement bought", ylabel="MW")
    ax = axs[1, 0]
    for c in ("batt_E_MWh", "tank_E_MWh"):
        if c in df:
            ax.plot(x, df[c], "o-", label=c)
    ax.legend(fontsize=8)
    ax.set(title="Storage energy capacity", ylabel="MWh")
    ax = axs[1, 1]
    for c in ("pv_curtail_GWh", "ev_shift_GWh", "ev_v2g_GWh", "tank_discharge_GWh"):
        if c in df:
            ax.plot(x, df[c], "o-", label=c)
    ax.legend(fontsize=8)
    ax.set(title="Curtailment and flexibility used", ylabel="GWh")
    for a in axs.flat:
        if a is not axs[0, 1]:
            a.set_xlabel("existing LV grid [MW]")
    fig.tight_layout()
    fig.savefig(out / "12_run_comparison.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--network", required=True,
                    help="path to solved .nc OR a name/pattern, e.g. base_s_3_lv1.2_Co2L0-168H_2030")
    ap.add_argument("--run", help="run name (folder under results/) to search in")
    ap.add_argument("--baseline", help="baseline .nc path or name/pattern (default: same name as --network)")
    ap.add_argument("--baseline-run", help="run name of the baseline (searched under results/)")
    ap.add_argument("--root", default="results", help="where to search (default: results)")
    ap.add_argument("--expect-constrained", action="store_true",
                    help="FAIL if no distribution link is fixed (headroom silently not applied)")
    ap.add_argument("--expect-reinforcement", action="store_true",
                    help="FAIL if the run has no reinforcement links")
    ap.add_argument("--flexoff-run", help="run with flexibility switched off (same distribution settings): "
                    "reports the value of flexibility")
    ap.add_argument("--compare", nargs="+", metavar="RUN",
                    help="compare several runs (e.g. a headroom sweep) in one table and plot; "
                    "runs without a solved network are listed as missing")
    ap.add_argument("--out", default="analysis_split_lv")
    ap.add_argument("--tol", type=float, default=1e-3, help="relative tolerance for checks")
    ap.add_argument("--expect-equal", action="store_true",
                    help="require objective == baseline (split (1,1), no headroom test)")
    args = ap.parse_args()

    if args.compare:
        return run_compare(args)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n, net_path = load_solved(args.network, args.run, args.root)
    print(f"  {len(n.buses)} buses, {len(n.links)} links, {len(n.snapshots)} snapshots, "
          f"mean resolution {weights(n).mean():.1f} h")

    hrs = float(weights(n).mean())
    months = sorted(set(time_axis(n).month))
    info("snapshot window", f"{time_axis(n)[0]:%Y-%m-%d} to {time_axis(n)[-1]:%Y-%m-%d}, months {months}")
    if not set(months) & {5, 6, 7, 8}:
        info("no summer months in the horizon",
             "PV-export limits cannot bind here; add a summer window before drawing export conclusions")
    if hrs > 6:
        check(False, "temporal resolution too coarse for peak-based limits",
              f"mean resolution {hrs:.0f} h: peaks are averaged out, so headroom limits "
              "and binding hours are not representative (use e.g. 3H)", warn_only=True)

    structural_checks(n)
    d_chk, _ = dist_interface(n)
    fwd_chk = d_chk[(~d_chk.rev) & ~d_chk.index.map(is_reinf)]
    if args.expect_constrained:
        check(bool((~fwd_chk.p_nom_extendable).any()),
              "headroom constraint is applied (fixed distribution links exist)",
              "no fixed links: check the headroom key in config.yaml, the prepare log "
              "('fixed at ... MW' lines) and results/<run>/config")
    if args.expect_reinforcement:
        check(bool(d_chk.index.map(is_reinf).any()), "reinforcement links exist",
              "reinforce flag was not applied")
    solution_checks(n, args.tol)

    print("\n=== 3. METRICS ===")
    bus_table, half = compute_metrics(n)
    pd.set_option("display.width", 200, "display.max_columns", 50, "display.float_format", "{:,.2f}".format)
    print("\nPer LV half:")
    print(half.T)
    bus_table.to_csv(out / "metrics_per_lv_bus.csv")
    half.to_csv(out / "metrics_per_half.csv")

    costs = system_costs(n)
    if costs is not None:
        costs.to_csv(out / "system_costs_by_carrier.csv")
        print("\nTop cost components (EUR):")
        print(costs.head(8))

    fm, flex_series = flex_metrics(n)
    ftot = flex_totals(fm)
    tot = pd.concat([totals(n, half), ftot])
    tot.to_csv(out / "totals.csv")
    if not fm.empty:
        print("\nFlexible assets per LV half (EV / heat / tanks):")
        print(fm.T)
        fm.to_csv(out / "metrics_flex_per_half.csv")

    # does a fixed limit actually bind anywhere?
    if "bind_import_pct" in half and half.bind_import_pct.notna().any():
        mx = max(half.bind_import_pct.max(), half.bind_export_pct.max())
        check(mx > 0.5, "headroom limit binds somewhere (>0.5% of time)",
              f"max time at limit {mx:.2f}% -> limits too loose to influence dispatch" if mx <= 0.5
              else f"max time at limit {mx:.1f}%", warn_only=True)

    # asymmetry info
    if {"a", "b"} <= set(half.index):
        for c in ["pv_opt_MW", "pv_curtail_GWh", "batt_E_MWh", "import_GWh", "export_GWh"]:
            if c in half:
                info(f"a vs b: {c}", f"a={half.at['a', c]:.2f}  b={half.at['b', c]:.2f}")

    # baseline comparison
    if args.baseline or args.baseline_run:
        print("\n=== BASELINE COMPARISON ===")
        nb, _ = load_solved(args.baseline or args.network, args.baseline_run, args.root)
        _, half_b = compute_metrics(nb)
        tot_b = pd.concat([totals(nb, half_b), flex_totals(flex_metrics(nb)[0])])
        cmp_df = pd.DataFrame({"baseline": tot_b, "this_run": tot})
        cmp_df["delta"] = cmp_df.this_run - cmp_df.baseline
        cmp_df["delta_pct"] = 100 * cmp_df.delta / cmp_df.baseline.abs().replace(0, np.nan)
        print(cmp_df)
        cmp_df.to_csv(out / "baseline_comparison.csv")
        rel = (tot["objective"] - tot_b["objective"]) / abs(tot_b["objective"])
        if args.expect_equal:
            check(abs(rel) < 1e-4, "objective equals baseline (split is neutral)",
                  f"rel. difference {rel:.2e}")
        else:
            info("objective vs baseline", f"rel. difference {rel:+.3e} (raw; fixed grid is free)")
            # price the fixed existing grid at the baseline's distribution capex so the
            # objectives become comparable (constant term, does not change the optimum)
            cost_mw = baseline_cost_mw(nb)
            adj, fixed_mw = adjusted_objective(n, cost_mw)
            if fixed_mw > 0:
                d_b, _ = dist_interface(nb)
                base_opt = d_b[(~d_b.rev) & ~d_b.index.map(is_reinf)].cap.sum()
                rel_adj = (adj - tot_b["objective"]) / abs(tot_b["objective"])
                info("adjusted objective (fixed grid priced at baseline capex)",
                     f"{adj:,.0f}  ({rel_adj:+.2%} vs baseline; existing grid {fixed_mw:,.0f} MW "
                     f"vs baseline-optimal {base_opt:,.0f} MW at {cost_mw:,.0f} EUR/MW)")
                check(rel_adj > -1e-4, "adjusted objective >= baseline (constraint cannot make the system cheaper)",
                      f"{rel_adj:+.3%}", warn_only=False)
        plot_baseline(tot, tot_b, out)

    flex_checks(n, args.tol)
    if not fm.empty:
        T, times = peak_tables(lv_decomposition(n))
        if not T.empty:
            print("\nLV peak decomposition (MW, + consumption / - supply):")
            print(T)
            T.to_csv(out / "lv_peak_decomposition.csv")
            for (h, kind), t in times.items():
                info(f"{kind} half {h}", f"at {t}")

    if args.flexoff_run:
        nf, _ = load_solved(args.network, args.flexoff_run, args.root, fatal=False)
        if nf is None:
            info("flex-off run has no solved network",
                 "if it is infeasible, that itself shows the grid limit cannot be met without flexibility")
        else:
            cm = baseline_cost_mw(nb) if (args.baseline or args.baseline_run) else 0.0
            value_of_flexibility(n, nf, cm, out)

    print("\n=== 4. PLOTS ===")
    plot_flow_duration(n, bus_table, out)
    plot_price_duration(n, bus_table, out)
    plot_pv(half, out)
    plot_battery(half, out)
    plot_flows_and_binding(half, out)
    plot_dispatch_week(n, bus_table, out)
    if not fm.empty:
        plot_flex_capacity(fm, out)
        plot_flex_activity(fm, out)
        plot_mean_day(n, flex_series, out)
        if not T.empty:
            plot_peak_decomposition(T, times, out)
    print(f"Saved to {out.resolve()}")

    res = pd.DataFrame(RESULTS)
    res.to_csv(out / "checks.csv", index=False)
    n_fail = int((res.status == "FAIL").sum())
    n_warn = int((res.status == "WARN").sum())
    print(f"\nSUMMARY: {(res.status == 'PASS').sum()} pass, {n_warn} warn, {n_fail} fail")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())