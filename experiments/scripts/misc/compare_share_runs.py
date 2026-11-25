"""
Compare solved PyPSA-Eur networks from share sweeps (1D and 2D) against the
40/30/30 baseline. Reads the .nc files directly; no legacy a/b split logic.

Every metric is reported PER COMPONENT AND CARRIER (Generator / Link / Store / Line),
so each number can be traced to the technology that produced it, and every metric
is also expressed as a percentage change:
    pct_vs_baseline : change relative to the baseline network
    pct_vs_prev_x   : change relative to the previous run along the x share axis
                      (for 1D sweeps this is simply the previous run)
    pct_vs_prev_y   : 2D only, change relative to the previous run along the y axis

Manifest CSV (one row per run), e.g. runs.csv:
    file,label,set,x,y,x_name,y_name
    runs/a.nc,elec 4/3/3->5/3/2,1d,,,,
    runs/b.nc,EV 4/3/3->5/3/2,1d,,,,
    runs/c.nc,elec + EV 5/3/2,2d,,,,
Columns:
    file    (required) path to the solved .nc
    label   (optional) readable description of the run, used on the plots (default: file name)
    set     (optional) "1d" or "2d"; only needed for the numeric line plots / heatmaps
    x, y    (optional) numeric share values. Only needed if the runs lie on a numeric axis
            (e.g. urban share 0.2, 0.3, 0.4). Leave empty when runs differ in WHICH sector
            was changed; you then get the overview heatmaps (carrier x run) instead.
    x_name, y_name (optional) axis labels for the numeric plots (first row is used).
Always produced: overview_pct_<metric>.png (carrier x run heatmap, % change vs baseline).
Produced only when x (and y) are numeric: 1d_pct_*.png line plots and 2d_pct_*.png heatmaps.

Usage:
    python compare_share_runs.py --manifest runs.csv \
        --baseline runs/baseline_433.nc --out results/

Metrics (per carrier, unit in the 'unit' column):
    Generator : capacity [MW], energy [GWh], peak [MW], curtailment [GWh]
                (curtailment only for generators with a varying availability profile)
    Link      : capacity [MW], energy [GWh] (bus0->bus1 flow), peak [MW],
                energy_reverse [GWh] (only if non-zero),
                binding_max / binding_mean [% time] (distribution links only)
    Store     : capacity [MWh], discharge [GWh], full_cycles [-]
    Line      : capacity [MW], energy [GWh]
    all       : capex [EUR], opex [EUR], total_cost [EUR]
    Distribution links are split into "(import)", "(export)" and
    "(reinforcement, import/export)" carriers.
    capex of fixed (free) distribution capacity is priced at the baseline grid capex,
    so costs are comparable between runs ("adjusted" cost). System/ALL holds the totals.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

DEFAULT_GRID_CAPEX = 1217.0  # EUR/MW, fallback if it cannot be read from the baseline
DIST = "distribution grid"
KEY = ["component", "carrier", "metric", "unit"]
MIN_SHOWN_PCT = 1e-3  # carriers whose largest change is below this (in %) are not plotted
TOP_K_LINES = 8
TOP_K_HEATMAPS = 4


# ----------------------------------------------------------------------------- helpers
def label_links(n):
    """Carrier label per link; distribution links are split by direction / reinforcement."""
    L = n.links
    lab = L.carrier.astype(str).copy()
    dist = L.index[lab.str.lower().str.contains(DIST, regex=False)]
    bus0_carrier = n.buses.carrier.reindex(L.loc[dist, "bus0"]).values
    for l, c0 in zip(dist, bus0_carrier):
        direction = "import" if c0 == "AC" else "export"
        if "reinforce" in str(l).lower():
            lab[l] = f"{L.at[l, 'carrier']} (reinforcement, {direction})"
        else:
            lab[l] = f"{L.at[l, 'carrier']} ({direction})"
    return lab


def segment_of(*names):
    s = " ".join(str(x).lower() for x in names)
    for seg in ("suburban", "urban", "rural"):
        if seg in s:
            return seg
    return ""


def ref_capex_from(n):
    L = n.links
    lab = label_links(n)
    idx = L.index[lab.str.contains("(import)", regex=False)]
    c = L.capital_cost[idx]
    c = c[c > 0]
    return float(c.median()) if len(c) else DEFAULT_GRID_CAPEX


def wsum(df, w):
    """Weighted sum over time and columns."""
    if df.shape[1] == 0:
        return 0.0
    return float(df.mul(w, axis=0).sum().sum())


def fm(v):
    """Readable EUR amount."""
    return f"{v / 1e6:,.2f}M" if abs(v) >= 1e6 else f"{v:,.0f}"


def flat(d, prefix=""):
    """Flatten nested dict to {'a.b.c': value}."""
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = d
    return out


def cfg_shares(n):
    """Profile-share settings that were used to build/solve this network (PyPSA-Eur stores
    its config in n.meta). Returns {} if the network carries no config."""
    try:
        meta = n.meta
        if isinstance(meta, str):
            import json
            meta = json.loads(meta)
        return flat(meta.get("sector", {}).get("lv_split_profiles", {}), "lv_split_profiles.")
    except Exception:
        return {}


def cfg_diff(cfg, base_cfg):
    """Readable list of the share settings that differ from the baseline network."""
    if not cfg or not base_cfg:
        return "n/a (no config stored in network)"
    d = [f"{k.replace('lv_split_profiles.', '')} {base_cfg.get(k)}->{v}"
         for k, v in cfg.items() if k in base_cfg and base_cfg[k] != v and k != "lv_split_profiles.enable"]
    return "; ".join(d) if d else "identical to baseline"


def run_ids(files):
    """Unique run id per file. File names are often identical (results/<run>/networks/<same>.nc),
    so fall back to the run folder name."""
    files = [Path(f) for f in files]
    stems = [f.stem for f in files]
    if len(set(stems)) == len(stems):
        return stems
    folders = [f.parent.parent.name for f in files]
    if len(set(folders)) == len(folders):
        return folders
    return [str(f) for f in files]


def stored(n, comp_t, attr, index, what):
    """Time series `attr` of a component for ALL components in `index`.
    PyPSA does not write all-zero time series to .nc files, so a component that never
    dispatched has no column (this is why indexing directly raises a KeyError). Such
    components are filled with 0."""
    return getattr(getattr(n, comp_t), attr).reindex(columns=index, fill_value=0.0)


# ----------------------------------------------------------------------------- extraction
def collect(n, ref_capex, report=False):
    """Return (rows, link_rows). rows: (component, carrier, metric, unit, value)."""
    obj = n.objective
    if obj is None or not np.isfinite(float(obj)) or float(obj) == 0.0:
        raise RuntimeError("network has no objective value: not solved?")
    w = n.snapshot_weightings.objective
    rows = []

    def add(comp, car, metric, unit, val):
        rows.append((comp, str(car), metric, unit, float(val)))

    existing_total = 0.0
    capex_raw_total = 0.0
    opex_total = 0.0

    # ---- Generators
    G = n.generators
    gp = stored(n, "generators_t", "p", G.index, "generators")
    pmax = n.get_switchable_as_dense("Generator", "p_max_pu")
    mc = n.get_switchable_as_dense("Generator", "marginal_cost")
    for car, idx in G.groupby(G.carrier.astype(str)).groups.items():
        idx = list(idx)
        cap = G.p_nom_opt[idx]
        add("Generator", car, "capacity", "MW", cap.sum())
        add("Generator", car, "energy", "GWh", wsum(gp[idx], w) / 1e3)
        add("Generator", car, "peak", "MW", gp[idx].sum(axis=1).max())
        varying = [g for g in idx if (pmax[g] != 1).any()]
        if varying:
            curt = (pmax[varying] * G.p_nom_opt[varying] - gp[varying]).clip(lower=0)
            add("Generator", car, "curtailment", "GWh", wsum(curt, w) / 1e3)
        capex = float((G.p_nom_opt[idx] * G.capital_cost[idx])[G.p_nom_extendable[idx]].sum())
        opex = wsum(gp[idx] * mc[idx], w)
        add("Generator", car, "capex", "EUR", capex)
        add("Generator", car, "opex", "EUR", opex)
        capex_raw_total += capex
        opex_total += opex

    # ---- Links
    L = n.links
    lab = label_links(n)
    p0 = stored(n, "links_t", "p0", L.index, "links")
    mcl = n.get_switchable_as_dense("Link", "marginal_cost")
    link_rows = []
    for car in lab.unique():
        idx = list(L.index[lab == car])
        cap = L.p_nom_opt[idx]
        f = p0[idx]
        add("Link", car, "capacity", "MW", cap.sum())
        add("Link", car, "energy", "GWh", wsum(f.clip(lower=0), w) / 1e3)
        add("Link", car, "peak", "MW", f.sum(axis=1).max())
        neg = wsum((-f).clip(lower=0), w) / 1e3
        if neg > 1e-6:
            add("Link", car, "energy_reverse", "GWh", neg)
        is_dist = DIST in car.lower()
        existing = 0.0
        if is_dist:
            bind = ((f >= 0.999 * cap) & (cap > 1e-6)).mul(w, axis=0).sum() / w.sum() * 100
            add("Link", car, "binding_max", "% time", bind.max())
            add("Link", car, "binding_mean", "% time", bind.mean())
            fixed = [l for l in idx if (not L.at[l, "p_nom_extendable"]) and L.at[l, "capital_cost"] == 0]
            existing = float(L.p_nom_opt[fixed].sum())
            add("Link", car, "existing_capacity", "MW", existing)
            if "(import)" in car:
                existing_total += existing
            for l in idx:
                other = L.at[l, "bus1"] if n.buses.carrier.get(L.at[l, "bus0"], "") == "AC" else L.at[l, "bus0"]
                link_rows.append({
                    "link": l, "carrier": car, "bus0": L.at[l, "bus0"], "bus1": L.at[l, "bus1"],
                    "segment": segment_of(l, other),
                    "p_nom_opt_MW": L.at[l, "p_nom_opt"],
                    "peak_flow_MW": float(p0[l].max()),
                    "binding_pct": float(bind[l]),
                    "energy_GWh": wsum(p0[[l]].clip(lower=0), w) / 1e3,
                })
        capex_raw = float((L.p_nom_opt[idx] * L.capital_cost[idx])[L.p_nom_extendable[idx]].sum())
        opex = wsum(p0[idx] * mcl[idx], w)
        # adjusted capex: free fixed distribution capacity priced at the reference grid capex
        # (the export link shares the physical connection of the import link, so it is not priced twice)
        priced = existing if "(import)" in car else 0.0
        add("Link", car, "capex", "EUR", capex_raw + priced * ref_capex)
        add("Link", car, "opex", "EUR", opex)
        capex_raw_total += capex_raw
        opex_total += opex

    # ---- Stores
    S = n.stores
    sp = stored(n, "stores_t", "p", S.index, "stores")
    mcs = n.get_switchable_as_dense("Store", "marginal_cost")
    for car, idx in S.groupby(S.carrier.astype(str)).groups.items():
        idx = list(idx)
        cap = float(S.e_nom_opt[idx].sum())
        dis = wsum(sp[idx].clip(lower=0), w)
        add("Store", car, "capacity", "MWh", cap)
        add("Store", car, "discharge", "GWh", dis / 1e3)
        add("Store", car, "full_cycles", "-", dis / cap if cap > 1e-6 else 0.0)
        capex = float((S.e_nom_opt[idx] * S.capital_cost[idx])[S.e_nom_extendable[idx]].sum())
        opex = wsum(sp[idx] * mcs[idx], w)
        add("Store", car, "capex", "EUR", capex)
        add("Store", car, "opex", "EUR", opex)
        capex_raw_total += capex
        opex_total += opex

    # ---- Lines
    Li = n.lines
    if len(Li):
        lp = stored(n, "lines_t", "p0", Li.index, "lines")
        for car, idx in Li.groupby(Li.carrier.astype(str)).groups.items():
            idx = list(idx)
            add("Line", car, "capacity", "MW", Li.s_nom_opt[idx].sum())
            add("Line", car, "energy", "GWh", wsum(lp[idx].abs(), w) / 1e3)
            capex = float((Li.s_nom_opt[idx] * Li.capital_cost[idx])[Li.s_nom_extendable[idx]].sum())
            add("Line", car, "capex", "EUR", capex)
            capex_raw_total += capex

    # ---- Storage units (hydro, pumped hydro)
    SU = n.storage_units
    if len(SU):
        sud = stored(n, "storage_units_t", "p_dispatch", SU.index, "storage units")
        soc = n.storage_units_t.state_of_charge.reindex(columns=SU.index, fill_value=0.0)
        mcu = n.get_switchable_as_dense("StorageUnit", "marginal_cost")
        mcu_s = SU["marginal_cost_storage"] if "marginal_cost_storage" in SU else pd.Series(0.0, index=SU.index)
        for car, idx in SU.groupby(SU.carrier.astype(str)).groups.items():
            idx = list(idx)
            add("StorageUnit", car, "capacity", "MW", SU.p_nom_opt[idx].sum())
            add("StorageUnit", car, "energy_capacity", "MWh", (SU.p_nom_opt[idx] * SU.max_hours[idx]).sum())
            add("StorageUnit", car, "discharge", "GWh", wsum(sud[idx], w) / 1e3)
            capex = float((SU.p_nom_opt[idx] * SU.capital_cost[idx])[SU.p_nom_extendable[idx]].sum())
            opex = wsum(sud[idx] * mcu[idx], w) + wsum(soc[idx] * mcu_s[idx], w)
            add("StorageUnit", car, "capex", "EUR", capex)
            add("StorageUnit", car, "opex", "EUR", opex)
            capex_raw_total += capex
            opex_total += opex

    # ---- per-carrier total_cost and system totals
    df = pd.DataFrame(rows, columns=["component", "carrier", "metric", "unit", "value"])
    cc = df[df.metric.isin(["capex", "opex"])].pivot_table(
        index=["component", "carrier"], columns="metric", values="value", aggfunc="sum").fillna(0.0)
    for (comp, car), r in cc.iterrows():
        add(comp, car, "total_cost", "EUR", r.get("capex", 0.0) + r.get("opex", 0.0))
    objective = float(n.objective)
    # PyPSA's objective EXCLUDES the capex of already-built capacity (p_nom of extendable
    # components); total system cost = objective + objective_constant. The per-carrier capex
    # above (p_nom_opt * capital_cost) includes that existing capacity, so it adds up to the sum.
    try:
        const = float(n.objective_constant)
        if not np.isfinite(const):
            const = 0.0
    except Exception:
        const = 0.0
    total = objective + const
    obj_adj = total + existing_total * ref_capex
    add("System", "ALL", "objective_raw", "EUR", objective)
    add("System", "ALL", "objective_constant", "EUR", const)
    add("System", "ALL", "total_cost", "EUR", obj_adj)
    add("System", "ALL", "existing_grid", "MW", existing_total)
    # cost closure: total system cost minus what the per-carrier capex/opex above add up to
    add("System", "ALL", "unattributed_cost", "EUR", total - (capex_raw_total + opex_total))
    if report:
        sub = df[df.metric.isin(["capex", "opex"])]
        print("  cost accounting of this network (raw, before grid adjustment):")
        print("    " + "  ".join(f"{k}={fm(v)}" for k, v in
                                  sub.groupby(["component", "metric"]).value.sum().items()))
        print(f"    sum of carriers = {fm(capex_raw_total + opex_total)} EUR   "
              f"objective + objective_constant = {fm(objective)} + {fm(const)} = {fm(total)} EUR")
        big = sub.groupby(["component", "carrier"]).value.sum().sort_values(ascending=False).head(8)
        print("    largest carrier costs (capex+opex, EUR): "
              + "; ".join(f"{c}:{k} {fm(v)}" for (c, k), v in big.items()))
    return rows, link_rows


# ----------------------------------------------------------------------------- percentages
def pct_change(new, old, tol):
    """100*(new-old)/|old|, NaN where |old| <= tol (per-row tolerance)."""
    out = 100.0 * (new - old) / old.abs()
    return out.where(old.abs() > tol)


def step_pct(wide, order, tol):
    """Percentage change of each run vs the previous run in 'order' (list of run names)."""
    out = pd.DataFrame(np.nan, index=wide.index, columns=wide.columns)
    for prev, cur in zip(order[:-1], order[1:]):
        out[cur] = pct_change(wide[cur], wide[prev], tol)
    return out


def build_tables(runs, base_vals, meta, min_rel):
    """runs: {run: DataFrame}. Returns wide values, pct tables."""
    frames = {"baseline": base_vals}
    frames.update(runs)
    wide = pd.concat({k: v.set_index(KEY)["value"] for k, v in frames.items()}, axis=1)
    wide = wide.fillna(0.0)  # carrier absent from a run -> 0
    scale = wide.abs().groupby(level=["component", "metric"]).transform("max").max(axis=1)
    tol = (scale * min_rel).clip(lower=1e-9)
    run_cols = [c for c in wide.columns if c != "baseline"]
    pct_base = pd.DataFrame({c: pct_change(wide[c], wide["baseline"], tol) for c in run_cols})
    pct_px = pd.DataFrame(np.nan, index=wide.index, columns=run_cols)
    pct_py = pd.DataFrame(np.nan, index=wide.index, columns=run_cols)
    m = meta.set_index("run")
    d1 = m[(m["set"] == "1d") & m["x"].notna()].sort_values("x")
    if len(d1):
        o = list(d1.index)
        pct_px[o] = step_pct(wide[o], o, tol)
    d2 = m[(m["set"] == "2d") & m["x"].notna() & m["y"].notna()]
    for yv, g in d2.groupby("y"):
        o = list(g.sort_values("x").index)
        pct_px[o] = step_pct(wide[o], o, tol)
    for xv, g in d2.groupby("x"):
        o = list(g.sort_values("y").index)
        pct_py[o] = step_pct(wide[o], o, tol)
    return wide, pct_base, pct_px, pct_py


# ----------------------------------------------------------------------------- plotting
def pick_rows(pct, metric, k):
    mask = pct.index.get_level_values("metric") == metric
    sub = pct[mask].dropna(how="all")
    # drop carriers that do not react at all (e.g. fixed hydro capacity): max |change| < 0.001 %
    sub = sub[sub.abs().max(axis=1) >= MIN_SHOWN_PCT]
    if sub.empty:
        return sub
    order = sub.abs().max(axis=1).sort_values(ascending=False).index[:k]
    return sub.loc[order]


def row_label(idx):
    comp, car, metric, unit = idx
    return f"{comp}: {car}"


def plot_overview(pct, meta, out, top_n=15):
    """Carrier x run heatmap of % change vs baseline, one figure per metric.
    Works without numeric x/y: columns are the runs (manifest order) named by 'label'."""
    if len(meta) < 1:
        return
    runs = list(meta.run)
    labels = [str(s) for s in meta.label]
    for metric in sorted(set(pct.index.get_level_values("metric"))):
        sub = pick_rows(pct[runs], metric, top_n)
        if sub.empty:
            continue
        lim = np.nanmax(np.abs(sub.values)) or 1.0
        fig, ax = plt.subplots(figsize=(1.0 * len(runs) + 5, 0.38 * len(sub) + 2))
        im = ax.imshow(sub.values, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_xticks(range(len(runs))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(sub))); ax.set_yticklabels([row_label(i) for i in sub.index], fontsize=8)
        for i in range(sub.shape[0]):
            for j in range(sub.shape[1]):
                v = sub.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(v) > 0.6 * lim else "black")
        unit = sub.index.get_level_values("unit")[0]
        ax.set_title(f"{metric} [{unit}]: % change vs baseline", fontsize=9)
        fig.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(out / f"overview_pct_{metric}.png", dpi=150); plt.close(fig)


def plot_1d(pct, meta, out, xname):
    d = meta[(meta["set"] == "1d") & meta["x"].notna()].sort_values("x")
    if d.empty:
        return
    for metric in sorted(set(pct.index.get_level_values("metric"))):
        sub = pick_rows(pct[list(d.run)], metric, TOP_K_LINES)
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for idx, r in sub.iterrows():
            ax.plot(d.x.values, r.values, marker="o", label=row_label(idx))
        ax.axhline(0, color="grey", lw=.8)
        unit = sub.index.get_level_values("unit")[0]
        ax.set_xlabel(xname); ax.set_ylabel(f"{metric} [{unit}]: % change vs baseline")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, .5))
        fig.tight_layout(); fig.savefig(out / f"1d_pct_{metric}.png", dpi=150); plt.close(fig)


def plot_2d(pct, meta, out, xname, yname):
    d = meta[(meta["set"] == "2d") & meta["x"].notna() & meta["y"].notna()]
    if d.empty:
        return
    for metric in sorted(set(pct.index.get_level_values("metric"))):
        sub = pick_rows(pct[list(d.run)], metric, TOP_K_HEATMAPS)
        if sub.empty:
            continue
        n = len(sub)
        ncol = 2 if n > 1 else 1
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4 * nrow), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (idx, r) in zip(axes.ravel(), sub.iterrows()):
            ax.axis("on")
            tab = pd.DataFrame({"x": d.set_index("run").x, "y": d.set_index("run").y, "v": r}) \
                .pivot_table(index="y", columns="x", values="v").sort_index()
            lim = np.nanmax(np.abs(tab.values)) or 1.0
            im = ax.imshow(tab.values, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_xticks(range(len(tab.columns))); ax.set_xticklabels([f"{c:g}" for c in tab.columns])
            ax.set_yticks(range(len(tab.index))); ax.set_yticklabels([f"{c:g}" for c in tab.index])
            for i in range(tab.shape[0]):
                for j in range(tab.shape[1]):
                    v = tab.values[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                                color="white" if abs(v) > 0.6 * lim else "black")
            ax.set_xlabel(xname); ax.set_ylabel(yname)
            ax.set_title(f"{row_label(idx)} [{idx[3]}]\n{metric}: % vs baseline", fontsize=8)
            fig.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(out / f"2d_pct_{metric}.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--baseline", required=True, help="solved 40/30/30 network (.nc)")
    ap.add_argument("--out", default="experiments/results/share_comparison")
    ap.add_argument("--grid-capex", type=float, default=None,
                    help="EUR/MW used to price fixed (free) distribution capacity; default: from baseline")
    ap.add_argument("--min-rel", type=float, default=1e-4,
                    help="percentages are suppressed where the reference value is below this fraction "
                         "of the largest value of the same component/metric (avoids noise from ~0 values)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print("loading baseline", a.baseline)
    nb = pypsa.Network(a.baseline)
    ref = a.grid_capex if a.grid_capex is not None else ref_capex_from(nb)
    print(f"grid capex used to price fixed distribution capacity: {ref:.1f} EUR/MW")
    try:
        rows, link_rows = collect(nb, ref, report=True)
    except Exception as e:
        raise SystemExit(f"Baseline cannot be used: {e}")
    base_df = pd.DataFrame(rows, columns=KEY[:3] + ["unit", "value"])
    all_links = [dict(run="baseline", **r) for r in link_rows]
    base_cfg = cfg_shares(nb)

    man = pd.read_csv(a.manifest)
    ids = run_ids(man["file"])
    runs, meta, skipped = {}, [], []
    for (_, r), name in zip(man.iterrows(), ids):
        try:
            ftime = pd.Timestamp(Path(str(r["file"])).stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")
        except OSError:
            ftime = ""
        print(f"loading {name} <- {r['file']}  (file saved {ftime})")
        try:
            n = pypsa.Network(str(r["file"]))
            rows, link_rows = collect(n, ref)
        except Exception as e:
            print(f"  SKIPPED {name}: {e}")
            skipped.append(name)
            continue
        runs[name] = pd.DataFrame(rows, columns=["component", "carrier", "metric", "unit", "value"])
        all_links += [dict(run=name, **lr) for lr in link_rows]
        cdiff = cfg_diff(cfg_shares(n), base_cfg)
        print(f"  shares actually used (vs baseline): {cdiff}")
        has = lambda c: c in man.columns and pd.notna(r[c])
        meta.append({
            "run": name,
            "set": str(r["set"]).lower() if has("set") else "",
            "x": pd.to_numeric(r["x"], errors="coerce") if has("x") else np.nan,
            "y": pd.to_numeric(r["y"], errors="coerce") if has("y") else np.nan,
            "label": str(r["label"]) if has("label") else name,
            "file_saved": ftime, "config_in_network": cdiff})
    if not runs:
        raise SystemExit("No run could be read; nothing to compare.")
    meta = pd.DataFrame(meta)

    wide, pct_b, pct_px, pct_py = build_tables(runs, base_df, meta, a.min_rel)

    # cost closure summary
    k = ("System", "ALL", "unattributed_cost", "EUR")
    if k in wide.index:
        res = wide.loc[k]
        tot = abs(wide.loc[("System", "ALL", "total_cost", "EUR"), "baseline"])
        print(f"cost closure: system total cost minus per-carrier capex+opex = {fm(res['baseline'])} EUR "
              f"in the baseline ({res['baseline'] / tot * 100:.2f}% of total); largest gap in any run "
              f"{fm(res.abs().max())} EUR")
        if res.abs().max() > 0.01 * tot:
            print("  WARNING: per-carrier costs do not add up to the system total (objective + objective_constant); "
                  "some cost term is not attributed to a carrier (e.g. global constraints, other components).")
    if skipped:
        print("SKIPPED runs (not in the results):", ", ".join(skipped))

    # long table
    run_cols = list(pct_b.columns)
    base = wide["baseline"]
    parts = []
    for c in run_cols:
        d = pd.DataFrame({
            "value": wide[c], "baseline_value": base, "delta": wide[c] - base,
            "pct_vs_baseline": pct_b[c], "pct_vs_prev_x": pct_px[c], "pct_vs_prev_y": pct_py[c]})
        d.insert(0, "run", c)
        parts.append(d.reset_index())
    long = pd.concat(parts, ignore_index=True).merge(meta, on="run")
    # materiality: for EUR rows, the change as % of the baseline SYSTEM total cost
    # (a +50% change of a carrier worth 0.01% of the system moves the system by 0.005%)
    sys_tot = float(base.loc[("System", "ALL", "total_cost", "EUR")])
    long["delta_pct_of_system_cost"] = np.where(long["unit"] == "EUR", long["delta"] / sys_tot * 100, np.nan)
    long = long[["run", "label", "config_in_network", "file_saved", "set", "x", "y"] + KEY +
                ["value", "baseline_value", "delta", "pct_vs_baseline", "delta_pct_of_system_cost",
                 "pct_vs_prev_x", "pct_vs_prev_y"]]
    long.to_csv(out / "long_metrics.csv", index=False)
    wide.to_csv(out / "values_wide.csv")
    pct_b.to_csv(out / "pct_vs_baseline_wide.csv")
    pd.DataFrame(all_links).to_csv(out / "per_link_metrics.csv", index=False)

    xn = man["x_name"].iloc[0] if "x_name" in man and pd.notna(man["x_name"].iloc[0]) else "share x (%)"
    yn = man["y_name"].iloc[0] if "y_name" in man and pd.notna(man["y_name"].iloc[0]) else "share y (%)"
    plot_overview(pct_b, meta, out)
    plot_1d(pct_b, meta, out, xn)
    plot_2d(pct_b, meta, out, xn, yn)
    print("done ->", out)


if __name__ == "__main__":
    main()