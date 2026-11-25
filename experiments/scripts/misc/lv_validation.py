#!/usr/bin/env python3
"""
LV / HV behavioural validation for the split-LV PyPSA-Eur networks.

Usage
-----
  python lv_validation.py results/.../postnetwork.nc
  python lv_validation.py C2_split.nc --baseline C1_nosplit.nc --outdir lv_val

What it does
------------
Part A (single network)
  A1  distribution-link capacity respected (forward and reversed)
  A2  no simultaneous import + export on the same LV bus
  A3  LV bus energy balance closes in every snapshot
  A4  shadow-price consistency (LV price vs HV price vs link efficiency)
  A5  link utilisation / binding hours per LV bus (bottleneck ranking)
  A6  rooftop solar: potential built, curtailment, local use vs export
  A7  home battery: equivalent full cycles, mean daily charge/discharge profile
  A8  HV side: net withdrawal per HV bus, distribution losses
  A9  asymmetry between the two LV halves ("a" / "b") of each node
Part B (optional, --baseline)
  B1  objective, capacities by carrier, curtailment, HV line congestion, AC prices

Edit the NAMING block below if your component names differ.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

# --------------------------------------------------------------------------
# NAMING BLOCK (taken from the infeasibility report; edit if needed)
# --------------------------------------------------------------------------
LV_BUS_PATTERN = "low voltage"
DIST_LINK_PATTERN = "electricity distribution grid"
REVERSED_SUFFIX = "-reversed"
ROOFTOP_PATTERN = "solar rooftop"
BATT_CHARGER = "home battery charger"
BATT_DISCHARGER = "home battery discharger"
BATT_STORE = "home battery"

TOL = 1e-3          # MW tolerance for feasibility checks
PRICE_TOL = 0.05    # EUR/MWh tolerance for price-consistency check
BIND = 0.99         # utilisation above which a link counts as "binding"

CHECKS = []         # (name, status, detail)


def record(name, ok, detail="", warn=False):
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    CHECKS.append((name, status, detail))
    print(f"[{status}] {name}: {detail}")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def node_of(name):
    """'GR0 2AC low voltage a' -> 'GR0 2AC low voltage' (strip trailing ' a'/' b')."""
    name = name.replace(REVERSED_SUFFIX, "")
    return re.sub(r"\s[a-z]$", "", name)


def split_of(name):
    m = re.search(r"\s([a-z])(?:-reversed)?$", name)
    return m.group(1) if m else ""


def lv_port(row):
    """Port ('0' or '1') of a link that sits on the LV bus."""
    return "1" if LV_BUS_PATTERN in row.bus1 else "0"


def other(port):
    return "0" if port == "1" else "1"


def load_network(path):
    n = pypsa.Network(path)
    if n.links_t.p0.empty:
        raise SystemExit(f"{path}: no optimisation results found (links_t.p0 empty). "
                         "Pass a solved network.")
    return n


def energy(df, w):
    """Weighted energy (MWh) per column."""
    return df.mul(w, axis=0).sum()


def get_dist_links(n):
    l = n.links[n.links.index.str.contains(DIST_LINK_PATTERN, regex=False)]
    fwd = l[~l.index.str.endswith(REVERSED_SUFFIX)]
    rev = l[l.index.str.endswith(REVERSED_SUFFIX)]
    return fwd, rev


def cap(n, names):
    return (n.links.loc[names, "p_nom_opt"] * n.links.loc[names, "p_max_pu"]).astype(float)


# --------------------------------------------------------------------------
# Part A
# --------------------------------------------------------------------------
def part_a(n, outdir):
    w = n.snapshot_weightings.generators
    fwd, rev = get_dist_links(n)
    if fwd.empty:
        raise SystemExit("No distribution links found - check DIST_LINK_PATTERN.")
    lv_buses = n.buses.index[n.buses.index.str.contains(LV_BUS_PATTERN, regex=False)]
    print(f"\nFound {len(lv_buses)} LV buses, {len(fwd)} forward and {len(rev)} reversed "
          f"distribution links, {len(n.snapshots)} snapshots.\n")

    p0 = n.links_t.p0
    cap_f, cap_r = cap(n, fwd.index), cap(n, rev.index)

    # A1 capacity respected -------------------------------------------------
    viol_f = (p0[fwd.index] - cap_f).clip(lower=0).max()
    viol_r = (p0[rev.index] - cap_r).clip(lower=0).max() if len(rev) else pd.Series(dtype=float)
    worst = max(viol_f.max(), viol_r.max() if len(viol_r) else 0)
    record("A1 link capacity respected", worst <= TOL, f"max violation {worst:.2e} MW")
    if len(rev):
        inf_cap = (~np.isfinite(cap_r)).sum() + (cap_r > 1e6).sum()
        record("A1b reversed links have a finite cap", inf_cap == 0,
               f"{inf_cap} reversed links uncapped (export direction unconstrained)"
               if inf_cap else "all capped", warn=True)
        same = np.allclose(cap_f.values, cap_r.reindex(
            [i + REVERSED_SUFFIX for i in fwd.index]).values, equal_nan=True, rtol=1e-6)
        print(f"      forward cap == reversed cap for every pair: {same}")

    # A2 simultaneous import/export ----------------------------------------
    sim_total = 0
    for f in fwd.index:
        r = f + REVERSED_SUFFIX
        if r in p0.columns:
            sim_total += ((p0[f] > TOL) & (p0[r] > TOL)).sum()
    record("A2 no simultaneous import & export", sim_total == 0,
           f"{sim_total} (link, snapshot) cases", warn=True)

    # A3 LV balance ----------------------------------------------------------
    imb = n.buses_t.p[lv_buses].abs().max().max()
    record("A3 LV bus balance closes", imb <= 1e-2, f"max |imbalance| {imb:.2e} MW")

    # A4 price consistency ---------------------------------------------------
    price = n.buses_t.marginal_price
    mu = n.links_t.mu_upper if "mu_upper" in n.links_t else None
    bad = 0
    tested = 0
    for name in list(fwd.index) + list(rev.index):
        row = n.links.loc[name]
        c = float(cap(n, [name]).iloc[0])
        if not np.isfinite(c) or c <= 0:
            continue
        flowing = (p0[name] > TOL) & (p0[name] < BIND * c)
        gap = row.efficiency * price[row.bus1] - price[row.bus0]
        tested += int(flowing.sum())
        bad += int((gap[flowing].abs() > PRICE_TOL).sum())
    record("A4 price = HV price / efficiency when link not binding",
           bad == 0, f"{bad} of {tested} unconstrained flowing snapshots deviate "
                     f"> {PRICE_TOL} EUR/MWh", warn=True)

    # A5 utilisation / bottleneck ranking -------------------------------------
    rows = []
    for f in fwd.index:
        r = f + REVERSED_SUFFIX
        lv = fwd.loc[f, f"bus{lv_port(fwd.loc[f])}"]
        c_f = cap_f[f]
        c_r = cap_r[r] if r in cap_r.index else np.nan
        imp, exp = p0[f], (p0[r] if r in p0.columns else pd.Series(0.0, index=p0.index))
        rows.append(dict(
            link=f, lv_bus=lv, node=node_of(lv), split=split_of(lv),
            cap_import_MW=c_f, cap_export_MW=c_r,
            peak_import_MW=imp.max(), peak_export_MW=exp.max(),
            util_import_max=imp.max() / c_f if c_f > 0 else np.nan,
            binding_import_pct=100 * (imp >= BIND * c_f).mean() if c_f > 0 else np.nan,
            binding_export_pct=100 * (exp >= BIND * c_r).mean() if c_r and c_r > 0 else np.nan,
            import_GWh=(imp * w).sum() / 1e3, export_GWh=(exp * w).sum() / 1e3,
            loss_GWh=((imp * w).sum() * (1 - fwd.loc[f, "efficiency"])) / 1e3,
            mu_import_mean=float(mu[f].mean()) if mu is not None and f in mu else np.nan,
        ))
    summ = pd.DataFrame(rows).set_index("link")
    summ.round(3).to_csv(outdir / "lv_link_summary.csv")
    top = summ.sort_values("binding_import_pct", ascending=False).head(10)
    print("\nTop-10 import bottlenecks (share of hours at >= 99% of cap):")
    print(top[["lv_bus", "cap_import_MW", "peak_import_MW",
               "binding_import_pct", "mu_import_mean"]].round(2).to_string())
    n_bind = (summ.binding_import_pct > 0).sum()
    record("A5 LV constraints bind somewhere", n_bind > 0,
           f"{n_bind}/{len(summ)} links bind in import direction "
           "(0 = constraint never active, i.e. model equals the no-split case)", warn=True)

    # A6 rooftop solar ---------------------------------------------------------
    gens = n.generators[n.generators.index.str.contains(ROOFTOP_PATTERN, regex=False)]
    if len(gens):
        avail = n.generators_t.p_max_pu[gens.index].mul(gens.p_nom_opt, axis=1)
        disp = n.generators_t.p[gens.index]
        curt = (avail - disp).clip(lower=0)
        solar = pd.DataFrame({
            "p_nom_opt_MW": gens.p_nom_opt,
            "p_nom_max_MW": gens.p_nom_max,
            "potential_built_pct": 100 * gens.p_nom_opt / gens.p_nom_max.replace(0, np.nan),
            "generation_GWh": energy(disp, w) / 1e3,
            "curtailment_pct": 100 * energy(curt, w) / energy(avail, w).replace(0, np.nan),
        })
        solar["lv_bus"] = [gens.loc[g, "bus"] for g in solar.index]
        # local use vs export
        exp_map = {}
        for f in fwd.index:
            r = f + REVERSED_SUFFIX
            if r in p0.columns:
                exp_map[fwd.loc[f, f"bus{lv_port(fwd.loc[f])}"]] = (p0[r] * w).sum() / 1e3
        solar["export_GWh_via_link"] = solar.lv_bus.map(exp_map)
        solar["exported_share_pct"] = 100 * solar.export_GWh_via_link / solar.generation_GWh.replace(0, np.nan)
        solar.round(2).to_csv(outdir / "lv_solar_summary.csv")
        print("\nRooftop solar (system):",
              f"built {solar.p_nom_opt_MW.sum():.0f} MW of {solar.p_nom_max_MW.sum():.0f} MW potential, "
              f"curtailment {100 * energy(curt, w).sum() / max(energy(avail, w).sum(), 1e-9):.2f}%")
        record("A6 solar dispatch <= availability",
               ((disp - avail) > TOL).sum().sum() == 0, "no snapshot above p_max_pu*p_nom_opt")

    # A7 home battery -----------------------------------------------------------
    stores = n.stores[n.stores.index.str.contains(BATT_STORE, regex=False)]
    ch = [i for i in n.links.index if BATT_CHARGER in i]
    dis = [i for i in n.links.index if BATT_DISCHARGER in i]
    if len(stores) and ch and dis:
        e_nom = stores.e_nom_opt.sum()
        e_dis = energy(p0[dis], w).sum()
        cycles = e_dis / e_nom if e_nom > 0 else np.nan
        print(f"\nHome batteries: {e_nom:.0f} MWh energy cap, {e_dis / 1e3:.1f} GWh discharged, "
              f"{cycles:.1f} equivalent full cycles over the run")
        hod = p0.index.hour
        prof = pd.DataFrame({
            "charge_MW": p0[ch].sum(axis=1).groupby(hod).mean(),
            "discharge_MW": p0[dis].sum(axis=1).groupby(hod).mean(),
        })
        if len(gens):
            prof["solar_MW"] = n.generators_t.p[gens.index].sum(axis=1).groupby(hod).mean()
        prof["lv_import_MW"] = p0[fwd.index].sum(axis=1).groupby(hod).mean()
        rev_cols = [f + REVERSED_SUFFIX for f in fwd.index if f + REVERSED_SUFFIX in p0.columns]
        prof["lv_export_MW"] = p0[rev_cols].sum(axis=1).groupby(hod).mean() if rev_cols else 0.0
        prof.round(2).to_csv(outdir / "daily_profile.csv")
        net = (n.stores_t.p[stores.index].mul(w, axis=0)).sum().abs().max()
        record("A7 battery net energy over run ~ 0 (cyclic, no standing loss)",
               net <= 1e-3 * max(e_nom, 1), f"max |net| {net:.2f} MWh, {cycles:.1f} cycles", warn=True)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            prof.plot(ax=ax[0], title="Mean daily profile, all LV buses (MW)")
            ax[0].set_xlabel("hour of day")
            util = pd.concat({f: p0[f] / cap_f[f] for f in fwd.index if cap_f[f] > 0}, axis=1)
            for c in top.index[:5]:
                ax[1].plot(np.sort(util[c].values)[::-1], label=c.replace(" electricity distribution grid", ""))
            ax[1].set_title("Import utilisation duration curves (top 5 bottlenecks)")
            ax[1].set_xlabel("snapshots (sorted)"); ax[1].set_ylabel("flow / cap"); ax[1].legend(fontsize=6)
            fig.tight_layout(); fig.savefig(outdir / "lv_overview.png", dpi=150)
        except Exception as e:  # plotting is optional
            print("      (plot skipped:", e, ")")

    # A8 HV side ------------------------------------------------------------------
    hv_rows = {}
    for name in list(fwd.index) + list(rev.index):
        row = n.links.loc[name]
        hv_port = other(lv_port(row))
        hv_bus = row[f"bus{hv_port}"]
        # withdrawal from HV bus = p<port>; negative = injection into HV bus
        hv_rows.setdefault(hv_bus, []).append(n.links_t[f"p{hv_port}"][name])
    hv = pd.DataFrame({b: sum(v) for b, v in hv_rows.items()})
    hv_summary = pd.DataFrame({
        "peak_withdrawal_MW": hv.max().clip(lower=0), "peak_injection_MW": (-hv.min()).clip(lower=0) + 0.0,
        "net_energy_GWh": energy(hv, w) / 1e3,
    })
    hv_summary.round(2).to_csv(outdir / "hv_summary.csv")
    all_p0 = energy(p0[fwd.index], w).sum()
    all_p1 = energy(n.links_t.p1[fwd.index], w).sum()
    losses = all_p0 + all_p1  # p1 is negative when delivering
    expected = all_p0 * (1 - fwd.efficiency.mean())
    print(f"\nHV side: peak withdrawal {hv_summary.peak_withdrawal_MW.max():.0f} MW, "
          f"peak injection {hv_summary.peak_injection_MW.max():.0f} MW; "
          f"distribution losses {losses / 1e3:.1f} GWh "
          f"({100 * losses / max(all_p0, 1e-9):.1f}% of import)")
    record("A8 distribution losses match link efficiency",
           abs(losses - expected) <= 1e-3 * max(all_p0, 1), f"{losses:.0f} vs {expected:.0f} MWh")

    # A9 asymmetry between LV halves -------------------------------------------------
    summ["flow_GWh"] = summ.import_GWh
    pivot = summ.pivot_table(index="node", columns="split", values="import_GWh")
    if pivot.shape[1] > 1:
        rel = (pivot.max(axis=1) - pivot.min(axis=1)) / pivot.mean(axis=1).replace(0, np.nan)
        print(f"\nA9 import asymmetry between LV halves: median {100 * rel.median():.1f}%, "
              f"max {100 * rel.max():.1f}% (0% means halves are identical)")
        pivot.round(2).to_csv(outdir / "lv_halves_import_GWh.csv")


# --------------------------------------------------------------------------
# Part B: comparison with the no-split baseline
# --------------------------------------------------------------------------
def summarise(n):
    out = {"objective_MEUR": n.objective / 1e6}
    g = n.generators.groupby("carrier").p_nom_opt.sum()
    l = n.links.groupby("carrier").p_nom_opt.sum()
    s = n.stores.groupby("carrier").e_nom_opt.sum()
    for k, v in g.items(): out[f"gen_MW | {k}"] = v
    for k, v in l.items(): out[f"link_MW | {k}"] = v
    for k, v in s.items(): out[f"store_MWh | {k}"] = v
    # curtailment of utility-scale variable renewables
    vre = n.generators[n.generators.carrier.isin(["solar", "onwind", "offwind-ac", "offwind-dc", "offwind-float"])]
    if len(vre):
        avail = n.generators_t.p_max_pu.reindex(columns=vre.index).mul(vre.p_nom_opt, axis=1)
        w = n.snapshot_weightings.generators
        out["VRE_curtailment_GWh"] = energy((avail - n.generators_t.p[vre.index]).clip(lower=0), w).sum() / 1e3
    # HV prices
    ac = n.buses.index[n.buses.carrier == "AC"]
    pr = n.buses_t.marginal_price[ac]
    out["AC_price_mean"] = pr.mean().mean()
    out["AC_price_p95"] = pr.stack().quantile(0.95)
    out["AC_price_std"] = pr.stack().std()
    # HV line congestion
    if len(n.lines) and not n.lines_t.p0.empty:
        c = (n.lines.s_nom_opt * n.lines.s_max_pu).replace(0, np.nan)
        u = n.lines_t.p0.abs().div(c, axis=1)
        out["HV_line_mean_util"] = u.mean().mean()
        out["HV_line_congested_pct"] = 100 * (u >= BIND).stack().mean()
    return pd.Series(out)


def part_b(n_split, n_base, outdir):
    a, b = summarise(n_base), summarise(n_split)
    df = pd.concat({"baseline": a, "split": b}, axis=1)
    df["diff"] = df.split - df.baseline
    df["diff_pct"] = 100 * df["diff"] / df.baseline.abs().replace(0, np.nan)
    df.round(3).to_csv(outdir / "comparison.csv")
    key = [i for i in df.index if not i.startswith(("gen_MW", "link_MW", "store_MWh"))]
    print("\nB1 split vs baseline (key indicators):")
    print(df.loc[key].round(3).to_string())
    big = df[df.index.str.startswith(("gen_MW", "link_MW", "store_MWh"))]
    big = big[big["diff"].abs() > 1e-3 * max(big.baseline.abs().max(), 1)]
    if len(big):
        print("\nCapacity changes (>0.1% of largest):")
        print(big.round(1).sort_values("diff_pct", key=abs, ascending=False).head(15).to_string())
    record("B1 split costs at least as much as baseline",
           df.loc["objective_MEUR", "diff"] >= -1e-6 * abs(df.loc["objective_MEUR", "baseline"]),
           f"delta objective {df.loc['objective_MEUR', 'diff']:.3f} MEUR "
           "(adding constraints/losses cannot lower cost unless inputs also changed)", warn=True)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("network", help="solved network (.nc) with split LV buses")
    ap.add_argument("--baseline", help="solved no-split network (.nc) for comparison")
    ap.add_argument("--outdir", default="experiments/results/lv_validation_out")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    n = load_network(args.network)
    part_a(n, outdir)
    if args.baseline:
        part_b(n, load_network(args.baseline), outdir)

    pd.DataFrame(CHECKS, columns=["check", "status", "detail"]).to_csv(outdir / "checks.csv", index=False)
    print("\n==== SUMMARY ====")
    for name, status, _ in CHECKS:
        print(f"{status:5s} {name}")
    print(f"\nOutputs written to {outdir.resolve()}")


if __name__ == "__main__":
    main()
    
    
#!/usr/bin/env python3
"""
Diagnose the two WARNs from lv_validation.py:
  A2  simultaneous import & export on the same LV bus
  A4  LV/HV price relation violated while the link is flowing but not at its cap

Usage:  python diagnose_a2_a4.py <split_network.nc>
(keep this file next to lv_validation.py)
"""
import sys
import numpy as np
import pandas as pd
from experiments.scripts.misc.lv_validation import (REVERSED_SUFFIX, TOL, BIND, PRICE_TOL,
                           get_dist_links, cap, lv_port, other, load_network)

n = load_network(sys.argv[1])
w = n.snapshot_weightings.generators
fwd, rev = get_dist_links(n)
p0, price = n.links_t.p0, n.buses_t.marginal_price
Z = 0.01  # EUR/MWh: "price is zero" threshold

print("=== 1. Link parameters (first pair) ===")
f0 = fwd.index[0]
cols = ["bus0", "bus1", "efficiency", "p_nom_extendable", "p_nom", "p_nom_opt", "p_min_pu", "p_max_pu"]
print(n.links.loc[[f0, f0 + REVERSED_SUFFIX], cols].to_string())
print("\nDo all reversed links have the same efficiency as their forward link?",
      np.allclose(n.links.loc[fwd.index, "efficiency"].values,
                  n.links.loc[[i + REVERSED_SUFFIX for i in fwd.index], "efficiency"].values))

print("\n=== 2. A2: simultaneous import/export per LV bus ===")
rows = []
for f in fwd.index:
    r = f + REVERSED_SUFFIX
    lvp = lv_port(n.links.loc[f])
    hv, lv = n.links.loc[f, f"bus{other(lvp)}"], n.links.loc[f, f"bus{lvp}"]
    pf, pr = p0[f], p0[r]
    sim = (pf > TOL) & (pr > TOL)
    ph = price[hv]
    cf = float(cap(n, [f]).iloc[0])
    rows.append(dict(link=f.replace(" electricity distribution grid", ""),
                     sim_snaps=int(sim.sum()),
                     sim_at_HV_price_zero=int((sim & (ph.abs() < Z)).sum()),
                     sim_at_HV_price_neg=int((sim & (ph < -Z)).sum()),
                     sim_at_HV_price_pos=int((sim & (ph > Z)).sum()),
                     loop_GWh=float((np.minimum(pf, pr) * w).sum() / 1e3),
                     HV_price_zero_snaps=int((ph.abs() < Z).sum()),
                     HV_price_min=round(float(ph.min()), 2),
                     net_import_binding_pct=round(100 * float(((pf - pr) >= BIND * cf).mean()), 1)))
d = pd.DataFrame(rows).set_index("link")
print(d.to_string())
print(f"\n(snapshots per link: {len(p0)})")

print("\n=== 3. A4: price-relation deviations (flowing, not at cap) ===")
recs = []
for name in list(fwd.index) + list(rev.index):
    row = n.links.loc[name]
    c = float(cap(n, [name]).iloc[0])
    if not np.isfinite(c) or c <= 0:
        continue
    flowing = (p0[name] > TOL) & (p0[name] < BIND * c)
    gap = row.efficiency * price[row.bus1] - price[row.bus0]
    dev = flowing & (gap.abs() > PRICE_TOL)
    is_rev = name.endswith(REVERSED_SUFFIX)
    partner = name.replace(REVERSED_SUFFIX, "") if is_rev else name + REVERSED_SUFFIX
    sim = (p0[name] > TOL) & (p0[partner] > TOL)
    hvp = price[row.bus0 if not is_rev else row.bus1]  # HV bus price
    for t in gap.index[dev]:
        recs.append(dict(link=name, dir="export" if is_rev else "import", gap=gap[t],
                         simultaneous=bool(sim[t]), hv_zero=abs(hvp[t]) < Z, hv_price=hvp[t]))
r = pd.DataFrame(recs)
if r.empty:
    print("no deviations")
else:
    print(f"{len(r)} deviating (link, snapshot) cases")
    print(r.groupby(["dir", "simultaneous", "hv_zero"]).size().rename("cases").to_string())
    print("\ngap = eff*price(bus1) - price(bus0); expected 0 when unconstrained")
    print(r.groupby("dir").gap.describe()[["count", "min", "50%", "max"]].round(2).to_string())
    only = r[~r.simultaneous & ~r.hv_zero]
    if len(only):
        print(f"\n{len(only)} deviations are NOT simultaneous and NOT at zero HV price -> "
              "these point to an extra constraint on the link (see hints below).")
        print(only.groupby("link").gap.agg(["count", "min", "median", "max"]).round(2).to_string())

print("""
=== How to read this ===
- sim_at_HV_price_pos > 0 : simultaneous flow at a positive price is not optimal in a plain LP
  (with efficiency < 1 both-way flow only burns energy). It means the link is tied to another
  constraint (shared p_nom, net-flow limit, reinforcement coupling) -> inspect that constraint.
- sim only at HV price zero/negative : harmless degeneracy (energy is free to dump), common in
  weeks with surplus renewables. Can be silenced by a tiny marginal_cost on the links.
- net_import_binding_pct >> binding in lv_validation : the limit acts on NET flow, not per link.
- deviations with simultaneous=False and hv_zero=False : price relation broken by an extra constraint.
""")    