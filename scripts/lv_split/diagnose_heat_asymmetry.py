#!/usr/bin/env python3
"""
Diagnose the a/b asymmetry in heat pump / resistive heater electricity.

Hypothesis: the decentral systems (rural, urban decentral) are split symmetrically
by split_heat(), but URBAN CENTRAL heat pumps / resistive heaters sit on the LV
halves while feeding ONE shared (unsplit) central heat bus. For those units a and b
are interchangeable, so the optimiser may load them unevenly at no cost (degenerate
optimum). flex_metrics() lumps them in with the decentral units, producing the gap.

This script breaks the numbers down by system class and half and prints a verdict.
Sits next to analyze_split_lv.py.

    python diagnose_heat_asymmetry.py --network $net --run $RUN --out analysis/heat_diag
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import analyze_split_lv as base
except ImportError:
    sys.exit("diagnose_heat_asymmetry.py must sit next to analyze_split_lv.py")

TECHS = (("heat pump", "heat pump"), ("resistive heater", "resistive heater"))
TOL = 1e-3  # relative a/b difference treated as symmetric


def classify(carrier):
    c = str(carrier).lower()
    if "urban central" in c:
        return "urban central"
    if "urban decentral" in c:
        return "urban decentral"
    if "rural" in c:
        return "rural"
    return "other"


def ab_table(df, value_cols):
    """Sum value_cols by (tech, cls, half) and put a / b side by side with a diff column."""
    out = {}
    for col in value_cols:
        g = df.groupby(["tech", "cls", "half"])[col].sum().unstack("half")
        g = g.reindex(columns=["a", "b"]).fillna(0.0)
        mean = ((g["a"] + g["b"]) / 2).replace(0, np.nan)
        g["diff_pct"] = 100 * (g["b"] - g["a"]) / mean
        out[col] = g
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--network", required=True, help="path to solved .nc OR a name/pattern")
    ap.add_argument("--run", help="run name (folder under results/)")
    ap.add_argument("--root", default="results")
    ap.add_argument("--out", default="analysis_heat_diag")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n, _ = base.load_solved(args.network, args.run, args.root)

    pd.set_option("display.width", 200, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
    w = base.weights(n)
    lv = base.lv_bus_set(n)
    K, L = n.links, n.loads
    kh = base.comp_half(K)

    # ------------------------------------------------------------------ #
    # 1. per-link table: electricity from the LV bus, capacity, class, half
    # ------------------------------------------------------------------ #
    rows = []
    for tech, pat in TECHS:
        idx = list(K.index[K.carrier.str.contains(pat)])
        if not idx:
            continue
        P = base.port_power(n, idx, lv).clip(lower=0)
        elec = P.mul(w, axis=0).sum() / 1e3
        cap = base.optcap(K.loc[idx], "p_nom")
        pmax = K.loc[idx, "p_nom_max"].where(np.isfinite(K.loc[idx, "p_nom_max"]), 0.0)
        node = base.lv_port_bus(K.loc[idx], lv).map(n.buses.location)
        rows.append(pd.DataFrame({
            "tech": tech,
            "cls": K.loc[idx, "carrier"].map(classify),
            "half": kh[idx],
            "node": node,
            "elec_GWh": elec,
            "P_MW": cap,
            "Pmax_MW": pmax,
            "n_links": 1,
        }, index=idx))
    if not rows:
        sys.exit("No heat pump / resistive heater links found.")
    df = pd.concat(rows)
    df.to_csv(out / "heat_links.csv")

    print("\n=== 1. ELECTRICITY, CAPACITY AND LINK COUNT BY SYSTEM CLASS AND HALF ===")
    tabs = ab_table(df, ["elec_GWh", "P_MW", "Pmax_MW", "n_links"])
    for col, t in tabs.items():
        print(f"\n{col}:")
        print(t)
        t.to_csv(out / f"by_class_{col}.csv")

    # ------------------------------------------------------------------ #
    # 2. heat demand by class and half
    # ------------------------------------------------------------------ #
    hl = list(L.index[L.carrier.str.contains("heat")])
    lp = base.load_power(n)
    dem = pd.DataFrame({
        "cls": L.loc[hl, "carrier"].map(classify),
        "half": base.comp_half(L).loc[hl],
        "GWh": lp[hl].mul(w, axis=0).sum() / 1e3,
    })
    print("\n=== 2. HEAT DEMAND BY CLASS (half 'all' = unsplit bus) ===")
    dt = dem.groupby(["cls", "half"]).GWh.sum().unstack("half").fillna(0.0)
    print(dt)
    dt.to_csv(out / "heat_demand_by_class.csv")

    # ------------------------------------------------------------------ #
    # 3. wiring sanity: no heat link may join an 'a' bus to a 'b' bus
    # ------------------------------------------------------------------ #
    print("\n=== 3. WIRING CHECKS ===")
    heat_links = K.loc[df.index]
    mixed = base.mixed_halves(heat_links)
    base.check(not mixed, "no heat pump / resistive heater joins a and b buses", str(mixed[:4]) if mixed else "")
    dec = df[df.cls.isin(["urban decentral", "rural"])]
    base.info("decentral heat links", f"{len(dec)} links ({(dec.half == 'a').sum()} on a, {(dec.half == 'b').sum()} on b)")
    cen = df[df.cls == "urban central"]
    shared = [b for b in K.loc[cen.index, "bus1"].unique() if not str(b).endswith((" a", " b"))]
    base.info("urban central heat pumps feed an UNSPLIT heat bus", f"{len(shared)} shared bus(es)" if shared else "none found")

    # ------------------------------------------------------------------ #
    # 4. verdict
    # ------------------------------------------------------------------ #
    print("\n=== 4. VERDICT ===")
    e = tabs["elec_GWh"]
    asym, sym = [], []
    for (tech, cls), r in e.iterrows():
        tot = r["a"] + r["b"]
        rel = abs(r["a"] - r["b"]) / tot if tot > 0 else 0.0
        (asym if rel > TOL else sym).append((tech, cls, r["a"], r["b"], rel))
    for tech, cls, a, b, rel in sym:
        print(f"[SYMMETRIC ] {tech:17s} {cls:16s} a={a:8.3f}  b={b:8.3f} GWh")
    for tech, cls, a, b, rel in asym:
        print(f"[ASYMMETRIC] {tech:17s} {cls:16s} a={a:8.3f}  b={b:8.3f} GWh  ({100 * rel:.1f}% of class total)")

    asym_cls = {c for _, c, *_ in asym}
    if not asym:
        print("\nNo asymmetry at all in this run: the earlier gap came from the metric, not the model.")
    elif asym_cls <= {"urban central"}:
        print("\nOnly URBAN CENTRAL units are asymmetric: hypothesis confirmed. The decentral split is correct;")
        print("a/b is meaningless for central units (shared heat bus) -> report them pooled, not per half.")
    else:
        print(f"\nAsymmetry also in {sorted(asym_cls - {'urban central'})}: the decentral split itself is suspect.")
        print("Look at tables 1-2: unequal capacity (P_MW / Pmax_MW) points to the config or the split code;")
        print("equal capacity but unequal electricity points to a degenerate optimum or a different COP profile.")

    # per-node view of the central units
    if len(cen):
        print("\nUrban central units, electricity per node and half (GWh):")
        nt = cen.groupby(["tech", "node", "half"]).elec_GWh.sum().unstack("half").reindex(columns=["a", "b"]).fillna(0.0)
        print(nt)
        nt.to_csv(out / "central_per_node.csv")

    pd.DataFrame(base.RESULTS).to_csv(out / "checks.csv", index=False)
    print(f"\nSaved to {out.resolve()}")


if __name__ == "__main__":
    main()