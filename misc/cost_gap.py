"""Where does the gap between my component cost sum and n.objective come from?

Usage:  python cost_gap.py network.nc [more.nc ...]
Prints, per component type: capex on all extendable capacity, the part of that capex that
sits on the MINIMUM (already installed) capacity, and opex. The objective typically only
pays for capacity above the minimum, so 'capex - capex_on_min' is the figure to compare.
Also prints n.objective, n.objective_constant, and PyPSA's own statistics totals.
"""
import sys
import pandas as pd
import pypsa

SPEC = [("generators", "p_nom"), ("links", "p_nom"), ("stores", "e_nom"),
        ("storage_units", "p_nom"), ("lines", "s_nom")]


def report(path):
    n = pypsa.Network(path)
    w = n.snapshot_weightings.generators
    rows, top = [], []
    for comp, a in SPEC:
        df = getattr(n, comp)
        if df.empty:
            continue
        ext = df[f"{a}_extendable"]
        capex = (df[f"{a}_opt"] * df.capital_cost * ext).sum()
        capex_min = (df[f"{a}_min"] * df.capital_cost * ext).sum()
        flow = {"generators": n.generators_t.p, "links": n.links_t.p0,
                "storage_units": n.storage_units_t.p_dispatch,
                "stores": n.stores_t.p}.get(comp)
        opex = 0.0
        if flow is not None:
            cols = [c for c in df.index if c in flow.columns]
            per = flow[cols].mul(w, axis=0).mul(df.loc[cols, "marginal_cost"], axis=1).sum()
            opex = per.sum()
            top.append(pd.DataFrame({"component": comp, "carrier": df.loc[cols, "carrier"], "opex": per}))
        rows.append({"component": comp, "capex": capex, "capex_on_min_capacity": capex_min,
                     "capex_above_min": capex - capex_min, "opex": opex})
    t = pd.DataFrame(rows).set_index("component")
    t.loc["TOTAL"] = t.sum()
    print(f"\n=== {path}")
    print(t.round(0).to_string())
    mine = t.loc["TOTAL", "capex_above_min"] + t.loc["TOTAL", "opex"]
    print(f"\nmine (capex above min + opex): {mine:,.0f}")
    print(f"n.objective                  : {n.objective:,.0f}")
    print(f"gap (mine - objective)       : {mine - n.objective:,.0f}")
    print("n.objective_constant         :", getattr(n, "objective_constant", "n/a"))
    try:
        s = n.statistics
        print(f"statistics capex+opex        : {(s.capex().sum() + s.opex().sum()):,.0f}")
    except Exception as e:           # noqa: BLE001
        print("statistics failed:", e)
    big = pd.concat(top).groupby(["component", "carrier"]).opex.sum()
    big = big[big.abs() > 1e3].sort_values(key=abs, ascending=False)
    print("\nlargest opex terms (negative = revenue / sign convention):")
    print(big.head(12).round(0).to_string())


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)