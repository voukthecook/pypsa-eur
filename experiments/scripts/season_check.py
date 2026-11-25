"""What happens in each season of a solved year run?  (Is PV higher in summer, heat higher in winter...)

Usage:  python -m experiments.scripts.season_check network.nc [other.nc ...]
Prints monthly tables (GWh unless stated) and writes to experiments/results/season_check/
  1  supply by carrier group
  2  PV: installed capacity, availability (p_max_pu x capacity) vs dispatched, curtailment, capacity factor
  3  demand: electricity, heat by system, EV
  4  heat supply by technology
  5  conversion links: DAC, heat pumps, resistive heaters, electrolysis
  6  storage: HV battery / home battery / EV throughput
  7  distribution net import, mean LV and AC price
Each section is wrapped in try/except so one failure does not stop the rest.
"""
import os
import sys
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pypsa

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
OUTPUT_DIR = Path("experiments/results/season_check")


def group_gen(c):
    c = str(c).lower()
    if "rooftop" in c:
        return "rooftop PV"
    if c.startswith("solar"):
        return "utility PV"
    if "wind" in c:
        return "wind"
    if c in ("ror", "hydro"):
        return "hydro"
    return "other (" + c + ")"


def monthly(df, w, idx):
    return df.mul(w, axis=0).groupby(idx.month).sum() / 1e3


def run(path):
    name = os.path.splitext(os.path.basename(path))[0]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n = pypsa.Network(path)
    idx = n.snapshots
    if not isinstance(idx, pd.DatetimeIndex):
        sys.exit("snapshots are not datetimes; this script needs a full-year run")
    w = n.snapshot_weightings.generators
    days = pd.Series(w.groupby(idx.month).sum() / 24, name="days")
    print(f"\n######## {path}\nsnapshots {idx[0]} .. {idx[-1]} ({len(idx)}), weights sum {w.sum():.0f} h")
    out = {}

    def sec(title, fn):
        print(f"\n==== {title} ====")
        try:
            t = fn()
            if t is not None:
                print(t.round(1).to_string())
                out[title.split()[0]] = t
        except Exception as e:                 # noqa: BLE001
            print(f"FAILED: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)

    g = n.generators
    gg = g.carrier.map(group_gen)

    def s1():
        p = n.generators_t.p.T.groupby(gg).sum().T
        t = monthly(p, w, idx)
        t["total"] = t.sum(axis=1)
        return t
    sec("1 supply by generator group (GWh)", s1)

    def s2():
        rows = {}
        for grp in ["rooftop PV", "utility PV"]:
            cols = g.index[gg == grp]
            cols = [c for c in cols if c in n.generators_t.p.columns]
            if not cols:
                continue
            cap = g.loc[cols, "p_nom_opt"]
            pmax = n.generators_t.p_max_pu[[c for c in cols if c in n.generators_t.p_max_pu.columns]]
            avail = pmax.mul(cap[pmax.columns], axis=1).sum(axis=1)
            disp = n.generators_t.p[cols].sum(axis=1)
            cf_av = avail.groupby(idx.month).mean() / max(cap.sum(), 1e-9)
            cf_di = disp.groupby(idx.month).mean() / max(cap.sum(), 1e-9)
            rows[f"{grp} capacity MW"] = pd.Series(cap.sum(), index=range(1, 13))
            rows[f"{grp} available GWh"] = monthly(avail.to_frame("a"), w, idx)["a"]
            rows[f"{grp} dispatched GWh"] = monthly(disp.to_frame("a"), w, idx)["a"]
            rows[f"{grp} curtailed %"] = 100 * (1 - rows[f"{grp} dispatched GWh"] / rows[f"{grp} available GWh"].where(lambda s: s > 0))
            rows[f"{grp} CF available %"] = 100 * cf_av
            rows[f"{grp} CF dispatched %"] = 100 * cf_di
            rows[f"{grp} p_max_pu min/max in month"] = pmax.groupby(idx.month).max().max(axis=1)
            pot = g.loc[cols, "p_nom_max"].replace(np.inf, np.nan).sum()
            print(f"{grp}: capacity {cap.sum():,.0f} MW, p_nom_max (potential) {pot:,.0f} MW, "
                  f"at-potential share {100 * cap.sum() / pot:.1f}%" if pot else f"{grp}: capacity {cap.sum():,.0f} MW")
        t = pd.DataFrame(rows)
        t.index = range(1, 13)
        return t
    sec("2 PV capacity, availability and dispatch", s2)

    def s3():
        ld = n.loads_t.p.T.groupby(n.loads.carrier).sum().T
        t = monthly(ld, w, idx)
        t["total"] = t.sum(axis=1)
        return t
    sec("3 demand by load carrier (GWh)", s3)

    def s4():
        l = n.links
        heat_buses = n.buses.index[n.buses.carrier.str.contains("heat", case=False, na=False)]
        res = {}
        for port in ("1", "2", "3"):
            if f"bus{port}" not in l.columns:
                continue
            flow = getattr(n.links_t, f"p{port}")
            for lk in l.index[l[f"bus{port}"].isin(heat_buses)]:
                if lk in flow.columns:
                    res[lk] = -flow[lk]
        if not res:
            return None
        f = pd.DataFrame(res)
        grp = f.T.groupby([n.links.loc[c, "carrier"] for c in f.columns]).sum().T
        t = monthly(grp, w, idx)
        t = t.loc[:, t.abs().sum() > 1]
        t["total"] = t.sum(axis=1)
        return t
    sec("4 heat delivered to heat buses by link carrier (GWh)", s4)

    def s5():
        cars = ["DAC", "urban central air heat pump", "urban decentral air heat pump", "rural air heat pump",
                "urban central resistive heater", "urban decentral resistive heater", "rural resistive heater",
                "H2 Electrolysis", "BEV charger", "V2G"]
        rows = {}
        for c in cars:
            lk = n.links.index[n.links.carrier == c]
            if len(lk):
                rows[c + " (input)"] = monthly(n.links_t.p0[lk].sum(axis=1).to_frame("a"), w, idx)["a"]
        return pd.DataFrame(rows)
    sec("5 conversion links: energy drawn at bus0 (GWh)", s5)

    def s6():
        rows = {}
        for car in ["battery charger", "battery discharger", "home battery charger", "home battery discharger", "BEV charger", "V2G"]:
            lk = n.links.index[n.links.carrier == car]
            if len(lk):
                rows[car] = monthly(n.links_t.p0[lk].sum(axis=1).to_frame("a"), w, idx)["a"]
        for car in ["battery", "home battery", "EV battery"]:
            st = n.stores.index[n.stores.carrier == car]
            if len(st):
                rows[car + " mean SOC %"] = 100 * (n.stores_t.e[st].sum(axis=1).groupby(idx.month).mean()
                                                  / max(n.stores.loc[st, "e_nom_opt"].sum(), 1e-9))
        return pd.DataFrame(rows)
    sec("6 storage activity (GWh through chargers; SOC %)", s6)

    def s7():
        dl = n.links[n.links.carrier == "electricity distribution grid"]
        fwd = dl.index[n.buses.loc[dl.bus1, "carrier"].eq("low voltage").values]
        rev = dl.index[n.buses.loc[dl.bus0, "carrier"].eq("low voltage").values]
        net = n.links_t.p1[fwd].sum(axis=1) * -1 - n.links_t.p0[rev].sum(axis=1)
        mp = n.buses_t.marginal_price
        lv = n.buses.index[n.buses.carrier == "low voltage"]
        ac = n.buses.index[n.buses.carrier == "AC"]
        t = pd.DataFrame({"net import GWh": monthly(net.to_frame("a"), w, idx)["a"],
                          "peak net import MW": net.groupby(idx.month).max(),
                          "mean LV price": mp[lv].mean(axis=1).groupby(idx.month).mean(),
                          "mean AC price": mp[ac].mean(axis=1).groupby(idx.month).mean(),
                          "max AC price": mp[ac].mean(axis=1).groupby(idx.month).max()})
        return t
    sec("7 distribution and prices", s7)

    out = {k: t.reindex(range(1, 13)) for k, t in out.items()}
    for k, t in out.items():
        t.to_csv(OUTPUT_DIR / f"{name}_{k}.csv")

    try:
        fig, ax = plt.subplots(2, 3, figsize=(17, 9))
        if "1" in out:
            out["1"].drop(columns="total").plot.bar(stacked=True, ax=ax[0, 0], width=.8)
            ax[0, 0].set_title("Generation by group (GWh)")
            ax[0, 0].legend(fontsize=6)
        if "2" in out:
            t = out["2"]
            for c in [c for c in t.columns if "CF" in c]:
                ax[0, 1].plot(range(12), t[c].values, marker="o", ls="--" if "dispatched" in c else "-", label=c)
            ax[0, 1].set_title("PV capacity factor by month (%)")
            ax[0, 1].legend(fontsize=6)
        if "3" in out:
            out["3"].drop(columns="total").plot.bar(stacked=True, ax=ax[0, 2], width=.8)
            ax[0, 2].set_title("Demand (GWh)")
            ax[0, 2].legend(fontsize=6)
        if "4" in out:
            out["4"].drop(columns="total").plot.bar(stacked=True, ax=ax[1, 0], width=.8)
            ax[1, 0].set_title("Heat delivered by technology (GWh)")
            ax[1, 0].legend(fontsize=6)
        if "5" in out:
            out["5"].plot(ax=ax[1, 1], marker="o")
            ax[1, 1].set_title("Conversion links, energy in (GWh)")
            ax[1, 1].legend(fontsize=6)
        if "7" in out:
            a = ax[1, 2]
            a.bar(range(12), out["7"]["net import GWh"].values, color="#9AA3AE")
            b = a.twinx()
            b.plot(range(12), out["7"]["mean AC price"].values, color="#E0782B", marker="o")
            a.set_title("Net import (bars, GWh) and mean AC price (line)")
        for a in ax.ravel():
            a.set_xticks(range(12), MON, fontsize=7)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{name}_seasons.png", dpi=130)
    except Exception as e:                       # noqa: BLE001
        print("figure failed:", e)
    print(f"\nwrote {OUTPUT_DIR}/{name}_*.csv and {name}_seasons.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        run(p)