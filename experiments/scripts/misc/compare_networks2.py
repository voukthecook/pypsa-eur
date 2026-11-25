"""Compare a solved UNSPLIT network with a solved 3-PROFILE SPLIT network
(lv_split_profiles: urban / suburban / rural). Everything is in % difference.
 
Usage:  python compare_networks.py unsplit.nc split.nc
Needs:  pip install pypsa matplotlib pandas
 
Weighting: each profile gets a "fair share" = its share x the unsplit total.
A profile's value is shown as % above/below that fair share, so profiles with
different shares (0.4 / 0.3 / 0.3) are directly comparable. EV profiles are
divided by the share, i.e. plotted at whole-system scale.
 
Carrier / bus names below are guesses -- adjust the CONFIG block if needed
(run with --debug to print the names the script finds).
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import pypsa
 
# ---- CONFIG (match config.yaml -> sector.lv_split_profiles) ----------------
SHARES = {"urban": 0.4, "suburban": 0.3, "rural": 0.3}      # demand / EV shares
PV_SHARES = {"urban": 0.4, "suburban": 0.3, "rural": 0.3}   # rooftop PV shares
DIST_CARRIER = "electricity distribution grid"   # links, AC -> LV (bus1 = LV)
EV_CHARGER = "BEV charger"                       # links, bus0 = LV, bus1 = EV battery
EV_V2G = "V2G"                                   # links, bus0 = EV battery, bus1 = LV
PV_CARRIER = "solar rooftop"                     # generators on LV buses
EV_STORE_KEYWORD = "EV battery"                  # stores sit on buses with this in the name
 
 
def profile_of(name):
    """'suburban' must be tested before 'urban' (substring)."""
    s = str(name).lower()
    for p in ("suburban", "urban", "rural"):
        if p in s:
            return p
    return "all"        # unsplit network / unmatched
 
 
def pct(new, old):
    return 100 * (new - old) / old if old else float("nan")
 
 
def summarise(path, debug=False):
    n = pypsa.Network(path)
    w = n.snapshot_weightings.generators
 
    def link_profile(df, first, second):
        """Profile from the first endpoint that names one (forward AND reversed
        distribution links both end up with their LV bus's profile)."""
        return [profile_of(a) if profile_of(a) != "all" else profile_of(b)
                for a, b in zip(df[first], df[second])]
 
    links = n.links
    dist = links[links.carrier == DIST_CARRIER].copy()
    dist["prof"] = link_profile(dist, "bus1", "bus0")
    chg = links[links.carrier == EV_CHARGER].copy()
    chg["prof"] = link_profile(chg, "bus0", "bus1")
    v2g = links[links.carrier == EV_V2G].copy()
    v2g["prof"] = link_profile(v2g, "bus1", "bus0")
    pv = n.generators[n.generators.carrier == PV_CARRIER].copy()
    pv["prof"] = pv.bus.map(profile_of)
 
    # EV store -> profile via the charger feeding its bus (names may not carry it)
    bus_prof = dict(zip(chg.bus1, chg.prof))
    stores = n.stores[n.stores.bus.str.contains(EV_STORE_KEYWORD)].copy()
    stores["prof"] = [bus_prof.get(b, profile_of(b)) for b in stores.bus]
 
    if debug:
        for label, df in [("dist", dist), ("charger", chg), ("v2g", v2g), ("pv", pv), ("ev stores", stores)]:
            print(f"[debug] {label}: {df.prof.value_counts().to_dict()}")
 
    profiles = sorted(set(dist.prof) | set(chg.prof) | set(pv.prof))
    rows = {}
    charge_ts = {}
    for p in profiles:
        c = n.links_t.p0[chg.index[chg.prof == p]].sum(axis=1)
        v = n.links_t.p1[v2g.index[v2g.prof == p]].sum(axis=1).abs()
        charge_ts[p] = c
        rows[p] = {
            "dist_cap_MW": dist.p_nom_opt[dist.prof == p].sum(),
            "pv_MW": pv.p_nom_opt[pv.prof == p].sum(),
            "ev_charge_GWh": (c * w).sum() / 1e3,
            "ev_v2g_GWh": (v * w).sum() / 1e3,
            "ev_peak_MW": c.max(),
        }
    per_prof = pd.DataFrame(rows).T
    costs = (n.statistics.capex() + n.statistics.opex()).groupby(level=1).sum()
    return n.objective, per_prof, pd.DataFrame(charge_ts), costs
 
 
if __name__ == "__main__":
    debug = "--debug" in sys.argv
    a_path, b_path = [x for x in sys.argv[1:] if not x.startswith("--")][:2]
    obj_a, pp_a, ts_a, cost_a = summarise(a_path, debug)
    obj_b, pp_b, ts_b, cost_b = summarise(b_path, debug)
 
    if "all" in pp_b.index:
        print("WARNING: some split-network assets match no profile; check names with --debug")
    print("\nSplit network, absolute per profile:")
    print(pp_b.round(1))
 
    tot_a = pp_a.sum()
    tot_a["ev_peak_MW"] = ts_a.sum(axis=1).max()
    tot_b = pp_b.sum()
    tot_b["ev_peak_MW"] = ts_b.sum(axis=1).max()     # coincident peak of all profiles
 
    # ---- totals: unsplit vs split
    tab = pd.DataFrame({"unsplit": tot_a, "split": tot_b})
    tab.loc["objective"] = [obj_a, obj_b]
    tab["diff_%"] = [pct(b, a) for a, b in zip(tab.unsplit, tab.split)]
    print(tab.round(2))
 
    # ---- per profile, weighted by share (% vs fair share of the unsplit total)
    sp = pp_b.drop(index="all", errors="ignore").copy()
    shares = pd.Series(SHARES)
    pvs = pd.Series(PV_SHARES)
    fair = pd.DataFrame({
        "dist_cap_MW": shares * tot_a["dist_cap_MW"],
        "pv_MW": pvs * tot_a["pv_MW"],
        "ev_charge_GWh": shares * tot_a["ev_charge_GWh"],
        "ev_v2g_GWh": shares * tot_a["ev_v2g_GWh"],
        "ev_peak_MW": shares * tot_a["ev_peak_MW"],
    }).loc[sp.index]
    weighted = 100 * (sp - fair) / fair
    print("\nSplit profiles, % vs fair share of unsplit (share-weighted):")
    print(weighted.round(2))
 
    # ---- costs by carrier (% change, carriers > 0.5% of unsplit total)
    cc = pd.concat([cost_a, cost_b], axis=1, keys=["unsplit", "split"]).fillna(0)
    cc = cc[cc.unsplit.abs() > 0.005 * cc.unsplit.abs().sum()]
    cc["diff_%"] = 100 * (cc.split - cc.unsplit) / cc.unsplit
    print("\nCost by carrier:")
    print(cc.round(2))


    # ---- plots
    fig, ax = plt.subplots(
        2, 2, figsize=(8, 5),
        gridspec_kw={"height_ratios": [1, 1.2]},
    )
    weighted[["dist_cap_MW", "pv_MW", "ev_charge_GWh", "ev_peak_MW"]].plot.bar(
        ax=ax[0, 0], title="Per profile: % vs fair share", fontsize=7)
    ax[0, 0].axhline(0, c="k", lw=.8)
    ax[0, 0].tick_params(axis="x", labelrotation=25, labelsize=7)
    ax[0, 0].legend(fontsize=6, ncol=2)
    cc["diff_%"].sort_values().plot.barh(
        ax=ax[0, 1], title="Cost by carrier: % change", fontsize=7)
    ax[0, 1].axvline(0, c="k", lw=.8)
    ax[0, 1].tick_params(axis="y", labelsize=6)
    ax[0, 1].set_ylabel("")
    ax[1, 0].remove()
    ax[1, 1].remove()
    ax_ts = fig.add_subplot(2, 1, 2)
    ts_a.sum(axis=1).plot(ax=ax_ts, c="k", lw=2, label="unsplit (total)")
    for p in sp.index:
        (ts_b[p] / SHARES[p]).plot(ax=ax_ts, lw=1, label=f"{p} / share")
    ax_ts.set_title("EV charging, whole-system scale (MW)")
    ax_ts.legend(fontsize=7, ncol=4)
    ax_ts.tick_params(axis="x", labelrotation=20, labelsize=7)
    fig.tight_layout()
    output_dir = Path("experiments/results/network_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "network_comparison.png", dpi=150)
    plt.show()