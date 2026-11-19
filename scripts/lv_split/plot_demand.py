# SPDX-License-Identifier: MIT
"""Plot consumption of shaped versus unshaped profiles, and its weight in the whole LV level.

Terminal use (from the repository root)::

    python -m scripts.lv_split.plot_demand CONTROL.nc SHAPED.nc [--out demand_out] [--country GR]

Figures (weekday averages in local time, 3-hour bins so that winter/summer clock
offsets do not zigzag):

1. ``household_profiles.png``  household electricity per profile, control vs shaped (MW)
2. ``lv_level.png``            everything consumed on the LV buses (household electricity,
                               EV charging, heating), control vs shaped, plus the difference
3. ``household_weeks.png``     the weeks with the highest and lowest household demand

Printed: how much of the LV consumption is household electricity (energy and at the peak)
and how the household effect carries through to the whole LV level. EV charging and heat
pumps are not shaped by the split, but they are optimised dispatch, so they can still
differ between the runs. Both are shown as they come out of the solved networks.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

from scripts.lv_split.compare_runs import (
    PROFILES,
    _base_of,
    _profile_of,
    _weights,
    load_series,
)
from scripts.lv_split.local_time_shapes import country_timezone

DISTRIBUTION = "electricity distribution grid"
CATEGORIES = ("household electricity", "EV charging", "heating", "other")


def _category(carrier: str) -> str:
    c = str(carrier).lower()
    if c == "bev charger":
        return "EV charging"
    if "heat pump" in c or "resistive heater" in c:
        return "heating"
    return "other"


def lv_consumption(n: pypsa.Network, carriers: list[str]) -> dict[str, pd.DataFrame]:
    """Consumption (MW) on each LV bus by category: {category: snapshots x LV buses}."""
    lv = list(n.buses.index[n.buses.carrier == "low voltage"])
    if not lv:
        raise ValueError("No buses with carrier 'low voltage' in the network.")
    out = {c: pd.DataFrame(0.0, index=n.snapshots, columns=lv) for c in CATEGORIES}
    loads = load_series(n)
    for name in n.loads.index[n.loads.bus.isin(lv) & n.loads.carrier.isin(carriers)]:
        out["household electricity"][n.loads.at[name, "bus"]] += loads[name]
    p0 = n.links_t.p0
    for name in n.links.index[n.links.bus0.isin(lv) & (n.links.carrier != DISTRIBUTION)]:
        if name in p0.columns:
            cat = _category(n.links.at[name, "carrier"])
            out[cat][n.links.at[name, "bus0"]] += p0[name].clip(lower=0)
    return out


def _by_profile(frame: pd.DataFrame, shares: pd.Series | None = None) -> pd.DataFrame:
    """Sum LV buses per profile; an unsplit network becomes one 'node' column."""
    key = pd.Series({b: _profile_of(b) or "node" for b in frame.columns})
    return frame.T.groupby(key).sum().T


def _by_node(frame: pd.DataFrame) -> pd.DataFrame:
    key = pd.Series({b: _base_of(b) for b in frame.columns})
    return frame.T.groupby(key).sum().T


def _weekday_curve(series: pd.DataFrame, tz: str, step: int) -> pd.DataFrame:
    local = series.index.tz_localize("UTC").tz_convert(tz)
    weekday = np.asarray(local.dayofweek < 5)
    bins = (local.hour // step) * step
    return series[weekday].groupby(bins[weekday]).mean()


def analyse(control: pypsa.Network, shaped: pypsa.Network, carriers, country=None) -> dict:
    cc, cs = lv_consumption(control, carriers), lv_consumption(shaped, carriers)
    if not cc["household electricity"].index.equals(cs["household electricity"].index):
        raise ValueError("The two networks have different snapshots.")
    w = _weights(shaped)
    countries = {b[:2].upper() for b in cs["household electricity"].columns}
    if country is None:
        if len(countries) != 1:
            raise ValueError(f"Several countries {sorted(countries)}; pass --country.")
        country = next(iter(countries))
    tz = country_timezone(country)
    step = max(1, int(round(float(w.median()))))

    hh_s = _by_profile(cs["household electricity"])
    hh_c = _by_profile(cc["household electricity"])
    if list(hh_c.columns) == ["node"]:
        # unsplit control: the plain-share version of each profile is share x node total
        energy = hh_s.mul(w, axis=0).sum()
        share = energy / energy.sum()
        hh_c = pd.DataFrame({p: share[p] * hh_c["node"] for p in hh_s.columns})
    hh_c = hh_c[hh_s.columns]

    totals_s = sum(cs[c] for c in CATEGORIES)
    totals_c = sum(cc[c] for c in CATEGORIES)
    node_c, node_s = _by_node(totals_c), _by_node(totals_s)

    def energy(frame):
        return float(frame.mul(w, axis=0).sum().sum())

    summary = pd.DataFrame(
        {
            run: {
                **{f"energy_share_{c}": energy(cats[c]) / max(energy(sum(cats[k] for k in CATEGORIES)), 1e-12)
                   for c in CATEGORIES},
                "LV_peak_MW": float(_by_node(sum(cats[k] for k in CATEGORIES)).max().sum()),
            }
            for run, cats in (("control", cc), ("shaped", cs))
        }
    )
    hh_peak_c = float(_by_node(cc["household electricity"]).max().sum())
    hh_peak_s_sum = float(cs["household electricity"].max().sum())
    lv_peak_c = float(node_c.max().sum())
    lv_peak_s_sum = float(totals_s.max().sum())  # sum of per-profile peaks
    effect = pd.DataFrame(
        {
            "household_non_coincidence_pct": [100 * (hh_peak_s_sum / hh_peak_c - 1)],
            "LV_total_non_coincidence_pct": [100 * (lv_peak_s_sum / lv_peak_c - 1)],
            "LV_node_peak_change_pct": [100 * (float(node_s.max().sum()) / lv_peak_c - 1)],
        }
    )
    return dict(
        tz=tz, step=step, country=country, w=w,
        hh_c=hh_c, hh_s=hh_s, cats_c=cc, cats_s=cs,
        summary=summary, effect=effect, totals_c=totals_c, totals_s=totals_s,
    )


def _plots(r: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tz, step = r["tz"], r["step"]
    colours = {"urban": "tab:blue", "suburban": "tab:orange", "rural": "tab:green",
               "node": "tab:gray"}

    # 1. household per profile
    cols = list(r["hh_s"].columns)
    fig, axes = plt.subplots(1, len(cols), figsize=(4.5 * len(cols), 3.8), sharey=False, squeeze=False)
    cc, cs = _weekday_curve(r["hh_c"], tz, step), _weekday_curve(r["hh_s"], tz, step)
    for ax, p in zip(axes[0], cols):
        ax.plot(cc.index, cc[p], "--", color=colours.get(p), label="unshaped (plain share)")
        ax.plot(cs.index, cs[p], "-", color=colours.get(p), label="shaped (LPG)")
        ax.set_title(f"{p}")
        ax.set_xlabel(f"local hour ({step}-h bins)")
        ax.set_ylabel("household electricity [MW]")
        ax.legend(fontsize=7)
    fig.suptitle(f"{r['country']}: household electricity, weekday average")
    fig.tight_layout()
    fig.savefig(out / "household_profiles.png", dpi=130)
    plt.close(fig)

    # 2. whole LV level
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    palette = {"household electricity": "tab:blue", "EV charging": "tab:green",
               "heating": "tab:red", "other": "tab:gray"}
    curves = {}
    for run, cats, ax in (("control", r["cats_c"], axes[0]), ("shaped", r["cats_s"], axes[1])):
        stack = {c: _weekday_curve(pd.DataFrame({"x": cats[c].sum(axis=1)}), tz, step)["x"]
                 for c in CATEGORIES}
        frame = pd.DataFrame(stack)
        curves[run] = frame
        frame = frame.loc[:, frame.abs().sum() > 0]
        ax.stackplot(frame.index, frame.T.to_numpy(), labels=list(frame.columns),
                     colors=[palette[c] for c in frame.columns])
        ax.set_title(f"{run}: consumption on LV buses")
        ax.set_xlabel(f"local hour ({step}-h bins)")
        ax.set_ylabel("MW")
        ax.legend(fontsize=7, loc="upper left")
    ymax = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(0, ymax)
    axes[1].set_ylim(0, ymax)
    diff = curves["shaped"] - curves["control"]
    for c in diff.columns:
        if diff[c].abs().sum() > 0:
            axes[2].plot(diff.index, diff[c], label=c, color=palette[c])
    axes[2].plot(diff.index, diff.sum(axis=1), "k-", lw=2, label="total")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_title("shaped minus control (all profiles together)")
    axes[2].set_xlabel(f"local hour ({step}-h bins)")
    axes[2].set_ylabel("MW")
    axes[2].legend(fontsize=7)
    fig.suptitle(f"{r['country']}: weekday average on the LV level")
    fig.tight_layout()
    fig.savefig(out / "lv_level.png", dpi=130)
    plt.close(fig)
    pd.concat(curves, axis=1).to_csv(out / "lv_level_weekday_curves.csv")

    # 3. highest and lowest weeks of household demand (per profile)
    total = r["hh_s"].sum(axis=1)
    weekly = total.groupby(total.index.to_period("W")).mean()
    weeks = {"highest-demand week": weekly.idxmax(), "lowest-demand week": weekly.idxmin()}
    fig, axes = plt.subplots(len(weeks), 1, figsize=(11, 3.6 * len(weeks)), squeeze=False)
    for ax, (label, wk) in zip(axes[:, 0], weeks.items()):
        mask = total.index.to_period("W") == wk
        for p in r["hh_s"].columns:
            ax.plot(r["hh_s"].index[mask], r["hh_s"][p][mask], color=colours.get(p), label=f"{p} shaped")
            ax.plot(r["hh_c"].index[mask], r["hh_c"][p][mask], "--", color=colours.get(p), alpha=0.8,
                    label=f"{p} unshaped")
        ax.set_title(f"{label} ({wk})")
        ax.set_ylabel("MW")
        ax.legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(out / "household_weeks.png", dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("control")
    parser.add_argument("shaped")
    parser.add_argument("--out", default="demand_out")
    parser.add_argument("--carriers", nargs="+", default=["electricity"])
    parser.add_argument("--country", help="country code if the network has several")
    args = parser.parse_args(argv)
    r = analyse(pypsa.Network(args.control), pypsa.Network(args.shaped), args.carriers, args.country)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("\nSHARE OF EACH CATEGORY IN THE ENERGY CONSUMED ON LV BUSES, AND LV PEAK")
    print(r["summary"].round(3).to_string())
    print("\nHOW THE HOUSEHOLD EFFECT CARRIES THROUGH TO THE WHOLE LV LEVEL")
    print("  non-coincidence = (sum of the profile peaks / control peak of one block) - 1")
    print(r["effect"].round(2).to_string(index=False))
    r["summary"].to_csv(out / "lv_summary.csv")
    r["effect"].to_csv(out / "lv_effect.csv", index=False)
    try:
        _plots(r, out)
        print(f"\nFigures written to {out}/ (household_profiles, lv_level, household_weeks)")
    except Exception as exc:
        print(f"\n(no figures: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
