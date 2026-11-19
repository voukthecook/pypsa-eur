# SPDX-License-Identifier: MIT
"""Validate that the profile shapes were implemented correctly, from two solved networks.

Terminal use (from the repository root)::

    python -m scripts.lv_split.validate_shapes CONTROL.nc SHAPED.nc \
        --table data/lv_profile_shapes/lpg_electricity.csv [--out validate_out] \
        [--timezone GR=Europe/Athens]

CONTROL is the run without shapes (split or unsplit), SHAPED the run with shapes.
The household demand of the shaped run is compared with what the shape table says it
should look like, read in the country's local time, in four groups of metrics:

A. Conservation: nothing is created or lost (node total per snapshot, yearly energy
   per profile, loads other than households).
B. Fidelity: the profile curves follow the table (snapshot by snapshot, as profile
   ratios, and as a daily pattern in local hours, peak hour included).
C. Clock: reading the table in local time fits better than reading it in UTC. This
   checks that the time-zone rule is really active in the run.
D. Effect: how different the profiles are and how much the peaks stop coinciding.

Exit code 1 if any check fails. Tables are written as CSV (and a PNG if matplotlib is
available) into ``--out``.
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
    _diagnose,
    _profile_of,
    _weights,
    household_columns,
    load_series,
)
from scripts.lv_split.local_time_shapes import country_timezone, resolve_shape

RANK_MIN = 0.95  # snapshot-level agreement with the table
CURVE_MIN = 0.90  # daily pattern agreement
CLOCK_TOLERANCE = 0.01  # local-time fit may be at most this much worse than UTC


def _by_base(frame: pd.DataFrame) -> pd.DataFrame:
    keys = pd.Series({c: _base_of(c) for c in frame.columns})
    return frame.T.groupby(keys).sum().T


def _circular_gap(a: int, b: int) -> int:
    d = abs(int(a) - int(b)) % 24
    return min(d, 24 - d)


def _expected_factor(raw: pd.DataFrame, share: pd.Series) -> pd.DataFrame:
    """Factor the table alone implies: raw / (share-weighted mean of raw), mean 1."""
    active = [p for p in PROFILES if p in share.index and share[p] > 0]
    mix = sum(share[p] * raw[p] for p in active) / sum(share[p] for p in active)
    return raw[active].div(mix, axis=0)


def _agreement(actual: pd.DataFrame, expected: pd.DataFrame) -> dict[str, float]:
    """Correlation of the profile factors with the table, per profile."""
    return {p: float(actual[p].corr(expected[p])) for p in actual.columns}


def _ratio_agreement(actual: pd.DataFrame, expected: pd.DataFrame) -> dict[str, tuple]:
    """Profile-to-profile log ratios versus the table: Pearson, Spearman, residual std."""
    cols = list(actual.columns)
    if len(cols) < 2:
        return {}
    ref = "rural" if "rural" in cols else cols[-1]
    out = {}
    for p in cols:
        if p == ref:
            continue
        a = np.log(actual[p] / actual[ref])
        e = np.log(expected[p] / expected[ref])
        resid = (a - e) - (a - e).mean()
        out[f"{p}/{ref}"] = (
            float(a.corr(e)),
            float(a.corr(e, method="spearman")),
            float(resid.std()),
        )
    return out


def _local_hours(index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    return index.tz_localize("UTC").tz_convert(tz)


def validate(
    control: pypsa.Network,
    shaped: pypsa.Network,
    table: pd.DataFrame,
    carriers: list[str] | None = None,
    timezones: dict[str, str] | None = None,
) -> dict:
    carriers = carriers or ["electricity"]
    pc, ps = load_series(control), load_series(shaped)
    if not pc.index.equals(ps.index):
        raise ValueError("The two networks have different snapshots.")
    w = _weights(control)
    hh = household_columns(shaped, carriers)
    if not hh:
        raise ValueError(_diagnose(shaped, carriers))
    table = table.set_index(["month", "daytype", "hour"])

    tot_c, tot_s = _by_base(pc), _by_base(ps)
    if set(tot_c.columns) != set(tot_s.columns):
        raise ValueError("Loads differ between the two networks.")
    tot_s = tot_s[tot_c.columns]
    base_of = pd.Series({c: _base_of(c) for c in hh})
    prof_of = pd.Series({c: _profile_of(c) for c in hh})
    bases = sorted(set(base_of))
    country_of = pd.Series(
        {c: str(shaped.loads.at[c, "bus"])[:2].upper() for c in hh}
    )
    checks: list[tuple[str, str, bool, str]] = []  # (group, name, ok, detail)

    # ---- A. conservation -------------------------------------------------------
    scale = tot_c[bases].abs().max().replace(0, np.nan)
    rel = ((tot_s[bases] - tot_c[bases]).abs().max() / scale).fillna(0)
    checks.append(
        ("A", "Household node total unchanged in every snapshot", bool(rel.max() < 1e-6),
         f"largest relative difference {rel.max():.2e} over {len(bases)} nodes")
    )
    other = [b for b in tot_c.columns if b not in bases]
    if other:
        scale = tot_c[other].abs().max().replace(0, np.nan)
        rel_o = ((tot_s[other] - tot_c[other]).abs().max() / scale).fillna(0)
        checks.append(
            ("A", "Other loads (EV, heat, ...) unchanged", bool(rel_o.max() < 1e-6),
             f"largest relative difference {rel_o.max():.2e} over {len(other)} loads")
        )
    energy = ps[hh].mul(w, axis=0).sum()
    share = (energy / energy.groupby(base_of).transform("sum").replace(0, np.nan)).fillna(0)
    spread = share.groupby(prof_of).agg(lambda s: s.max() - s.min())
    checks.append(
        ("A", "Each profile has the same yearly share at every node",
         bool(spread.max() < 1e-6),
         "shares " + ", ".join(
             f"{p} {share[prof_of == p].mean():.3f}" for p in PROFILES if (prof_of == p).any()))
    )
    if any(_profile_of(c) for c in pc.columns):
        hh_c = [c for c in hh if c in pc.columns]
        ratio = (energy[hh_c] / pc[hh_c].mul(w, axis=0).sum().replace(0, np.nan)).dropna()
        checks.append(
            ("A", "Yearly energy per profile unchanged versus control",
             bool((ratio - 1).abs().max() < 1e-6),
             f"energy ratio {ratio.min():.6f} to {ratio.max():.6f}")
        )

    # ---- B/C per country ----------------------------------------------------------
    ref = pd.DataFrame({c: share[c] * tot_c[base_of[c]] for c in hh})
    rows, curves, implied = [], {}, {}
    for country in sorted(country_of.unique()):
        tz = country_timezone(country, timezones)
        cols = list(country_of.index[country_of == country])
        profiles = [p for p in PROFILES if (prof_of[cols] == p).any()]
        actual = pd.DataFrame(
            {
                p: ps[[c for c in cols if prof_of[c] == p]].sum(axis=1)
                / ref[[c for c in cols if prof_of[c] == p]].sum(axis=1).where(lambda s: s > 0)
                for p in profiles
            }
        ).dropna()
        c_share = pd.Series(
            {p: share[[c for c in cols if prof_of[c] == p]].mean() for p in profiles}
        )
        views = {}
        for label, over in (("local", timezones), ("utc", {**(timezones or {}), country: "UTC"})):
            raw = resolve_shape(ps.index, w, table, [country], over)[country]
            views[label] = _expected_factor(raw.loc[actual.index], c_share)
        local, utc = views["local"], views["utc"]

        corr_local, corr_utc = _agreement(actual, local), _agreement(actual, utc)
        ratios = _ratio_agreement(actual, local)
        # slope of (model - 1) on (table - 1): the share of the load that follows the shape
        slopes = {
            p: float(np.polyfit(local[p] - 1.0, actual[p] - 1.0, 1)[0])
            for p in actual.columns
        }
        implied[country] = float(np.mean(list(slopes.values())))
        for p in actual.columns:
            rows.append(
                dict(country=country, timezone=tz, profile=p, kind="factor",
                     corr_local=corr_local[p], corr_utc=corr_utc[p],
                     slope_vs_table=slopes[p])
            )
        for pair, (pear, spear, resid) in ratios.items():
            rows.append(
                dict(country=country, timezone=tz, profile=pair, kind="log_ratio",
                     corr_local=pear, spearman_local=spear, residual_std=resid)
            )

        worst = min(corr_local.values())
        checks.append(
            ("B", f"[{country}] Profile curves follow the table snapshot by snapshot",
             worst > RANK_MIN,
             ", ".join(f"{p} {v:.3f}" for p, v in corr_local.items()))
        )
        if ratios:
            low = min(v[1] for v in ratios.values())
            checks.append(
                ("B", f"[{country}] Profile-to-profile ratios follow the table (rank)",
                 low > RANK_MIN,
                 ", ".join(f"{k} {v[1]:.3f}" for k, v in ratios.items()))
            )

        # daily pattern in local clock hours (weekdays)
        loc = _local_hours(actual.index, tz)
        weekday = np.asarray(loc.dayofweek < 5)
        step = int(round(float(w.median())))
        gaps, curve_corr = {}, {}
        for p in actual.columns:
            a = actual[p][weekday].groupby(loc.hour[weekday]).mean()
            e = local[p][weekday].groupby(loc.hour[weekday]).mean()
            curves[(country, p)] = pd.DataFrame({"achieved": a, "table": e})
            curve_corr[p] = float(a.corr(e))
            gaps[p] = (int(a.idxmax()), int(e.idxmax()))
        checks.append(
            ("B", f"[{country}] Daily pattern (local hours, weekdays) follows the table",
             min(curve_corr.values()) > CURVE_MIN,
             ", ".join(f"{p} {v:.3f}" for p, v in curve_corr.items()))
        )
        checks.append(
            ("B", f"[{country}] Peak hour matches the table within one snapshot step",
             all(_circular_gap(a, e) <= step for a, e in gaps.values()),
             ", ".join(f"{p} {a:02d}h vs {e:02d}h" for p, (a, e) in gaps.items()))
        )

        # weekend versus weekday direction of the urban/rural gap
        if ratios and "urban/rural" in ratios and (~weekday).any():
            la = np.log(actual["urban"] / actual["rural"])
            le = np.log(local["urban"] / local["rural"])
            da = la[~weekday].mean() - la[weekday].mean()
            de = le[~weekday].mean() - le[weekday].mean()
            checks.append(
                ("B", f"[{country}] Weekend vs weekday change in urban/rural gap has the table's sign",
                 abs(de) < 1e-6 or np.sign(da) == np.sign(de),
                 f"achieved {da:+.3f}, table {de:+.3f}")
            )

        # C. clock: local-time reading should fit at least as well as UTC reading
        gain = {p: corr_local[p] - corr_utc[p] for p in actual.columns}
        offset = (pd.Timestamp("2013-01-15", tz=tz).utcoffset().total_seconds() / 3600)
        checks.append(
            ("C", f"[{country}] Local-time reading fits at least as well as UTC reading",
             min(gain.values()) > -CLOCK_TOLERANCE,
             f"UTC{offset:+.0f} in winter; gain in correlation "
             + ", ".join(f"{p} {v:+.3f}" for p, v in gain.items()))
        )

    # ---- D. effect ------------------------------------------------------------------
    unit = pd.DataFrame({c: ps[c] / ref[c].where(ref[c] > 0) for c in hh}).astype(float)
    by_profile = unit.T.groupby(prof_of).mean().T
    gap = float((by_profile.max(axis=1) - by_profile.min(axis=1)).max())
    checks.append(
        ("D", "Profiles follow different shapes", gap > 0.01,
         f"largest gap between profile factors in one snapshot {gap:.3f}")
    )
    node_peak = tot_c[bases].max()
    node_peak = float(node_peak.sum())
    prof_peaks = {
        p: float(ps[[c for c in hh if prof_of[c] == p]].sum(axis=1).max())
        for p in PROFILES if (prof_of == p).any()
    }
    # per node peaks summed, to avoid mixing nodes at different hours
    sum_of_peaks = float(
        sum(
            ps[[c for c in hh if prof_of[c] == p and base_of[c] == b]].sum(axis=1).max()
            for b in bases for p in prof_peaks
        )
    )
    effect = pd.DataFrame(
        {
            "node_peak_MW": [node_peak],
            "sum_of_profile_peaks_MW": [sum_of_peaks],
            "non_coincidence_pct": [100 * (sum_of_peaks / node_peak - 1)],
        }
    )
    fidelity = pd.DataFrame(rows)
    return dict(
        checks=checks,
        fidelity=fidelity,
        effect=effect,
        curves=curves,
        profile_factor_stats=by_profile.describe().T,
        implied_fraction=implied,
    )


def _print(result: dict, out: Path) -> bool:
    out.mkdir(parents=True, exist_ok=True)
    print("\nCHECKS")
    groups = {
        "A": "A. Conservation",
        "B": "B. Fidelity to the table",
        "C": "C. Local clock",
        "D": "D. Effect",
    }
    last = None
    for group, name, ok, detail in result["checks"]:
        if group != last:
            print(f"  {groups[group]}")
            last = group
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print("\nFIDELITY DETAIL (corr_local = agreement with the table in local time)")
    print(result["fidelity"].round(3).to_string(index=False))
    print("\nSHARE OF THE ELECTRICITY LOAD THAT FOLLOWS THE SHAPE (implied by the model)")
    print("  slope of (model factor - 1) on (table factor - 1); 1.0 = whole load shaped")
    for country, value in result["implied_fraction"].items():
        print(f"  {country}: {value:.2f}")
    print("\nEFFECT OF THE SHAPES ON PEAKS (per node, summed)")
    print(result["effect"].round(1).to_string(index=False))
    result["fidelity"].to_csv(out / "shape_fidelity.csv", index=False)
    result["effect"].to_csv(out / "shape_effect.csv", index=False)
    result["profile_factor_stats"].to_csv(out / "profile_factor_stats.csv")
    pd.concat(result["curves"], names=["country", "profile", "local_hour"]).to_csv(
        out / "daily_curves.csv"
    )
    pd.DataFrame(
        [(g, n, ok, d) for g, n, ok, d in result["checks"]],
        columns=["group", "check", "pass", "detail"],
    ).to_csv(out / "shape_checks.csv", index=False)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        countries = sorted({k[0] for k in result["curves"]})
        fig, axes = plt.subplots(1, len(countries), figsize=(6 * len(countries), 4), squeeze=False)
        for ax, country in zip(axes[0], countries):
            for (c, p), frame in result["curves"].items():
                if c != country:
                    continue
                line = ax.plot(frame.index, frame["achieved"], label=f"{p} (model)")[0]
                ax.plot(frame.index, frame["table"], "--", color=line.get_color(),
                        label=f"{p} (table)")
            ax.set_title(f"{country}: weekday, local hour")
            ax.set_xlabel("local hour")
            ax.set_ylabel("factor")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "daily_curves.png", dpi=130)
        print(f"\nPlot written to {out / 'daily_curves.png'}")
    except Exception as exc:  # plotting is optional
        print(f"\n(no plot: {exc})")
    print(f"CSV files written to {out}/")
    return all(ok for _, _, ok, _ in result["checks"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("control", help="solved network, shapes off")
    parser.add_argument("shaped", help="solved network, shapes on")
    parser.add_argument("--table", required=True, help="shape table CSV used in the run")
    parser.add_argument("--out", default="validate_out", help="output folder")
    parser.add_argument("--carriers", nargs="+", default=["electricity"])
    parser.add_argument(
        "--timezone", action="append", default=[], metavar="CC=ZONE",
        help="override a country's timezone (as in sector.lv_split_profiles.timezones)",
    )
    args = parser.parse_args(argv)
    overrides = dict(item.split("=", 1) for item in args.timezone) or None
    result = validate(
        pypsa.Network(args.control),
        pypsa.Network(args.shaped),
        pd.read_csv(args.table),
        carriers=args.carriers,
        timezones=overrides,
    )
    return 0 if _print(result, Path(args.out)) else 1


if __name__ == "__main__":
    sys.exit(main())
