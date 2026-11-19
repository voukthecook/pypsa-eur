# SPDX-License-Identifier: MIT
"""Compare a shapes-off control run with a shapes-on run of the LV profile split.

Terminal use (from the repository root)::

    python -m scripts.lv_split.compare_runs CONTROL.nc SHAPED.nc \
        [--table data/lv_profile_shapes/lpg_electricity.csv] [--out compare_out]

Both networks must come from the same config except for ``shapes`` /
``household_fraction``. Prints PASS/FAIL checks and summary tables and writes them
as CSV files into ``--out``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

PROFILES = ("urban", "suburban", "rural")
DISTRIBUTION_CARRIER = "electricity distribution grid"
CHARGER_CARRIER = "BEV charger"


def _profile_of(name: str) -> str | None:
    last = name.rsplit(" ", 1)[-1]
    return last if last in PROFILES else None


def _base_of(name: str) -> str:
    return name.rsplit(" ", 1)[0] if _profile_of(name) else name


def load_series(n: pypsa.Network) -> pd.DataFrame:
    """Load demand per load (snapshots x loads); static values fill missing series."""
    series = pd.DataFrame(
        {name: float(p) for name, p in n.loads["p_set"].items()}, index=n.snapshots
    )
    dynamic = n.loads_t.p_set
    series[dynamic.columns] = dynamic
    return series


def _weights(n: pypsa.Network) -> pd.Series:
    return n.snapshot_weightings.generators


def household_columns(n: pypsa.Network, carriers: list[str]) -> list[str]:
    loads = n.loads
    keep = loads.index[loads.carrier.isin(carriers)]
    return [c for c in keep if _profile_of(c)]


def _diagnose(n: pypsa.Network, carriers: list[str]) -> str:
    """Explain why no household loads were found, using what is in the network."""
    loads = n.loads
    split = loads[[bool(_profile_of(i)) for i in loads.index]]
    split_buses = [b for b in n.buses.index if _profile_of(b)]
    lines = [
        f"No loads with carrier {carriers} and a profile suffix "
        f"({', '.join(PROFILES)}) found.",
        f"  loads in network: {len(loads)}; with profile suffix: {len(split)}; "
        f"buses with profile suffix: {len(split_buses)}",
        f"  load carriers: {loads.carrier.value_counts().to_dict()}",
    ]
    if len(split):
        lines.append(f"  carriers of profile-suffixed loads: "
                     f"{split.carrier.value_counts().to_dict()}")
        lines.append("  -> rerun with --carriers <one of these>")
    elif split_buses:
        lines.append("  buses are split but loads are not; sample load names: "
                     f"{list(loads.index[:5])}")
    else:
        lines.append("  nothing is split: lv_split_profiles was probably not enabled "
                     "for this run")
        lines.append(f"  sample load names: {list(loads.index[:5])}")
    return "\n".join(lines)


def _links(n: pypsa.Network) -> pd.DataFrame:
    """Distribution-grid and charger links with capacity, peak flow, link-hours at limit."""
    sel = n.links.index[n.links.carrier.isin([DISTRIBUTION_CARRIER, CHARGER_CARRIER])]
    if len(sel) == 0:
        return pd.DataFrame()
    w = _weights(n)
    opt = n.links.loc[sel, "p_nom_opt" if "p_nom_opt" in n.links else "p_nom"].astype(float)
    existing = n.links.loc[sel, "p_nom"].astype(float)
    extendable = n.links.loc[sel, "p_nom_extendable"].astype(bool)
    added = (opt - existing).clip(lower=0).where(extendable, 0.0)
    flow = n.links_t.p0[n.links_t.p0.columns.intersection(sel)]
    peak = flow.abs().max().reindex(sel)
    full = ((flow.abs() >= 0.999 * opt[flow.columns]) & (opt[flow.columns] > 0)).mul(w, axis=0)
    return pd.DataFrame(
        {
            "carrier": n.links.loc[sel, "carrier"],
            "direction": ["reverse" if "reversed" in i else "forward" for i in sel],
            "profile": [_profile_of(i) or "node" for i in sel],
            "existing_MW": existing,
            "capacity_MW": opt,
            "added_MW": added,
            "peak_flow_MW": peak,
            "link_hours_at_limit": full.sum().reindex(sel),
        }
    )


def compare(
    control: pypsa.Network,
    shaped: pypsa.Network,
    carriers: list[str] | None = None,
    table: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame | list[tuple[str, bool, str]]]:
    """Compare a control run (split or not) with a shaped run (split)."""
    carriers = carriers or ["electricity"]
    pc, ps = load_series(control), load_series(shaped)
    if not pc.index.equals(ps.index):
        raise ValueError("The two networks have different snapshots.")
    w = _weights(control)
    control_split = any(_profile_of(c) for c in pc.columns)

    hh = household_columns(shaped, carriers)
    if not hh:
        raise ValueError(_diagnose(shaped, carriers))

    def by_base(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.T.groupby(pd.Series({c: _base_of(c) for c in frame.columns})).sum().T

    tot_c, tot_s = by_base(pc), by_base(ps)
    if set(tot_c.columns) != set(tot_s.columns):
        diff = sorted(set(tot_c.columns) ^ set(tot_s.columns))
        raise ValueError(f"Loads differ between the networks, e.g. {diff[:5]}")
    tot_s = tot_s[tot_c.columns]
    hh_bases = sorted({_base_of(c) for c in hh})
    checks: list[tuple[str, bool, str]] = []

    # 1. household demand per node (three profiles added up) equals the control
    scale = tot_c[hh_bases].abs().max().replace(0, np.nan)
    rel = ((tot_s[hh_bases] - tot_c[hh_bases]).abs().max() / scale).fillna(0)
    checks.append(
        (
            "Household node total unchanged in every snapshot",
            bool(rel.max() < 1e-6),
            f"largest relative difference {rel.max():.2e} over {len(hh_bases)} node loads",
        )
    )

    # 2. all other loads (EV, heat, industry, ...) add up to the control as well
    other = [b for b in tot_c.columns if b not in hh_bases]
    if other:
        scale = tot_c[other].abs().max().replace(0, np.nan)
        rel_o = ((tot_s[other] - tot_c[other]).abs().max() / scale).fillna(0)
        checks.append(
            (
                "Other loads (EV, heat, ...) unchanged after adding profiles up",
                bool(rel_o.max() < 1e-6),
                f"largest relative difference {rel_o.max():.2e} over {len(other)} node loads",
            )
        )

    # implied yearly share of each profile load = its energy / node energy
    base_of = pd.Series({c: _base_of(c) for c in hh})
    e_s = ps[hh].mul(w, axis=0).sum()
    e_base = e_s.groupby(base_of).transform("sum")
    share = (e_s / e_base.replace(0, np.nan)).fillna(0)
    prof = pd.Series({c: _profile_of(c) for c in hh})
    shares = share.groupby(prof).agg(["mean", "min", "max"])
    shares.index.name = "profile"

    if control_split:
        hh_c = [c for c in hh if c in pc.columns]
        e_c = pc[hh_c].mul(w, axis=0).sum()
        ratio = (e_s[hh_c] / e_c.replace(0, np.nan)).dropna()
        checks.append(
            (
                "Yearly energy per profile unchanged versus control",
                bool((ratio - 1).abs().max() < 1e-6),
                f"energy ratio range {ratio.min():.6f} to {ratio.max():.6f}",
            )
        )

    # 3. profiles follow different shapes: factor = profile demand / (share * node demand)
    ref = pd.DataFrame({c: share[c] * tot_c[base_of[c]] for c in hh})
    factor = (ps[hh] / ref.where(ref > 0)).astype(float)
    by_profile = factor.T.groupby(prof).mean().T
    spread = float((by_profile.max(axis=1) - by_profile.min(axis=1)).max())
    checks.append(
        (
            "Profiles follow different shapes",
            spread > 0.01,
            f"largest gap between profile factors in one snapshot: {spread:.3f}",
        )
    )

    if table is not None:
        keys = pd.MultiIndex.from_arrays(
            [
                control.snapshots.month,
                np.where(control.snapshots.dayofweek >= 5, "weekend", "weekday"),
                control.snapshots.hour,
            ]
        )
        raw = table.set_index(["month", "daytype", "hour"]).reindex(keys)
        raw.index = control.snapshots
        # The split rebalances the curves, so a profile's factor is not proportional to
        # its raw table value. The ratio between two profiles is (constants cancel),
        # so compare log ratios between profiles with the table's log ratios.
        corr = {}
        for p, q in (("urban", "rural"), ("suburban", "rural")):
            if p in by_profile and q in by_profile:
                actual = np.log(by_profile[p] / by_profile[q])
                expected = np.log(raw[p] / raw[q])
                corr[f"{p}/{q}"] = float(actual.corr(expected, method="spearman"))
        checks.append(
            (
                "Profile-to-profile ratios follow the shape table (rank correlation)",
                all(v > 0.9 for v in corr.values()),
                ", ".join(f"{k} {v:.2f}" for k, v in corr.items()),
            )
        )

    # per-load summary and per-profile peaks (shaped vs the same share without shape)
    node_peak = pd.Series({c: tot_c[base_of[c]].max() for c in hh}) * share
    summary = pd.DataFrame(
        {
            "share": share,
            "factor_min": factor.min(),
            "factor_mean": factor.mean(),
            "factor_max": factor.max(),
            "peak_unshaped_MW": node_peak,
            "peak_shaped_MW": ps[hh].max(),
            "peak_hour_unshaped": ref.idxmax().map(lambda t: t.hour),
            "peak_hour_shaped": ps[hh].idxmax().map(lambda t: t.hour),
        }
    )
    summary.index.name = "load"
    per_profile = summary.assign(profile=prof).groupby("profile").agg(
        factor_min=("factor_min", "min"),
        factor_mean=("factor_mean", "mean"),
        factor_max=("factor_max", "max"),
        peak_unshaped_MW=("peak_unshaped_MW", "sum"),
        peak_shaped_MW=("peak_shaped_MW", "sum"),
    )
    per_profile.loc["sum of profiles"] = per_profile[
        ["peak_unshaped_MW", "peak_shaped_MW"]
    ].sum().reindex(per_profile.columns)
    per_profile.loc["node total (one block)"] = [
        np.nan, np.nan, np.nan, float(node_peak.groupby(base_of).sum().sum()),
        float(tot_c[hh_bases].max().sum()),
    ]

    lc, ls = _links(control), _links(shaped)
    link_cols = ["existing_MW", "capacity_MW", "added_MW", "peak_flow_MW", "link_hours_at_limit"]
    links = {}
    if not lc.empty and not ls.empty:
        links["control"] = lc.groupby(["carrier", "direction", "profile"])[link_cols].sum()
        links["shaped"] = ls.groupby(["carrier", "direction", "profile"])[link_cols].sum()

    def _objective(n: pypsa.Network) -> float:
        try:
            return float(n.objective)
        except Exception:  # unsolved or missing
            return float("nan")

    return {
        "objective": (_objective(control), _objective(shaped)),
        "checks": checks,
        "shares": shares,
        "load_factors": summary,
        "profile_summary": per_profile,
        "links": links,
    }


def _print(result: dict, out: Path) -> bool:
    out.mkdir(parents=True, exist_ok=True)
    print("\nCHECKS")
    for name, ok, detail in result["checks"]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    for key in ("shares", "profile_summary", "load_factors"):
        result[key].to_csv(out / f"{key}.csv")
    print("\nYEARLY SHARE OF NODE HOUSEHOLD ENERGY PER PROFILE (mean, min, max over nodes)")
    print(result["shares"].round(4).to_string())
    print("\nPER PROFILE: factor = demand / (share x node demand); peaks summed over nodes")
    print("  'sum of profiles' > 'node total' means the profile peaks do not coincide")
    print(result["profile_summary"].round(3).to_string())
    links = result["links"]
    if links:
        for label, frame in links.items():
            frame.to_csv(out / f"links_{label}.csv")
            print(f"\nGRID LINKS, {label} run (summed over nodes)")
            print(frame.round(1).to_string())
        print("\nSUM OF LINK PEAKS AND CAPACITY PER CARRIER AND DIRECTION (control vs shaped)")
        print("  existing = fixed capacity, added = reinforcement chosen by the optimiser")
        total = pd.concat(
            {k: v.groupby(["carrier", "direction"])[
                ["existing_MW", "capacity_MW", "added_MW", "peak_flow_MW"]].sum()
             for k, v in links.items()}, axis=1,
        )
        print(total.round(1).to_string())
        total.to_csv(out / "links_total.csv")
    c_obj, s_obj = result.get("objective", (float("nan"),) * 2)
    if c_obj == c_obj and s_obj == s_obj:
        print(f"\nOBJECTIVE (total system cost): control {c_obj:,.0f}, shaped {s_obj:,.0f}, "
              f"difference {100 * (s_obj - c_obj) / abs(c_obj):+.3f}%")
    print(f"\nCSV files written to {out}/")
    return all(ok for _, ok, _ in result["checks"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("control", help="solved network, shapes off")
    parser.add_argument("shaped", help="solved network, shapes on")
    parser.add_argument("--table", help="shape table CSV, enables the table check")
    parser.add_argument("--out", default="compare_out", help="output folder")
    parser.add_argument("--carriers", nargs="+", default=["electricity"],
                        help="household load carriers (default: electricity)")
    args = parser.parse_args(argv)
    result = compare(
        pypsa.Network(args.control),
        pypsa.Network(args.shaped),
        carriers=args.carriers,
        table=pd.read_csv(args.table) if args.table else None,
    )
    return 0 if _print(result, Path(args.out)) else 1


if __name__ == "__main__":
    sys.exit(main())
