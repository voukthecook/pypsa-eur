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


def compare(
    control: pypsa.Network,
    shaped: pypsa.Network,
    carriers: list[str] | None = None,
    table: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame | list[tuple[str, bool, str]]]:
    carriers = carriers or ["electricity"]
    pc, ps = load_series(control), load_series(shaped)
    if not pc.columns.equals(ps.columns) and set(pc.columns) != set(ps.columns):
        missing = set(pc.columns) ^ set(ps.columns)
        raise ValueError(f"The two networks have different loads, e.g. {sorted(missing)[:5]}")
    ps = ps[pc.columns]
    if not pc.index.equals(ps.index):
        raise ValueError("The two networks have different snapshots.")
    w = _weights(control)

    hh = household_columns(control, carriers)
    if not hh:
        raise ValueError(
            "No profile-split household loads found. Was lv_split_profiles enabled?"
        )
    checks: list[tuple[str, bool, str]] = []

    # 1. node total of household demand must not change
    groups = pd.Series({c: _base_of(c) for c in hh})
    tot_c = pc[hh].T.groupby(groups).sum().T
    tot_s = ps[hh].T.groupby(groups).sum().T
    rel = ((tot_s - tot_c).abs().max() / tot_c.abs().max().replace(0, np.nan)).fillna(0)
    checks.append(
        (
            "Household node total unchanged in every snapshot",
            bool(rel.max() < 1e-6),
            f"largest relative difference {rel.max():.2e} over {len(tot_c.columns)} node loads",
        )
    )

    # 2. yearly energy per profile unchanged
    e_c, e_s = pc[hh].mul(w, axis=0).sum(), ps[hh].mul(w, axis=0).sum()
    ratio = (e_s / e_c.replace(0, np.nan)).dropna()
    checks.append(
        (
            "Yearly energy per profile unchanged (shares kept)",
            bool((ratio - 1).abs().max() < 1e-6),
            f"energy ratio range {ratio.min():.6f} to {ratio.max():.6f}",
        )
    )

    # 3. everything that is not a household load must be untouched
    other = [c for c in pc.columns if c not in hh]
    other_diff = float((ps[other] - pc[other]).abs().to_numpy().max()) if other else 0.0
    checks.append(
        (
            "Non-household loads untouched",
            other_diff < 1e-6,
            f"largest absolute difference {other_diff:.2e} over {len(other)} loads",
        )
    )

    # 4. shapes actually differ between profiles
    factor = (ps[hh] / pc[hh].where(pc[hh] > 0)).astype(float)
    by_profile = factor.T.groupby(pd.Series({c: _profile_of(c) for c in hh})).mean().T
    spread = float((by_profile.max(axis=1) - by_profile.min(axis=1)).max())
    checks.append(
        (
            "Profiles follow different shapes",
            spread > 0.01,
            f"largest gap between profile factors in one snapshot: {spread:.3f}",
        )
    )

    summary = pd.DataFrame(
        {
            "factor_min": factor.min(),
            "factor_mean": factor.mean(),
            "factor_max": factor.max(),
            "peak_control_MW": pc[hh].max(),
            "peak_shaped_MW": ps[hh].max(),
            "peak_hour_control": pc[hh].idxmax().map(lambda t: t.hour),
            "peak_hour_shaped": ps[hh].idxmax().map(lambda t: t.hour),
        }
    )
    summary.index.name = "load"
    per_profile = summary.assign(profile=[_profile_of(c) for c in summary.index])
    per_profile = per_profile.groupby("profile")[
        ["factor_min", "factor_mean", "factor_max", "peak_control_MW", "peak_shaped_MW"]
    ].agg({"factor_min": "min", "factor_mean": "mean", "factor_max": "max",
           "peak_control_MW": "sum", "peak_shaped_MW": "sum"})

    # 5. optional: do the factors follow the LPG table (rank correlation)?
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

    # 6. grid links: capacity, peak flow, hours at the limit
    links = {}
    for label, n in (("control", control), ("shaped", shaped)):
        sel = n.links.index[
            n.links.carrier.isin([DISTRIBUTION_CARRIER, CHARGER_CARRIER])
            & n.links.index.map(lambda x: _profile_of(x) is not None)
        ]
        if len(sel) == 0:
            continue
        cap_col = "p_nom_opt" if "p_nom_opt" in n.links else "p_nom"
        cap = n.links.loc[sel, cap_col]
        flow = n.links_t.p0[n.links_t.p0.columns.intersection(sel)]
        peak = flow.abs().max() if not flow.empty else pd.Series(np.nan, index=sel)
        at_limit = (
            ((flow.abs() >= 0.999 * cap[flow.columns]) & (cap[flow.columns] > 0))
            .mul(w, axis=0)
            .sum()
            if not flow.empty
            else pd.Series(np.nan, index=sel)
        )
        links[label] = pd.DataFrame(
            {"carrier": n.links.loc[sel, "carrier"], "capacity_MW": cap,
             "peak_flow_MW": peak, "hours_at_limit": at_limit}
        )
    link_table = (
        pd.concat(links, axis=1) if len(links) == 2 else pd.DataFrame()
    )
    if not link_table.empty:
        link_table.columns = [f"{a}_{b}" for a, b in link_table.columns]
        link_table = link_table.drop(columns=["shaped_carrier"]).rename(
            columns={"control_carrier": "carrier"}
        )
        link_table["profile"] = [_profile_of(i) for i in link_table.index]

    return {
        "checks": checks,
        "load_factors": summary,
        "profile_summary": per_profile,
        "links": link_table,
    }


def _print(result: dict, out: Path) -> bool:
    out.mkdir(parents=True, exist_ok=True)
    print("\nCHECKS")
    for name, ok, detail in result["checks"]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    for key in ("profile_summary", "load_factors", "links"):
        frame = result[key]
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frame.to_csv(out / f"{key}.csv")
    print("\nPER PROFILE (factor = shaped / control demand; peaks summed over nodes)")
    print(result["profile_summary"].round(3).to_string())
    links = result["links"]
    if isinstance(links, pd.DataFrame) and not links.empty:
        agg = links.groupby(["carrier", "profile"])[
            [c for c in links.columns if c.endswith(("_MW", "_limit"))]
        ].sum()
        print("\nGRID LINKS summed over nodes (capacity, peak flow, hours at limit)")
        print(agg.round(1).to_string())
        agg.to_csv(out / "links_by_profile.csv")
        total = links.groupby("carrier")[["control_capacity_MW", "shaped_capacity_MW"]].sum()
        print("\nTOTAL CAPACITY over the three profiles")
        print(total.round(1).to_string())
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
