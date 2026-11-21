# SPDX-License-Identifier: MIT
"""Turn pyLPG household simulations into the ``shapes`` table of the LV profile split.

Input : one CSV per (profile, household, season) with the minute-by-minute
        ``Electricity_HH1`` series from pyLPG (columns: timestamp, value).
        File names: ``<profile>__<household>__<season>.csv``, season in jan/apr/jul/oct.
        A household may be ``<template>-s<seed>`` (several random seeds of one template).
        An optional ``mix.json`` in the folder gives the template weights per profile.
Output: CSV with columns month, daytype, hour, urban, suburban, rural, to be given
        to ``sector.lv_split_profiles.shapes.electricity``.
        With ``--bootstrap N`` also ``<out>_bands.csv`` (5-95% band per cell),
        ``<out>_stability.csv`` (how often the ordering of the profiles holds) and
        ``<out>_bands.png``.

Steps: hourly energy per household -> average over weekday / weekend days per season
-> weighted average over the households of a profile -> each month takes its season ->
each profile is scaled to a mean of 1 (weekday and weekend weighted 5:2), so only
timing differs between profiles. The household mix per profile is an assumption.

Bootstrap: households of a profile are drawn with replacement (in proportion to their
weight) and the table is rebuilt; the spread of the rebuilt tables shows how much the
result depends on which households happen to be in the sample.
"""

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PROFILES = ("urban", "suburban", "rural")
SEASONS = ("jan", "apr", "jul", "oct")
SEASON_OF_MONTH = {
    12: "jan", 1: "jan", 2: "jan",
    3: "apr", 4: "apr", 5: "apr",
    6: "jul", 7: "jul", 8: "jul",
    9: "oct", 10: "oct", 11: "oct",
}


def household_hourly(path: Path) -> pd.Series:
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    hourly = s.resample("h").sum()
    df = pd.DataFrame({"v": hourly})
    df["daytype"] = ["weekend" if d >= 5 else "weekday" for d in df.index.dayofweek]
    df["hour"] = df.index.hour
    return df.groupby(["daytype", "hour"])["v"].mean()


def _template(household: str) -> str:
    return re.sub(r"-s\d+$", "", household)


def load_households(job_dir: Path):
    """Per profile: array (households, cells, seasons), weights, index of cells, names."""
    found: dict[tuple[str, str], dict[str, pd.Series]] = {}
    for path in sorted(job_dir.glob("*__*__*.csv")):
        profile, household, season = path.stem.split("__")
        found.setdefault((profile, household), {})[season] = household_hourly(path)
    if not found:
        raise FileNotFoundError(f"no simulation CSVs in {job_dir}")
    mix_file = job_dir / "mix.json"
    mix = json.loads(mix_file.read_text()) if mix_file.exists() else {}

    out = {}
    for profile in PROFILES:
        names = sorted(h for (p, h) in found if p == profile)
        names = [h for h in names if set(found[(profile, h)]) == set(SEASONS)]
        if not names:
            raise FileNotFoundError(f"no complete households for profile '{profile}'")
        cells = found[(profile, names[0])]["jan"].index
        data = np.stack(
            [
                np.column_stack([found[(profile, h)][s].reindex(cells).to_numpy() for s in SEASONS])
                for h in names
            ]
        )
        per_template = pd.Series([_template(h) for h in names]).value_counts()
        weights = np.array(
            [
                mix.get(profile, {}).get(_template(h), 1.0 / len(set(map(_template, names))))
                / per_template[_template(h)]
                for h in names
            ]
        )
        out[profile] = (data, weights / weights.sum(), cells, names)
    return out


def _curve(data: np.ndarray, weights: np.ndarray, cells: pd.Index) -> np.ndarray:
    """Weighted mean over households, scaled to a mean of 1 (weekday:weekend = 5:2)."""
    mean = np.tensordot(weights, data, axes=1)  # (cells, seasons)
    day_w = np.array([5.0 if dt == "weekday" else 2.0 for dt, _ in cells])[:, None]
    return mean / ((mean * day_w).sum() / (day_w.sum() * mean.shape[1]))


def _to_table(curves: dict[str, np.ndarray], cells: pd.Index) -> pd.DataFrame:
    rows = []
    for profile in PROFILES:
        for month, season in SEASON_OF_MONTH.items():
            column = curves[profile][:, SEASONS.index(season)]
            for (daytype, hour), value in zip(cells, column):
                rows.append((month, daytype, hour, profile, value))
    long = pd.DataFrame(rows, columns=["month", "daytype", "hour", "profile", "value"])
    table = long.pivot_table(index=["month", "daytype", "hour"], columns="profile", values="value")
    return table[list(PROFILES)].reset_index()


def build(job_dir: Path) -> pd.DataFrame:
    loaded = load_households(job_dir)
    cells = next(iter(loaded.values()))[2]
    return _to_table({p: _curve(d, w, c) for p, (d, w, c, _) in loaded.items()}, cells)


def bootstrap(job_dir: Path, n_boot: int = 300, seed: int = 0):
    """Bands and ordering stability from resampling the households of each profile."""
    loaded = load_households(job_dir)
    cells = next(iter(loaded.values()))[2]
    rng = np.random.default_rng(seed)
    point = {p: _curve(d, w, c) for p, (d, w, c, _) in loaded.items()}
    draws = {p: [] for p in PROFILES}
    for _ in range(n_boot):
        for p, (data, weights, c, _) in loaded.items():
            pick = rng.choice(len(weights), size=len(weights), replace=True, p=weights)
            equal = np.full(len(pick), 1.0 / len(pick))
            draws[p].append(_curve(data[pick], equal, c))
    draws = {p: np.stack(v) for p, v in draws.items()}  # (boot, cells, seasons)

    rows = []
    for p in PROFILES:
        lo, hi = np.percentile(draws[p], [5, 95], axis=0)
        for i, (daytype, hour) in enumerate(cells):
            for j, season in enumerate(SEASONS):
                rows.append((p, season, daytype, hour, point[p][i, j], lo[i, j], hi[i, j]))
    bands = pd.DataFrame(rows, columns=["profile", "season", "daytype", "hour", "mean", "lo", "hi"])

    stability = []
    for a, b in combinations(PROFILES, 2):
        sign = np.sign(draws[a] - draws[b])  # (boot, cells, seasons)
        agree = np.maximum((sign > 0).mean(axis=0), (sign < 0).mean(axis=0))
        stability.append(
            dict(
                pair=f"{a} vs {b}",
                mean_agreement=float(agree.mean()),
                cells_ordered_95pct=float((agree >= 0.95).mean()),
                cells_ordered_80pct=float((agree >= 0.80).mean()),
            )
        )
    peaks = []
    weekday = np.array([dt == "weekday" for dt, _ in cells])
    hours = np.array([h for _, h in cells])[weekday]
    for p in PROFILES:
        annual = draws[p][:, weekday, :].mean(axis=2)  # (boot, 24)
        point_peak = hours[point[p][weekday].mean(axis=1).argmax()]
        peak = hours[annual.argmax(axis=1)]
        peaks.append(
            dict(
                profile=p,
                peak_hour_estimate=int(point_peak),
                share_within_1h=float((np.minimum(abs(peak - point_peak), 24 - abs(peak - point_peak)) <= 1).mean()),
                peak_hour_5pct=int(np.percentile(peak, 5)),
                peak_hour_95pct=int(np.percentile(peak, 95)),
            )
        )
    return bands, pd.DataFrame(stability), pd.DataFrame(peaks)


def _plot(bands: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"urban": "tab:blue", "suburban": "tab:orange", "rural": "tab:green"}
    fig, axes = plt.subplots(1, len(SEASONS), figsize=(4.2 * len(SEASONS), 3.6), sharey=True)
    for ax, season in zip(axes, SEASONS):
        for p in PROFILES:
            sel = bands[(bands.profile == p) & (bands.season == season) & (bands.daytype == "weekday")]
            ax.fill_between(sel.hour, sel.lo, sel.hi, color=colours[p], alpha=0.25)
            ax.plot(sel.hour, sel["mean"], color=colours[p], label=p)
        ax.set_title(f"{season}, weekday")
        ax.set_xlabel("local hour")
    axes[0].set_ylabel("relative demand (mean = 1)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("job_dir")
    parser.add_argument("out")
    parser.add_argument("--bootstrap", type=int, default=0, help="number of resamples (0 = off)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    job_dir, out = Path(args.job_dir), Path(args.out)
    table = build(job_dir)
    table.to_csv(out, index=False, float_format="%.6f")
    print(f"wrote {out} ({len(table)} rows)")
    if args.bootstrap:
        bands, stability, peaks = bootstrap(job_dir, args.bootstrap, args.seed)
        stem = out.with_suffix("")
        bands.to_csv(f"{stem}_bands.csv", index=False, float_format="%.6f")
        stability.to_csv(f"{stem}_stability.csv", index=False, float_format="%.4f")
        peaks.to_csv(f"{stem}_peaks.csv", index=False, float_format="%.4f")
        print("\nORDERING OF THE PROFILES (share of cells where the order holds in >=95% / >=80% of resamples)")
        print(stability.round(3).to_string(index=False))
        print("\nPEAK HOUR (weekday, mean over seasons) across resamples")
        print(peaks.round(3).to_string(index=False))
        try:
            _plot(bands, Path(f"{stem}_bands.png"))
            print(f"\nplot written to {stem}_bands.png")
        except Exception as exc:
            print(f"\n(no plot: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
