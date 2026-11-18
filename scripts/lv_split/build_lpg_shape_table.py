# SPDX-License-Identifier: MIT
"""Turn pyLPG household simulations into the ``shapes`` table of the LV profile split.

Input : one CSV per (profile, household, season) with the minute-by-minute
        ``Electricity_HH1`` series from pyLPG (columns: timestamp, value).
        File names: ``<profile>__<household>__<season>.csv``, season in jan/apr/jul/oct.
Output: CSV with columns month, daytype, hour, urban, suburban, rural, to be given
        to ``sector.lv_split_profiles.shapes.electricity``.

Steps: hourly energy per household -> average over weekday / weekend days per season
-> average over the households of a profile -> each month takes its season ->
each profile is scaled to a mean of 1 (weekday and weekend weighted 5:2), so only
timing differs between profiles. The household mix per profile is an assumption.
"""
import sys
from pathlib import Path

import pandas as pd

PROFILES = ("urban", "suburban", "rural")
SEASON_OF_MONTH = {
    12: "jan", 1: "jan", 2: "jan",
    3: "apr", 4: "apr", 5: "apr",
    6: "jul", 7: "jul", 8: "jul",
    9: "oct", 10: "oct", 11: "oct",
}


def household_hourly(path: Path) -> pd.DataFrame:
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    hourly = s.resample("h").sum()
    df = pd.DataFrame({"v": hourly})
    df["daytype"] = ["weekend" if d >= 5 else "weekday" for d in df.index.dayofweek]
    df["hour"] = df.index.hour
    return df.groupby(["daytype", "hour"])["v"].mean()


def build(job_dir: Path) -> pd.DataFrame:
    parts = {}
    for path in sorted(job_dir.glob("*__*__*.csv")):
        profile, household, season = path.stem.split("__")
        parts.setdefault((profile, season), []).append(household_hourly(path))
    if not parts:
        raise FileNotFoundError(f"no simulation CSVs in {job_dir}")
    seasonal = {k: pd.concat(v, axis=1).mean(axis=1) for k, v in parts.items()}
    rows = []
    for profile in PROFILES:
        frame = pd.DataFrame(
            {season: seasonal[(profile, season)] for season in ("jan", "apr", "jul", "oct")}
        )
        weights = pd.Series(
            [5 if dt == "weekday" else 2 for dt, _ in frame.index], index=frame.index
        )
        mean = (frame.mul(weights, axis=0).sum().sum()) / (weights.sum() * frame.shape[1])
        frame = frame / mean
        for month, season in SEASON_OF_MONTH.items():
            for (daytype, hour), value in frame[season].items():
                rows.append((month, daytype, hour, profile, value))
    long = pd.DataFrame(rows, columns=["month", "daytype", "hour", "profile", "value"])
    table = long.pivot_table(
        index=["month", "daytype", "hour"], columns="profile", values="value"
    )[list(PROFILES)]
    return table.reset_index()


if __name__ == "__main__":
    job_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    table = build(job_dir)
    table.to_csv(out, index=False, float_format="%.6f")
    print(f"wrote {out} ({len(table)} rows)")
