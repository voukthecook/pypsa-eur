import json

import numpy as np
import pandas as pd

from scripts.lv_split.build_lpg_shape_table import (
    PROFILES,
    SEASONS,
    SEASON_OF_MONTH,
    bootstrap,
    build,
    main,
)

PEAK = {"urban": 19, "suburban": 18, "rural": 11}
START = {"jan": "2022-01-03", "apr": "2022-04-04", "jul": "2022-07-04", "oct": "2022-10-03"}


def _series(season: str, peak: int, level: float, noise: float, rng) -> pd.Series:
    index = pd.date_range(START[season], periods=14 * 24 * 60, freq="min")
    hour = index.hour + index.minute / 60
    base = level * (1 + 0.8 * np.exp(-((hour - peak) ** 2) / 6))
    return pd.Series(np.clip(base + noise * rng.standard_normal(len(index)), 0, None), index=index)


def _write(tmp_path, households, mix=None):
    rng = np.random.default_rng(0)
    for profile, names in households.items():
        for name in names:
            for season in SEASONS:
                level = 1.0 + 0.3 * rng.random()
                s = _series(season, PEAK[profile] + int(rng.integers(-1, 2)), level, 0.1, rng)
                s.to_csv(tmp_path / f"{profile}__{name}__{season}.csv")
    if mix is not None:
        (tmp_path / "mix.json").write_text(json.dumps(mix))


def _seeded(n_seeds=3):
    return {p: [f"{t}-s{s}" for t in ("TA", "TB") for s in range(1, n_seeds + 1)] for p in PROFILES}


def test_table_has_shape_and_unit_mean(tmp_path) -> None:
    _write(tmp_path, _seeded())
    table = build(tmp_path)
    assert len(table) == 12 * 2 * 24
    assert list(table.columns) == ["month", "daytype", "hour", *PROFILES]
    cells = table[table.month == 1]
    weights = np.where(cells.daytype == "weekday", 5.0, 2.0)
    for p in PROFILES:
        # each season block is normalised jointly; the Jan block mean is close to 1
        assert abs((cells[p] * weights).sum() / weights.sum() - 1) < 0.05
    # the peak hour of each profile is where we put it
    weekday = table[(table.month == 7) & (table.daytype == "weekday")].set_index("hour")
    for p in PROFILES:
        assert abs(weekday[p].idxmax() - PEAK[p]) <= 1


def test_mix_weights_pull_the_curve(tmp_path) -> None:
    rng = np.random.default_rng(1)
    households = {p: [f"TA-s1", "TB-s1"] for p in PROFILES}
    for profile, names in households.items():
        for name, peak in zip(names, (8, 21)):
            for season in SEASONS:
                _series(season, peak, 1.0, 0.0, rng).to_csv(tmp_path / f"{profile}__{name}__{season}.csv")
    mix = {p: {"TA": 0.9, "TB": 0.1} for p in PROFILES}
    (tmp_path / "mix.json").write_text(json.dumps(mix))
    table = build(tmp_path)
    day = table[(table.month == 1) & (table.daytype == "weekday")].set_index("hour")
    assert day["urban"].loc[8] > day["urban"].loc[21]  # template A (peak 8h) dominates


def test_bootstrap_bands_and_ordering(tmp_path) -> None:
    _write(tmp_path, _seeded(4))
    bands, stability, peaks = bootstrap(tmp_path, n_boot=60, seed=1)
    assert (bands["lo"] <= bands["mean"] + 1e-9).mean() > 0.9
    assert (bands["hi"] >= bands["mean"] - 1e-9).mean() > 0.9
    assert (bands["hi"] > bands["lo"]).all()
    assert set(stability["pair"]) == {"urban vs suburban", "urban vs rural", "suburban vs rural"}
    # urban (evening) and rural (late morning) are far apart: ordering is stable
    row = stability.set_index("pair").loc["urban vs rural"]
    assert row["mean_agreement"] > 0.9
    assert (peaks["share_within_1h"] > 0.5).all()


def test_cli_writes_all_outputs(tmp_path) -> None:
    _write(tmp_path, _seeded(2))
    out = tmp_path / "shapes.csv"
    assert main([str(tmp_path), str(out), "--bootstrap", "20"]) == 0
    for suffix in ("_bands.csv", "_stability.csv", "_peaks.csv", "_bands.png"):
        assert (tmp_path / f"shapes{suffix}").exists()
    assert len(pd.read_csv(out)) == 576


def test_old_naming_without_seeds_or_mix_still_works(tmp_path) -> None:
    _write(tmp_path, {p: ["H1", "H2", "H3", "H4"] for p in PROFILES})
    assert len(build(tmp_path)) == 576
