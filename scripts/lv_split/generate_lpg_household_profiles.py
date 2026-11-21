# SPDX-License-Identifier: MIT
"""Generate per-profile household electricity series with pyLPG (several seeds per template).

Run from an empty scratch directory::

    python /path/to/generate_lpg_household_profiles.py [workers] [seeds] [out_dir]

ASSUMPTION (not data): ``MIX`` says which LPG household templates stand for urban,
suburban and rural households and with which weight. It is a judgement and should be
documented and varied in a sensitivity run. Each template is simulated with several
random seeds, because one seed is one random draw of the household's schedule.

Output: ``<out_dir>/<profile>__<template>-s<seed>__<season>.csv`` (minute-by-minute
``Electricity_HH1``) and ``<out_dir>/mix.json`` (template weights, read by
``build_lpg_shape_table.py``). Finished jobs are skipped, so the run can be resumed.
"""

import json
import os
import shutil
import sys
import time
from multiprocessing import Pool

from pylpg import lpg_execution, lpgdata

HOUSE = lpgdata.HouseTypes.HT20_Single_Family_House_no_heating_cooling

# template -> weight within the profile (weights sum to 1)
MIX = {
    "urban": {  # small, mostly working households, students, some elderly singles
        "CHR07_Single_with_work": 0.20,
        "CHR33_Couple_under_30_years_with_work": 0.20,
        "CHR39_Couple_30_64_years_with_work": 0.20,
        "CHR35_Single_woman_30_64_years_with_work": 0.15,
        "CHR13_Student_with_Work": 0.10,
        "CHR24_Single_woman_over_65_years": 0.15,
    },
    "suburban": {  # families, mostly both parents working
        "CHR01_Couple_both_at_Work": 0.20,
        "CHR03_Family_1_child_both_at_work": 0.20,
        "CHR27_Family_both_at_work_2_children": 0.20,
        "CHR45_Family_with_1_child_1_at_work_1_at_home": 0.15,
        "CHR44_Family_with_2_children_1_at_work_1_at_home": 0.15,
        "CHR41_Family_with_3_children_both_at_work": 0.10,
    },
    "rural": {  # retirees and larger households, but also commuters
        "CHR16_Couple_over_65_years": 0.20,
        "CHR54_Retired_Couple_no_work": 0.15,
        "CHR15_Multigenerational_Home_working_couple_2_children_2_seniors": 0.15,
        "CHR20_one_at_work_one_work_home_3_children": 0.15,
        "CHR02_Couple_30_64_age_with_work": 0.20,
        "CHR05_Family_3_children_both_with_work": 0.15,
    },
}
# two full weeks (Mon-Sun) per season; the table builder maps months onto seasons
SEASONS = {
    "jan": ("2022-01-03", "2022-01-16"),
    "apr": ("2022-04-04", "2022-04-17"),
    "jul": ("2022-07-04", "2022-07-17"),
    "oct": ("2022-10-03", "2022-10-16"),
}


BASE = os.getcwd()  # work folders are created here, never inside another job's folder


def job(args):
    profile, template, seed, season, out_dir = args
    out = f"{out_dir}/{profile}__{template}-s{seed}__{season}.csv"
    if os.path.exists(out):
        return args, "cached"
    workdir = os.path.join(BASE, "lpg_wd", f"{profile}_{season}_{template[:5]}_{seed}")
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    try:
        start, end = SEASONS[season]
        shutil.rmtree(os.path.join(workdir, "C1"), ignore_errors=True)  # no stale results
        data = lpg_execution.execute_lpg_single_household(
            2022,
            getattr(lpgdata.Households, template),
            HOUSE,
            startdate=start,
            enddate=end,
            random_seed=seed,
        )
        data["Electricity_HH1"].to_csv(out + ".part")
        os.replace(out + ".part", out)  # a half-written file is never taken as finished
        os.chdir(BASE)
        shutil.rmtree(workdir, ignore_errors=True)  # LPG leaves large result folders behind
        return args, "ok"
    except Exception as exc:  # keep going, report at the end
        return args, "FAIL " + repr(exc)[:200]


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    out_dir = os.path.abspath(sys.argv[3] if len(sys.argv) > 3 else "lpg_jobs")
    os.makedirs(out_dir, exist_ok=True)
    for profile, weights in MIX.items():
        assert abs(sum(weights.values()) - 1) < 1e-9, f"{profile} weights must sum to 1"
    with open(f"{out_dir}/mix.json", "w") as fh:
        json.dump(MIX, fh, indent=2)
    tasks = [
        (profile, template, seed, season, out_dir)
        for profile, weights in MIX.items()
        for template in weights
        for seed in range(1, seeds + 1)
        for season in SEASONS
    ]
    start = time.time()
    failed = 0
    with Pool(workers) as pool:
        for i, (args, result) in enumerate(pool.imap_unordered(job, tasks), 1):
            failed += result.startswith("FAIL")
            print(f"{round(time.time() - start):>5}s {i}/{len(tasks)} {args[:4]} {result}", flush=True)
    print(f"done, {failed} failed")
