"""Generate per-profile household electricity shapes with pyLPG.

ASSUMPTION (not data): the household mix per profile is a judgement of which LPG
templates resemble urban / suburban / rural households. Edit MIX to change it.
"""
import os, sys, json, time, traceback
from multiprocessing import Pool
import pandas as pd
from pylpg import lpg_execution, lpgdata

HOUSE = lpgdata.HouseTypes.HT20_Single_Family_House_no_heating_cooling
MIX = {
    "urban": ["CHR07_Single_with_work", "CHR09_Single_woman_30_64_years_with_work",
              "CHR33_Couple_under_30_years_with_work", "CHR13_Student_with_Work"],
    "suburban": ["CHR01_Couple_both_at_Work", "CHR03_Family_1_child_both_at_work",
                 "CHR27_Family_both_at_work_2_children", "CHR45_Family_with_1_child_1_at_work_1_at_home"],
    "rural": ["CHR15_Multigenerational_Home_working_couple_2_children_2_seniors",
              "CHR16_Couple_over_65_years", "CHR20_one_at_work_one_work_home_3_children",
              "CHR54_Retired_Couple_no_work"],
}
# two full weeks (Mon-Sun) per season; month -> season map is applied afterwards
SEASONS = {"jan": ("2022-01-03", "2022-01-16"), "apr": ("2022-04-04", "2022-04-17"),
           "jul": ("2022-07-04", "2022-07-17"), "oct": ("2022-10-03", "2022-10-16")}
OUT = "lpg_jobs"  # one CSV per (profile, household, season); run from an empty scratch directory

def job(args):
    profile, hh, season = args
    out = f"{OUT}/{profile}__{hh}__{season}.csv"
    if os.path.exists(out):
        return args, "cached"
    wd = f"lpg_wd/{profile}_{season}_{hh[:5]}"
    wd = os.path.abspath(wd); os.makedirs(wd, exist_ok=True); os.chdir(wd)
    try:
        s, e = SEASONS[season]
        d = lpg_execution.execute_lpg_single_household(
            2022, getattr(lpgdata.Households, hh), HOUSE,
            startdate=s, enddate=e, random_seed=42)
        d["Electricity_HH1"].to_csv(out)
        return args, "ok"
    except Exception as ex:
        return args, "FAIL " + repr(ex)[:200]

if __name__ == "__main__":
    tasks = [(p, hh, s) for p, hhs in MIX.items() for hh in hhs for s in SEASONS]
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    t = time.time()
    with Pool(n) as pool:
        for a, r in pool.imap_unordered(job, tasks):
            print(round(time.time() - t), a, r, flush=True)
