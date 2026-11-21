# LV profile shapes (household electricity)

`lpg_electricity.csv` is the `shapes.electricity` table for `sector.lv_split_profiles`.
Columns: month, daytype (weekday/weekend), hour, urban, suburban, rural. Only relative
values matter; the split rebalances them (see `scripts/lv_profile_split.py`).

Provenance
- Simulated with pyLPG (`pyloadprofilegenerator`, LPG v11.0.0), house type HT20
  (single-family house, no heating or cooling), seed 42, year 2022.
- 4 LPG household templates per profile, 2 weeks each in Jan, Apr, Jul, Oct;
  other months take the nearest season (Dec-Feb = Jan, Mar-May = Apr, ...).
- The household mix per profile is an ASSUMPTION (see MIX in
  `scripts/lv_split/generate_lpg_household_profiles.py`), not data. LPG behaviour is German.
- Regenerate: run the generator in an empty directory, then
  `python scripts/lv_split/build_lpg_shape_table.py lpg_jobs data/lv_profile_shapes/lpg_electricity.csv`.

## lpg_electricity_v2.csv (larger sample)

Same recipe as above, but 24 households per profile: 6 LPG templates per profile with
documented weights (`lpg_v2_mix.json`, an assumption, see `MIX` in
`scripts/lv_split/generate_lpg_household_profiles.py`), each run with 4 random seeds,
2 weeks per season (Jan/Apr/Jul/Oct), house type HT20, 288 runs in total. The household
mix is more balanced than v1 (rural also contains commuting households).

Use it with `sector.lv_split_profiles.shapes.electricity: data/lv_profile_shapes/lpg_electricity_v2.csv`.

Robustness (500 bootstrap resamples of the households, see `lpg_electricity_v2_*.csv`):
- `_bands.csv/png`: 5-95% band of each curve.
- `_stability.csv`: share of (season, daytype, hour) cells where the order of two profiles
  holds in at least 95% (80%) of resamples.
- `_peaks.csv`: weekday peak hour per profile across resamples.
Reproduce: `python scripts/lv_split/generate_lpg_household_profiles.py 2 4 lpg_jobs` in an empty
folder, then `python -m scripts.lv_split.build_lpg_shape_table lpg_jobs <out.csv> --bootstrap 500`.
