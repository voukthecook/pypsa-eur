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
