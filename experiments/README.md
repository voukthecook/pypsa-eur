# Experiments

Ad-hoc analysis tools and their outputs live here, separate from the maintained
workflow scripts and the standard PyPSA-Eur `results/` directory.

- `scripts/` contains standalone experiment and analysis scripts.
- `results/` contains generated reports and experiment outputs.

The low-voltage profile comparison tools can be run from the repository root:

```text
python -m experiments.scripts.lv_split.compare_runs CONTROL.nc SHAPED.nc
python -m experiments.scripts.lv_split.plot_demand CONTROL.nc SHAPED.nc
python -m experiments.scripts.lv_split.validate_shapes CONTROL.nc SHAPED.nc --table data/lv_profile_shapes/lpg_electricity.csv
```

By default, these tools write to `experiments/results/`.
