# SPDX-License-Identifier: MIT
"""
Resolve the LV profile shape tables to snapshots in each country's local time.

Inputs:
-------
- `shape_<key>`: month/daytype/hour tables (e.g. LPG) from `sector.lv_split_profiles.shapes`.
- `snapshot_weightings`: snapshots and their duration in hours.

Outputs:
--------
- `resources/<run_name>/lv_profile_shapes_base_s_{clusters}_elec_{opts}_{sector_opts}.csv`:
  one row per snapshot, columns (shape key, country, profile). Read back with
  `pd.read_csv(path, header=[0, 1, 2], index_col=0, parse_dates=True)`.

Notes
-----
- Daylight saving and time-zone differences between countries are applied.
- A 3H snapshot averages the three hourly values it covers.
"""

import logging

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.lv_profile_split import _load_profile_shapes
from scripts.lv_split.local_time_shapes import resolve_shape

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_lv_profile_shapes", clusters=39, opts="", sector_opts="", planning_horizons="2050"
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    weights = pd.read_csv(
        snakemake.input.snapshot_weightings, index_col=0, parse_dates=True
    )
    tables = {
        key[len("shape_") :]: path
        for key, path in snakemake.input.items()
        if key.startswith("shape_")
    }
    shapes = _load_profile_shapes(tables)
    overrides = snakemake.params.timezones or {}

    resolved = {
        key: resolve_shape(
            weights.index,
            weights["generators"],
            table,
            snakemake.params.countries,
            overrides,
        )
        for key, table in shapes.items()
    }
    out = pd.concat(resolved, axis=1)
    out.to_csv(snakemake.output[0])
    logger.info(
        "Wrote LV profile shapes for %s in local time (%d snapshots).",
        sorted(resolved),
        len(out),
    )
