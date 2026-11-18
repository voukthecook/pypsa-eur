# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Classify clustered network nodes into distribution-grid archetypes
(rural / semi-urban / urban) and determine how many parallel copies
of that archetype's representative feeder each node needs.

Depends on the clustering resolution (via clustered_pop_layout and
regions_onshore, both per-{clusters}), unlike
build_representative_feeder_library.py which is archetype-level only
and runs once. `insert_electricity_distribution_grid_detailed` in
prepare_sector_network.py reads this script's output together with
the feeder library.
"""

import logging
import math

import geopandas as gpd
import numpy as np
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def classify_archetypes(
    pop_layout: pd.DataFrame,
    onshore_regions: gpd.GeoDataFrame,
    urban_share_thresholds: tuple[float, float] = (0.2, 0.6),
) -> pd.Series:
    """
    Classify each node as rural / semi-urban / urban using the urban
    population share already computed by
    build_clustered_population_layouts.py (pop_layout["urban"] /
    pop_layout["total"]) -- a more direct signal than raw population
    density for this purpose, since it comes from the same World Bank
    urbanisation statistics used elsewhere in the model.

    `onshore_regions` is joined in for reference/diagnostics (e.g. if
    you want to sanity-check archetypes against node area) but is not
    currently part of the classification rule itself -- extend here
    for a density-based refinement if urban share alone proves too
    coarse.
    """
    urban_share = (pop_layout["urban"] / pop_layout["total"]).reindex(
        onshore_regions.index
    )

    low, high = urban_share_thresholds
    archetype = pd.cut(
        urban_share,
        bins=[-np.inf, low, high, np.inf],
        labels=["rural", "semiurban", "urban"],
    ).astype(str)
    archetype.name = "archetype"
    return archetype


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_distribution_grid_archetypes", clusters=48
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    max_replicas = snakemake.params.max_feeder_replicas
    persons_per_household = snakemake.params.get("persons_per_household", 2.2)

    pop_layout = pd.read_csv(snakemake.input.clustered_pop_layout, index_col=0)
    onshore_regions = gpd.read_file(snakemake.input.regions_onshore).set_index("name")
    feeder_load_shares = pd.read_csv(snakemake.input.feeder_load_shares)

    archetypes = classify_archetypes(pop_layout, onshore_regions)

    feeder_n_households = feeder_load_shares.groupby("archetype")[
        "feeder_n_households"
    ].first()

    node_table = archetypes.to_frame()
    node_table["feeder_n_households"] = node_table["archetype"].map(
        feeder_n_households
    )

    # How many real SimBench-sized feeders would this node actually need?
    # pop_layout["total"] is population in THOUSANDS (see
    # build_clustered_population_layouts.py). This replaces an earlier,
    # circular version of this heuristic that scaled replicas relative
    # to the feeder's own peak load rather than to real household counts
    # -- that version badly undercounted needed replicas (by 1-2 orders
    # of magnitude for a typical PyPSA-Eur node), which is what produced
    # the capacity/demand mismatch and solver infeasibility this
    # revision fixes. persons_per_household is a rough EU-wide average;
    # override via config if you have a better country-specific figure.
    node_households = pop_layout["total"].reindex(archetypes.index) * 1000 / persons_per_household
    needed_replicas = (node_households / node_table["feeder_n_households"]).apply(
        lambda x: max(1, math.ceil(x))
    )

    # Building `needed_replicas` copies of a ~15-100 bus feeder per node
    # is not computationally tractable (a single DK node can need
    # thousands of real feeders). Instead, build only a small SAMPLE of
    # up to `max_replicas` feeder copies, and scale each sampled
    # replica's cable/transformer capacity up by `scale_factor` so the
    # aggregate capacity still represents the true number of real
    # feeders -- see insert_electricity_distribution_grid_detailed(),
    # which reads `scale_factor` and multiplies p_nom/s_nom by it. This
    # keeps model size bounded while keeping the represented physical
    # capacity realistic; congestion resolution is coarser than 1
    # real feeder, bounded by `max_replicas`.
    node_table["n_replicas"] = needed_replicas.clip(upper=max_replicas)
    node_table["scale_factor"] = needed_replicas / node_table["n_replicas"]

    node_table[["archetype", "n_replicas", "scale_factor"]].to_csv(
        snakemake.output.node_archetypes
    )

    logger.info(
        "Classified %d nodes into archetypes: %s (needed_replicas range %d-%d, "
        "max scale_factor %.1f)",
        len(node_table),
        node_table["archetype"].value_counts().to_dict(),
        int(needed_replicas.min()),
        int(needed_replicas.max()),
        node_table["scale_factor"].max(),
    )
