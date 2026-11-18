# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build the representative low-voltage feeder library used by the
detailed distribution-grid extension.

This does not depend on the clustering resolution -- there are only
three archetypes (rural / semi-urban / urban), each mapped to one
SimBench benchmark LV grid, so this runs once regardless of
{clusters}. `build_distribution_grid_archetypes.py` (per-cluster)
and `prepare_sector_network.py`'s
`insert_electricity_distribution_grid_detailed` both read the tables
written here instead of importing `simbench` themselves.

Note: simbench.get_simbench_net() reads from data bundled with the
`simbench` package -- no network access is required at run time.
"""

import logging

import numpy as np
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)

# SimBench codes for the representative archetypes. "--0-sw" = base
# year, winter case. See simbench.get_all_simbench_codes_by_condition()
# for alternative years / seasons / DER penetration levels.
SIMBENCH_CODES = {
    "rural": "1-LV-rural1--0-sw",
    "semiurban": "1-LV-semiurb4--0-sw",
    "urban": "1-LV-urban6--0-sw",
}


def get_feeder_reference_data(archetype: str) -> dict:
    """Load one archetype's SimBench feeder and summarise it into the
    tables written to disk."""
    import simbench as sb

    code = SIMBENCH_CODES[archetype]
    net = sb.get_simbench_net(code)

    v_kv = net.bus.loc[net.line.from_bus, "vn_kv"].values
    s_nom_mva = np.sqrt(3) * v_kv * net.line["max_i_ka"].values

    lines = pd.DataFrame(
        {
            "archetype": archetype,
            "from_bus": net.line["from_bus"].values,
            "to_bus": net.line["to_bus"].values,
            "s_nom_mva": s_nom_mva,
        }
    )

    buses = pd.DataFrame(
        {
            "archetype": archetype,
            "bus": net.bus.index,
            "vn_kv": net.bus["vn_kv"].values,
        }
    )

    if len(net.trafo) != 1:
        raise NotImplementedError(
            f"Expected exactly one MV/LV transformer in {code}, "
            f"found {len(net.trafo)}. Extend get_feeder_reference_data() "
            "for multi-transformer feeders before using this archetype."
        )
    trafo_row = net.trafo.iloc[0]
    trafo = pd.DataFrame(
        {
            "archetype": [archetype],
            "hv_bus": [trafo_row["hv_bus"]],
            "lv_bus": [trafo_row["lv_bus"]],
            "s_nom_mva": [trafo_row["sn_mva"]],
        }
    )

    peak_load_mw = net.load["p_mw"].sum()
    load_shares = (
        net.load.groupby("bus")["p_mw"].sum().div(peak_load_mw).rename("share")
    )
    load_shares = load_shares.reset_index()
    load_shares.insert(0, "archetype", archetype)
    # carry peak_load_mw along so downstream scripts don't need to
    # re-derive it from the raw SimBench net
    load_shares["feeder_peak_load_mw"] = peak_load_mw
    # number of household/consumer connection points in this SimBench
    # feeder -- used downstream to size how many real feeders a
    # PyPSA-Eur node's population actually needs, instead of the
    # circular population-ratio heuristic this replaced.
    load_shares["feeder_n_households"] = len(net.load)

    return dict(buses=buses, lines=lines, trafo=trafo, load_shares=load_shares)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_representative_feeder_library")

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    feeder_reference = {a: get_feeder_reference_data(a) for a in SIMBENCH_CODES}

    pd.concat([d["buses"] for d in feeder_reference.values()], ignore_index=True).to_csv(
        snakemake.output.feeder_buses, index=False
    )
    pd.concat([d["lines"] for d in feeder_reference.values()], ignore_index=True).to_csv(
        snakemake.output.feeder_lines, index=False
    )
    pd.concat([d["trafo"] for d in feeder_reference.values()], ignore_index=True).to_csv(
        snakemake.output.feeder_trafos, index=False
    )
    pd.concat(
        [d["load_shares"] for d in feeder_reference.values()], ignore_index=True
    ).to_csv(snakemake.output.feeder_load_shares, index=False)

    logger.info(
        "Built representative feeder library for archetypes: %s",
        list(SIMBENCH_CODES),
    )
