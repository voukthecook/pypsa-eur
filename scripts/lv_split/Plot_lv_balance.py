"""
Energy balance of the low-voltage (LV) buses of a solved network, per node
and per LV sub-bus, with NO threshold hiding small contributors.

Unlike the balances plot from plot_summary.py (which drops every technology
below plotting.energy_threshold / 10 TWh/a and aggregates everything into one
bar), this shows each LV sub-bus ("... low voltage a" / "... low voltage b")
separately, so the effect of split_low_voltage_buses() is visible. For a
network without the split, each node's single LV bus is shown as one bar.

Run from the repo root with the pypsa-eur conda environment active:

    python scripts/Plot_lv_balance.py results/<run name>/networks/base_s_3_lv1.2_Co2L0-168H_2030.nc
"""

import sys
from pathlib import Path

import matplotlib

import pandas as pd

SUFFIXES = [" low voltage a", " low voltage b", " low voltage"]


def parse_bus(bus: str) -> tuple[str, str]:
    """Split an LV bus name into (node, side); side is '-' for an unsplit bus."""
    for suffix in SUFFIXES:
        if bus.endswith(suffix):
            side = suffix[-1] if suffix.endswith((" a", " b")) else "-"
            return bus[: -len(suffix)], side
    return bus, "-"


def build_tables(eb: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    eb: energy balance in MWh/a indexed by (component, bus, carrier).
    Returns the long table and a carrier x sub-bus table in TWh/a.
    """
    long = eb.reset_index(name="MWh")
    parsed = long["bus"].map(parse_bus)
    long["node"] = parsed.map(lambda t: t[0])
    long["side"] = parsed.map(lambda t: t[1])
    long["label"] = long.apply(
        lambda r: r["node"] if r["side"] == "-" else f"{r['node']} {r['side']}",
        axis=1,
    )
    table = (
        long.pivot_table(index="carrier", columns="label", values="MWh", aggfunc="sum")
        .fillna(0.0)
        / 1e6
    )
    table = table[table.abs().max(axis=1) > 1e-9]
    return long, table


def main(path: str) -> None:
    import matplotlib.pyplot as plt
    import pypsa

    n = pypsa.Network(path)
    eb = n.statistics.energy_balance(
        bus_carrier="low voltage", groupby=["bus", "carrier"]
    )
    long, table = build_tables(eb)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

    print(f"Network: {path}\n")
    print("LV energy balance by carrier and sub-bus [TWh/a] (no threshold applied):")
    print(table, "\n")

    total = table.sum(axis=1)
    print("Total over all LV buses [TWh/a]:")
    print(total.sort_values(ascending=False), "\n")

    supply = total[total > 0].sum()
    demand = total[total < 0].sum()
    print(f"Supply {supply:,.3f} | demand {demand:,.3f} | net {supply + demand:,.3f} TWh/a")
    print("(net should be ~0 apart from conversion losses on the distribution links)\n")

    print("Net per LV bus [TWh/a] (each should be ~0):")
    print(table.sum(axis=0), "\n")

    fig, ax = plt.subplots(figsize=(max(8, 1.1 * table.shape[1] + 4), 7))
    table.T.plot(kind="bar", stacked=True, ax=ax, width=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Energy [TWh/a]")
    ax.set_xlabel("")
    ax.set_title("Low-voltage energy balance per node / sub-bus")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    fig.tight_layout()

    out = Path(f"lv_balance_{Path(path).stem}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {out.resolve()}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python plot_lv_balance.py <path to solved network .nc>")
    main(sys.argv[1])