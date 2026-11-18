# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Guards for the three-profile LV split (``scripts/lv_profile_split.py``).

Three independent checks, none of which changes model behaviour:

1. ``classify_component``  - label every component that gets cloned as one of
   demand / fixed_equipment / household_asset / process_plant / shared_infrastructure
   / unknown. ``unknown`` is logged, and raises if ``strict`` is set.
2. ``snapshot_totals`` + ``check_conservation`` - per (component, carrier, attribute)
   totals of demand energy and finite capacity limits must be identical before and
   after cloning (shares sum to one).
3. ``extendable_clone_report`` - extendable clones with an infinite ``*_nom_max`` are
   NOT limited by the shares; they are counted and logged so the person reading the
   log knows which capacities the optimiser may reallocate freely between profiles.
"""

import logging
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEMAND = "demand"
FIXED = "fixed_equipment"
HOUSEHOLD = "household_asset"
PROCESS = "process_plant"
SHARED = "shared_infrastructure"
UNKNOWN = "unknown"

# Substring rules on the carrier (lower-case). First match wins.
HOUSEHOLD_KEYS = (
    "heat pump",
    "resistive heater",
    "boiler",
    "micro chp",
    "solar thermal",
    "heat vent",
    "water tank",
    "water pit",
    "bev charger",
    "v2g",
    "home battery",
    "ev battery",
    "battery charger",
    "battery discharger",
)
SHARED_KEYS = ("distribution grid", "transmission", "low voltage")
FIXED_KEYS = ("solar rooftop", "rooftop")
CAPACITY_ATTRS = {
    "Link": ("p_nom", "p_nom_min", "p_nom_max"),
    "Generator": ("p_nom", "p_nom_min", "p_nom_max"),
    "StorageUnit": ("p_nom", "p_nom_min", "p_nom_max"),
    "Store": ("e_nom", "e_nom_min", "e_nom_max"),
    "Line": ("s_nom", "s_nom_min", "s_nom_max"),
    "Transformer": ("s_nom", "s_nom_min", "s_nom_max"),
}
NOM_FLAG = {
    "Link": "p_nom_extendable",
    "Generator": "p_nom_extendable",
    "StorageUnit": "p_nom_extendable",
    "Store": "e_nom_extendable",
    "Line": "s_nom_extendable",
    "Transformer": "s_nom_extendable",
}
NOM_MAX = {
    "Link": "p_nom_max",
    "Generator": "p_nom_max",
    "StorageUnit": "p_nom_max",
    "Store": "e_nom_max",
    "Line": "s_nom_max",
    "Transformer": "s_nom_max",
}
PLURAL = {
    "Load": "loads",
    "Link": "links",
    "Generator": "generators",
    "StorageUnit": "storage_units",
    "Store": "stores",
    "Line": "lines",
    "Transformer": "transformers",
}


def classify_component(component, carrier, touches_heat=False):
    """Return the class of one component that is about to be cloned.

    Order matters: demand by component type, rooftop PV, shared infrastructure,
    household assets, then anything else on a heat bus (process plant).
    """
    carrier = str(carrier).lower()
    if component == "Load":
        return DEMAND
    if any(k in carrier for k in FIXED_KEYS):
        return FIXED
    if any(k in carrier for k in SHARED_KEYS) or component in ("Line", "Transformer"):
        return SHARED
    if any(k in carrier for k in HOUSEHOLD_KEYS):
        return HOUSEHOLD
    if touches_heat:
        return PROCESS
    return UNKNOWN


def report_classes(classes, strict=False):
    """``classes``: list of (component, name, carrier, class). Log, raise if strict."""
    counts = Counter((c[0], c[3]) for c in classes)
    for (component, cls), count in sorted(counts.items()):
        logger.info("LV profile split: %-12s %-22s x%d", component, cls, count)
    unknown = sorted({(c[0], c[2]) for c in classes if c[3] == UNKNOWN})
    if unknown:
        msg = (
            "LV profile split: components with unclassified carriers "
            f"(cloned with the general shares): {unknown}"
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
    return counts


def _load_energy(n, plural_t, static, weights):
    """Sum over snapshots of p_set per carrier, honouring dynamic overrides."""
    dyn = plural_t.get("p_set", pd.DataFrame(index=n.snapshots))
    energy = pd.Series(0.0, index=static.index)
    for name in static.index:
        if name in dyn.columns:
            energy[name] = float((dyn[name] * weights).sum())
        else:
            energy[name] = float(static.at[name, "p_set"]) * float(weights.sum())
    return energy


def snapshot_totals(n):
    """Totals that the split must conserve: {(component, carrier, attribute): value}."""
    weights = n.snapshot_weightings.generators
    totals = {}
    if not n.loads.empty:
        energy = _load_energy(n, n.loads_t, n.loads, weights)
        for carrier, value in energy.groupby(n.loads.carrier).sum().items():
            totals[("Load", carrier, "energy")] = float(value)
    for component, attrs in CAPACITY_ATTRS.items():
        static = getattr(n, PLURAL[component])
        if static.empty:
            continue
        for attr in attrs:
            values = static[attr].astype(float)
            finite = values.where(np.isfinite(values), 0.0)
            for carrier, value in finite.groupby(static.carrier).sum().items():
                totals[(component, carrier, attr)] = float(value)
    return totals


def check_conservation(before, after, rtol=1e-9, atol=1e-6):
    """Raise if any conserved total changed. Returns the number of keys checked."""
    bad = []
    for key in sorted(set(before) | set(after)):
        b, a = before.get(key, 0.0), after.get(key, 0.0)
        if not np.isclose(a, b, rtol=rtol, atol=atol):
            bad.append((key, b, a))
    if bad:
        details = "; ".join(f"{k}: {b:.8g} -> {a:.8g}" for k, b, a in bad[:15])
        raise ValueError(
            f"LV profile split did not conserve {len(bad)} total(s): {details}"
        )
    logger.info("LV profile split: conservation check passed (%d totals).", len(before))
    return len(before)


def extendable_clone_report(n, names_by_component):
    """Count extendable clones with infinite nom_max; they ignore the shares.

    ``names_by_component``: {component: [clone names]}.
    Returns {(component, carrier): count}.
    """
    report = Counter()
    for component, names in names_by_component.items():
        if component not in NOM_FLAG or not names:
            continue
        static = getattr(n, PLURAL[component]).reindex(names)
        free = static[NOM_FLAG[component]].astype(bool) & ~np.isfinite(
            static[NOM_MAX[component]].astype(float)
        )
        for carrier, count in static.loc[free].groupby("carrier").size().items():
            report[(component, carrier)] = int(count)
    if report:
        logger.info(
            "LV profile split: %d extendable clone groups have infinite nom_max and "
            "are not constrained by the shares (optimiser may reallocate them): %s",
            len(report),
            dict(report),
        )
    return dict(report)


# ---------------------------------------------------------------------------
# Port guard (ported from the legacy ``split_transport``, generalised to every link)
# ---------------------------------------------------------------------------
SPLIT_BUS_CARRIERS = ("low voltage", "home battery", "EV battery")
HEAT_BUS_KEYS = ("heat", "water tank", "water pit", "aquifer")
EV_LINK_CARRIERS = ("BEV charger", "V2G")
EV_LOAD_CARRIER = "land transport EV"
EV_STORE_CARRIER = "EV battery"


def bus_profiles(n, profiles):
    """Series bus -> profile for the buses the split creates; NaN for all other buses.

    A split bus is an LV / home-battery / EV-battery / heat-chain bus whose name ends in
    a profile token (``"... low voltage urban"``).
    """
    carrier = n.buses.carrier.astype(str)
    last = pd.Series(
        [str(b).rsplit(" ", 1)[-1] for b in n.buses.index], index=n.buses.index
    )
    split_type = carrier.isin(SPLIT_BUS_CARRIERS) | carrier.map(
        lambda c: any(k in c for k in HEAT_BUS_KEYS)
    )
    return last.where(split_type & last.isin(list(profiles)))


def _port_cols(frame):
    return [c for c in frame.columns if c.startswith("bus") and c[3:].isdigit()]


def orphan_split_buses(n, profiles):
    """Split buses with nothing attached (e.g. zero-share profiles that were not dropped)."""
    prof = bus_profiles(n, profiles)
    split = set(prof.index[prof.notna()])
    used = set(n.loads.bus) | set(n.generators.bus) | set(n.stores.bus) | set(n.storage_units.bus)
    for column in _port_cols(n.links):
        used |= set(n.links[column].dropna())
    for plural in ("lines", "transformers"):
        static = getattr(n, plural)
        for column in ("bus0", "bus1"):
            if column in static.columns:
                used |= set(static[column].dropna())
    return sorted(split - used)


def check_profile_ports(n, profiles, raise_on_error=True):
    """Fail if any component connects buses of two different profiles.

    Checks, on every Link / Line / Transformer (all ports) and every one-port
    component (bus): at most one distinct profile among its split buses. Then the
    transport rules of the legacy guard: each BEV charger / V2G link has exactly one
    LV bus and one EV bus, both on the same profile; every EV bus that carries a load
    has a charger. Unsplit LV / EV / home-battery buses (no profile token) are errors.
    Returns a dict of counts; raises ValueError listing offenders when
    ``raise_on_error``.
    """
    prof = bus_profiles(n, profiles)
    carrier = n.buses.carrier.astype(str)
    errors = []

    unsplit = n.buses.index[carrier.isin(SPLIT_BUS_CARRIERS) & prof.isna()]
    if len(unsplit):
        errors.append(f"unsplit LV/EV/home-battery buses: {list(unsplit[:5])}")

    cross = {}
    for comp, plural in (("Link", "links"), ("Line", "lines"), ("Transformer", "transformers")):
        static = getattr(n, plural)
        cols = _port_cols(static) if comp == "Link" else [c for c in ("bus0", "bus1") if c in static.columns]
        if static.empty or not cols:
            continue
        mapped = static[cols].apply(lambda col: col.map(prof))
        bad = static.index[mapped.nunique(axis=1) > 1]
        if len(bad):
            cross[comp] = list(bad)
    for comp, names in cross.items():
        errors.append(f"{comp} connecting two profiles ({len(names)}): {names[:5]}")

    links = n.links
    ev = links[links.carrier.isin(EV_LINK_CARRIERS)]
    n_ev_links = len(ev)
    if n_ev_links:
        cols = _port_cols(links)
        for name, row in ev.iterrows():
            buses = [row[c] for c in cols if isinstance(row[c], str) and row[c] in n.buses.index]
            lv = [b for b in buses if carrier[b] == "low voltage"]
            evb = [b for b in buses if carrier[b] == EV_STORE_CARRIER]
            if len(lv) != 1 or len(evb) != 1:
                errors.append(
                    f"{row['carrier']} link '{name}' must join exactly one LV bus and "
                    f"one EV bus; found LV {lv}, EV {evb}"
                )
            elif prof[lv[0]] != prof[evb[0]]:
                errors.append(
                    f"{row['carrier']} link '{name}' joins LV profile {prof[lv[0]]} to "
                    f"EV profile {prof[evb[0]]}"
                )

    ev_loads = n.loads[n.loads.bus.isin(carrier.index[carrier == EV_STORE_CARRIER])]
    chargers = set(links.loc[links.carrier == "BEV charger", _port_cols(links)].stack())
    no_charger = sorted(set(ev_loads.bus) - chargers)
    if no_charger:
        errors.append(f"EV buses with a load but no BEV charger: {no_charger[:5]}")

    orphans = orphan_split_buses(n, profiles)
    if orphans:
        logger.info(
            "LV profile split: %d split buses have no component attached (zero-share "
            "profiles until they are dropped; expect 0 after drop_zero_share_profiles): %s",
            len(orphans), orphans[:5],
        )
    summary = {
        "orphan_split_buses": len(orphans),
        "split_buses": int(prof.notna().sum()),
        "links_checked": int(len(links)),
        "ev_links_checked": int(n_ev_links),
        "errors": len(errors),
    }
    if errors and raise_on_error:
        raise ValueError("LV profile split port check failed: " + "; ".join(errors))
    for e in errors:
        logger.error("LV profile split: %s", e)
    if not errors:
        logger.info(
            "LV profile split: port check passed (%d split buses, %d links, %d EV links).",
            summary["split_buses"], summary["links_checked"], summary["ev_links_checked"],
        )
    set_report(port_check=summary)
    return summary


def ev_split_report(n, profiles, ev_shares, tol=1e-9, raise_on_error=True):
    """Realised EV split per profile vs ``ev_shares``.

    Compares EV demand energy, BEV charger and V2G ``p_nom`` and EV-battery ``e_nom``.
    Each must match the configured shares (the split scales them all by ``ev_shares``).
    Returns {quantity: {profile: realised share}}.
    """
    prof = bus_profiles(n, profiles)
    weights = n.snapshot_weightings.generators
    cols = _port_cols(n.links)
    link_prof = n.links[cols].apply(lambda col: col.map(prof)).bfill(axis=1).iloc[:, 0]

    quantities = {}
    loads = n.loads[n.loads.carrier == EV_LOAD_CARRIER]
    if len(loads):
        energy = _load_energy(n, n.loads_t, loads, weights)
        quantities["demand"] = energy.groupby(loads.bus.map(prof)).sum()
    for carrier in EV_LINK_CARRIERS:
        sel = n.links[n.links.carrier == carrier]
        if len(sel):
            vals = sel.p_nom.astype(float)
            quantities[carrier] = vals.where(np.isfinite(vals), 0.0).groupby(link_prof[sel.index]).sum()
    stores = n.stores[n.stores.carrier == EV_STORE_CARRIER]
    if len(stores):
        vals = stores.e_nom.astype(float)
        quantities["EV battery"] = vals.where(np.isfinite(vals), 0.0).groupby(stores.bus.map(prof)).sum()

    expected = pd.Series({p: float(ev_shares.get(p, 0.0)) for p in profiles})
    realised, bad = {}, []
    for name, series in quantities.items():
        total = float(series.sum())
        if total <= 0:
            continue
        share = (series / total).reindex(list(profiles)).fillna(0.0)
        realised[name] = {p: float(share[p]) for p in profiles}
        diff = (share - expected).abs().max()
        if diff > tol:
            bad.append(f"{name}: realised {realised[name]} vs ev_shares {expected.to_dict()}")
    set_report(ev_realised_shares=realised)
    if bad and raise_on_error:
        raise ValueError("EV split does not follow ev_shares: " + "; ".join(bad))
    if not bad:
        logger.info("LV profile split: EV demand, chargers, V2G and EV battery follow ev_shares.")
    return realised


# ---------------------------------------------------------------------------
# Run report (written next to the rule's log file, i.e. results/<run>/logs/)
# ---------------------------------------------------------------------------
_REPORT: dict = {}


def set_report(**sections):
    """Remember report sections produced inside ``apply_lv_profile_split``."""
    _REPORT.update(sections)


def write_report(log_path, removed=None):
    """Write ``<log stem>_lv_split_report.json`` and ``..._classes.csv`` next to the log.

    ``log_path``: the rule's log file (``snakemake.log[0]``). Returns the paths written,
    or [] if no report was produced (split disabled) or ``log_path`` is empty.
    """
    import json
    from pathlib import Path

    if not _REPORT or not log_path:
        return []
    base = Path(str(log_path))
    base.parent.mkdir(parents=True, exist_ok=True)
    stem = base.with_suffix("")
    report = dict(_REPORT)
    classes = report.pop("classes", [])
    report["removed_by_drop_zero_share_and_pin"] = {
        k: len(v) for k, v in (removed or {}).items()
    }
    report["removed_names"] = {k: list(v) for k, v in (removed or {}).items()}
    json_path = Path(f"{stem}_lv_split_report.json")
    csv_path = Path(f"{stem}_lv_split_classes.csv")
    json_path.write_text(json.dumps(report, indent=2, default=str))
    pd.DataFrame(classes, columns=["component", "name", "carrier", "class"]).to_csv(
        csv_path, index=False
    )
    logger.info("LV profile split: report written to %s and %s", json_path, csv_path)
    return [json_path, csv_path]


# ---------------------------------------------------------------------------
# PyPSA "attribute is misleading" warning (one per cloned component)
# ---------------------------------------------------------------------------
class _AttrMisleadingFilter(logging.Filter):
    """Drop PyPSA's ``warning-attr-misleading`` records and count them by attribute."""

    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def filter(self, record):
        message = record.getMessage()
        if "warning-attr-misleading" not in message:
            return True
        parts = message.split("'")
        self.counts[parts[1] if len(parts) > 1 else "?"] += 1
        return False


def silence_attr_misleading():
    """Install a filter on PyPSA's transform logger; call ``release_attr_misleading``."""
    flt = _AttrMisleadingFilter()
    logging.getLogger("pypsa.network.transform").addFilter(flt)
    return flt


def release_attr_misleading(flt):
    """Remove the filter, log ONE summary line, return {attribute: count}."""
    logging.getLogger("pypsa.network.transform").removeFilter(flt)
    counts = dict(flt.counts)
    if counts:
        logger.info(
            "LV profile split: suppressed %d PyPSA 'attribute is misleading' warnings "
            "(custom PyPSA-Eur columns copied to the clones, harmless): %s",
            sum(counts.values()),
            counts,
        )
    return counts
