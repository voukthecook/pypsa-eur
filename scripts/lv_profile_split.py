# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Three-profile split of the low-voltage (LV) distribution grid.

Used by prepare_sector_network.py when ``sector.lv_split_profiles.enable`` is true:

1. ``apply_lv_profile_split`` clones every LV, EV and heat-chain bus (and everything
   attached to it) once per profile (urban / suburban / rural) and scales demands and
   capacity limits by the configured shares (``shares``, ``pv_shares``,
   ``heat_shares``, ``ev_shares``).
2. ``drop_zero_share_profiles`` validates the shares, removes buses (and everything
   attached to them) of profiles with a zero share, and optionally pins links that touch
   a heat bus but are not household heating technologies (DAC, electrolysis, ...) to one
   profile (``pin_profile``; null disables), merging their capacity limits.

Bus names must end with " <profile>".
"""

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import pypsa

from scripts.lv_profile_split_guards import (
    check_conservation,
    classify_component,
    extendable_clone_report,
    report_classes,
    set_report,
    snapshot_totals,
)

from scripts.lv_profile_split_guards import (
    release_attr_misleading,
    silence_attr_misleading,
)

from scripts.lv_profile_split_guards import (
    check_profile_ports,
    ev_split_report,
)

logger = logging.getLogger(__name__)


PROFILES = ("urban", "suburban", "rural")


HEAT_SYSTEMS = ("rural", "urban decentral", "urban central")


PROFILE_SPLIT_OUTPUT_COLUMNS = ("p_nom_opt", "e_nom_opt", "s_nom_opt", "sub_network")


PROFILE_SPLIT_COMPONENTS = {
    "Link": ("links", ("p_nom", "p_nom_min", "p_nom_max"), ("p_set",)),
    "Load": ("loads", (), ("p_set", "q_set")),
    "Generator": (
        "generators",
        ("p_nom", "p_nom_min", "p_nom_max"),
        ("p_set",),
    ),
    "Store": ("stores", ("e_nom", "e_nom_min", "e_nom_max"), ("e_set",)),
    "StorageUnit": (
        "storage_units",
        ("p_nom", "p_nom_min", "p_nom_max"),
        ("p_set",),
    ),
    "Line": ("lines", ("s_nom", "s_nom_min", "s_nom_max"), ()),
    "Transformer": (
        "transformers",
        ("s_nom", "s_nom_min", "s_nom_max"),
        (),
    ),
}


def _profile_split_shares(
    value: Any, name: str, default: Mapping[str, float]
) -> dict[str, float]:
    if value is None:
        value = default
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping of profile names to shares.")

    unknown = set(value) - set(PROFILES)
    missing = set(PROFILES) - set(value)
    if unknown or missing:
        raise ValueError(
            f"{name} must contain exactly {PROFILES}; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}."
        )

    result = {}
    for profile in PROFILES:
        share = value[profile]
        if isinstance(share, bool) or not isinstance(share, (int, float)):
            raise TypeError(f"{name}.{profile} must be a number, got {share!r}.")
        if not np.isfinite(share) or share < 0:
            raise ValueError(f"{name}.{profile} must be finite and non-negative.")
        result[profile] = float(share)

    if not np.isclose(sum(result.values()), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError(f"{name} shares must sum to 1.0, got {sum(result.values())}.")
    return result


PROFILE_SHAPE_KEYS = ("electricity", "ev")


def _load_profile_shapes(value: Any) -> dict[str, pd.DataFrame]:
    """Read optional per-profile daily shapes (e.g. from LPG/HiSim).

    ``value`` maps a shape key ("electricity", "ev") to a CSV path or a DataFrame.
    Each table needs the columns ``month`` (1-12), ``daytype`` (weekday/weekend),
    ``hour`` (0-23) and one column per profile. Only relative values matter: the
    split rebalances them so the node total is unchanged in every snapshot and each
    profile keeps its configured share of the yearly energy (timing moves, not volume).
    """
    if not value:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("shapes must map shape keys to CSV paths or DataFrames.")
    unknown = set(value) - set(PROFILE_SHAPE_KEYS)
    if unknown:
        misplaced = unknown & {"household_fraction", "shape_load_carriers"}
        hint = (
            f" {sorted(misplaced)} belong next to 'shapes' (same indentation), "
            "not inside it."
            if misplaced
            else ""
        )
        raise ValueError(
            f"shapes has unsupported keys {sorted(unknown)}; use {PROFILE_SHAPE_KEYS}."
            + hint
        )
    shapes = {}
    for key, source in value.items():
        table = source if isinstance(source, pd.DataFrame) else pd.read_csv(source)
        needed = {"month", "daytype", "hour", *PROFILES}
        if needed - set(table.columns):
            raise ValueError(
                f"shapes.{key} is missing columns {sorted(needed - set(table.columns))}."
            )
        table = table.set_index(["month", "daytype", "hour"])[list(PROFILES)]
        if table.index.has_duplicates:
            raise ValueError(f"shapes.{key} has duplicate (month, daytype, hour) rows.")
        if (table < 0).any().any() or table.isna().any().any():
            raise ValueError(f"shapes.{key} must be finite and non-negative.")
        shapes[key] = table.astype(float)
    return shapes


def _profile_shape_factors(
    snapshots: pd.Index,
    shape: pd.DataFrame,
    shares: Mapping[str, float],
    base: pd.Series,
    weights: pd.Series,
    max_iter: int = 500,
    tol: float = 1e-12,
    resolved: pd.DataFrame | None = None,
) -> dict[str, pd.Series]:
    """Per-snapshot multiplier for each profile.

    The raw shape only sets the *relative* timing between profiles. It is rebalanced
    (iterative row/column scaling) so that, for this load,

    * the profile parts add up to the node demand in every snapshot, and
    * each profile still receives exactly its configured share of the yearly energy.

    ``resolved`` (snapshots x profiles, already in the node's local time, see
    ``scripts/lv_split/local_time_shapes.py``) replaces the UTC month/daytype/hour
    lookup in ``shape``.
    """
    if not isinstance(snapshots, pd.DatetimeIndex):
        raise NotImplementedError(
            "Profile shapes need plain datetime snapshots (not multi-period)."
        )
    if resolved is not None:
        raw = resolved.reindex(snapshots)[list(PROFILES)]
        if raw.isna().any().any() or (raw < 0).any().any():
            raise ValueError("Resolved profile shape must be finite and non-negative.")
    else:
        keys = pd.MultiIndex.from_arrays(
            [
                snapshots.month,
                np.where(snapshots.dayofweek >= 5, "weekend", "weekday"),
                snapshots.hour,
            ],
            names=["month", "daytype", "hour"],
        )
        raw = shape.reindex(keys)
        if raw.isna().any().any():
            raise ValueError("Profile shape table does not cover every snapshot hour.")
    active = [p for p in PROFILES if shares[p] > 0]
    raw = raw[active].to_numpy(dtype=float)
    if (raw.sum(axis=1) <= 0).any():
        raise ValueError("Profile shape is zero for every profile in some snapshots.")

    energy = (base * weights).to_numpy(dtype=float)
    live = energy > 0
    share = np.array([shares[p] for p in active])
    kernel = raw[live] * share * energy[live, None]
    row_target = energy[live]
    col_target = share * energy[live].sum()
    r = np.ones(kernel.shape[0])
    c = np.ones(kernel.shape[1])
    for _ in range(max_iter):
        r = row_target / (kernel * c).sum(axis=1)
        c = col_target / (kernel * r[:, None]).sum(axis=0)
        fitted = kernel * r[:, None] * c
        err = max(
            np.abs(fitted.sum(axis=1) - row_target).max() / row_target.max(),
            np.abs(fitted.sum(axis=0) - col_target).max() / col_target.max(),
        )
        if err < tol:
            break
    else:
        raise ValueError(
            f"Profile shape rebalancing did not converge (residual {err:.2e}); "
            "the shape is too far from the configured shares."
        )
    # fitted / (energy * share) = multiplier on the plain "base * share" series
    factors = {p: pd.Series(1.0, index=snapshots) for p in PROFILES}
    for i, p in enumerate(active):
        values = np.ones(len(snapshots))
        values[live] = fitted[:, i] / (energy[live] * shares[p])
        factors[p] = pd.Series(values, index=snapshots)
    return factors


def _resolved_for(table: pd.DataFrame | None, bus: str) -> pd.DataFrame | None:
    """Local-time shape of the bus's country (columns are (country, profile))."""
    if table is None:
        return None
    country = str(bus)[:2].upper()
    if country not in table.columns.get_level_values(0):
        raise KeyError(f"Resolved shape has no country '{country}' (bus '{bus}').")
    return table[country]


def household_electricity_fraction(totals: pd.DataFrame) -> pd.Series:
    """Household share of the base electricity load per node.

    Appliance-type use only: residential and services electricity minus the parts
    PyPSA-Eur already models as heat (space heating, water heating), because
    ``build_heat_demand`` removes those from the base load. Cooking stays in.
    Industry and other residual use in the base load are not counted, so the
    result is a share of households versus services, an approximation.
    """
    def appliance(sector: str) -> pd.Series:
        return (
            totals[f"electricity {sector}"]
            - totals[f"electricity {sector} space"]
            - totals[f"electricity {sector} water"]
        ).clip(lower=0.0)

    households, services = appliance("residential"), appliance("services")
    return (households / (households + services)).fillna(1.0)


def _household_fraction_for(
    value: Any, n: pypsa.Network, bus: str
) -> float:
    """Fraction of a load that follows the profile shape (rest keeps plain shares)."""
    if value is None:
        return 1.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        fraction = float(value)
    else:
        node = n.buses.at[bus, "location"] if "location" in n.buses else bus
        if node not in value:
            raise ValueError(f"household_fraction has no entry for node '{node}'.")
        fraction = float(value[node])
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"household_fraction must be in [0, 1], got {fraction}.")
    return fraction


def _profile_split_heat_system(carrier: Any) -> str | None:
    carrier = str(carrier).lower()
    if "urban central" in carrier:
        return "urban central"
    if "urban decentral" in carrier:
        return "urban decentral"
    if "rural" in carrier:
        return "rural"
    return None


def apply_lv_profile_split(
    n: pypsa.Network, config: Mapping[str, Any]
) -> pypsa.Network:
    """Clone LV-connected components across three profiles using configured shares.

    General shares apply to demand and non-PV LV assets. Rooftop PV, heat
    systems, and EV components can use separate shares. All share sets sum to
    one; unspecified overrides inherit the general shares.
    """
    if not config.get("enable", False):
        return n

    totals_before = snapshot_totals(n)
    attr_filter = silence_attr_misleading()
    strict_classification = bool(config.get("strict_classification", False))

    uniform = {profile: 1 / len(PROFILES) for profile in PROFILES}
    general_shares = _profile_split_shares(
        config.get("shares", uniform), "shares", uniform
    )
    pv_shares = _profile_split_shares(
        config.get("pv_shares"), "pv_shares", general_shares
    )
    ev_shares = _profile_split_shares(
        config.get("ev_shares"), "ev_shares", general_shares
    )

    profile_shapes = _load_profile_shapes(config.get("shapes"))
    resolved_shapes = config.get("resolved_shapes") or {}
    for key in resolved_shapes:
        profile_shapes.setdefault(key, None)
    shape_loads = set(config.get("shape_load_carriers", ["electricity"]))
    household_fraction = config.get("household_fraction")

    heat_config = config.get("heat_shares", {}) or {}
    if not isinstance(heat_config, Mapping):
        raise TypeError("heat_shares must map heat systems to profile share mappings.")
    unknown_heat_systems = set(heat_config) - set(HEAT_SYSTEMS)
    if unknown_heat_systems:
        raise ValueError(
            "heat_shares contains unsupported systems: "
            f"{sorted(unknown_heat_systems)}."
        )
    heat_shares = {
        system: _profile_split_shares(
            heat_config.get(system), f"heat_shares.{system}", general_shares
        )
        for system in HEAT_SYSTEMS
    }

    bus_domains: dict[str, str] = {}
    for bus, row in n.buses.iterrows():
        carrier = str(row.carrier)
        if carrier in ("low voltage", "home battery"):
            bus_domains[bus] = "general"
        elif carrier == "EV battery":
            bus_domains[bus] = "ev"
        elif (
            (system := _profile_split_heat_system(carrier)) is not None
            and any(
                component in carrier
                for component in ("heat", "water tank", "water pits", "aquifer")
            )
        ):
            bus_domains[bus] = f"heat:{system}"

    low_voltage_buses = [
        bus for bus in n.buses.index if n.buses.at[bus, "carrier"] == "low voltage"
    ]
    if not low_voltage_buses:
        raise ValueError("Profile split is enabled but the network has no LV buses.")

    bus_names = {
        bus: {profile: f"{bus} {profile}" for profile in PROFILES}
        for bus in bus_domains
    }
    for profile_names in bus_names.values():
        collisions = set(profile_names.values()) & set(n.buses.index)
        if collisions:
            raise ValueError(
                f"Profile split would overwrite existing bus(es): {sorted(collisions)}."
            )

    component_plans = {}
    classes = []
    for component, (plural, _, _) in PROFILE_SPLIT_COMPONENTS.items():
        static = getattr(n, plural)
        if static.empty:
            continue
        if component == "Link":
            bus_columns = [
                column
                for column in static.columns
                if column.startswith("bus") and column[3:].isdigit()
            ]
        elif component in ("Line", "Transformer"):
            bus_columns = ["bus0", "bus1"]
        else:
            bus_columns = ["bus"]

        plans = []
        for name, row in static.iterrows():
            connected = [
                row[column]
                for column in bus_columns
                if isinstance(row[column], str) and row[column] in bus_domains
            ]
            if not connected:
                continue
            domains = {bus_domains[bus] for bus in connected}
            heat_domains = {domain for domain in domains if domain.startswith("heat:")}
            if len(heat_domains) > 1 or ("ev" in domains and heat_domains):
                raise ValueError(
                    f"{component} '{name}' connects incompatible profile domains: "
                    f"{sorted(domains)}."
                )
            if heat_domains:
                system = next(iter(heat_domains)).split(":", maxsplit=1)[1]
                shares = heat_shares[system]
            elif "ev" in domains:
                shares = ev_shares
            elif component == "Generator" and row.get("carrier") == "solar rooftop":
                shares = pv_shares
            else:
                shares = general_shares
            classes.append(
                (
                    component,
                    name,
                    row.get("carrier", ""),
                    classify_component(
                        component, row.get("carrier", ""), bool(heat_domains)
                    ),
                )
            )
            shape_key = None
            if component == "Load" and not heat_domains:
                if "ev" in domains and "ev" in profile_shapes:
                    shape_key = "ev"
                elif (
                    domains == {"general"}
                    and "electricity" in profile_shapes
                    and row.get("carrier") in shape_loads
                ):
                    shape_key = "electricity"
            plans.append((name, row, bus_columns, shares, shape_key))

        component_plans[component] = plans
        collisions = {
            f"{name} {profile}"
            for name, _, _, _, _ in plans
            for profile in PROFILES
        } & set(static.index)
        if collisions:
            raise ValueError(
                f"Profile split would overwrite existing {component}(s): "
                f"{sorted(collisions)}."
            )

    clone_names: dict[str, list[str]] = {}
    for bus, profile_names in bus_names.items():
        row = n.buses.loc[bus].drop(
            labels=[
                column
                for column in PROFILE_SPLIT_OUTPUT_COLUMNS
                if column in n.buses.columns
            ]
        )
        for profile, new_bus in profile_names.items():
            n.add("Bus", new_bus, **row.to_dict())

    for component, plans in component_plans.items():
        plural, capacity_attrs, dynamic_scaled_attrs = PROFILE_SPLIT_COMPONENTS[
            component
        ]
        dynamic = getattr(n, f"{plural}_t")
        clone_names[component] = []
        for name, row, bus_columns, shares, shape_key in plans:
            factors = None
            if shape_key:
                load_series = (
                    dynamic["p_set"][name]
                    if name in dynamic["p_set"].columns
                    else pd.Series(float(row["p_set"]), index=n.snapshots)
                )
                factors = _profile_shape_factors(
                    n.snapshots,
                    profile_shapes[shape_key],
                    shares,
                    load_series,
                    n.snapshot_weightings.generators,
                    resolved=_resolved_for(
                        resolved_shapes.get(shape_key), row["bus"]
                    ),
                )
                if shape_key == "electricity":
                    # only the household part follows the shape; the rest keeps
                    # the plain shares, which leaves both conservation rules intact
                    fraction = _household_fraction_for(
                        household_fraction, n, row["bus"]
                    )
                    factors = {
                        p: 1.0 + fraction * (f - 1.0) for p, f in factors.items()
                    }
            for profile in PROFILES:
                share = shares[profile]
                if share == 0.0:
                    continue
                attrs = row.drop(
                    labels=[
                        column
                        for column in PROFILE_SPLIT_OUTPUT_COLUMNS
                        if column in row.index
                    ]
                ).to_dict()
                for column in bus_columns:
                    bus = attrs[column]
                    if bus in bus_names:
                        attrs[column] = bus_names[bus][profile]
                for attribute in capacity_attrs:
                    if attribute in attrs and pd.notna(attrs[attribute]):
                        attrs[attribute] *= share
                if component == "Load":
                    attrs["p_set"] *= share

                new_name = f"{name} {profile}"
                n.add(component, new_name, **attrs)
                clone_names[component].append(new_name)
                for attribute, frame in dynamic.items():
                    if name in frame.columns:
                        values = frame[name]
                        if attribute in dynamic_scaled_attrs:
                            values = values * share
                            if factors is not None and attribute == "p_set":
                                values = values * factors[profile]
                        frame[new_name] = values
                if factors is not None and new_name not in dynamic["p_set"].columns:
                    # constant load: becomes a time series once it has a shape
                    dynamic["p_set"][new_name] = (
                        float(attrs["p_set"]) * factors[profile]
                    )

        _remove(n, component, [name for name, _, _, _, _ in plans])

    _remove(n, "Bus", list(bus_domains))
    suppressed_warnings = release_attr_misleading(attr_filter)
    report_classes(classes, strict=strict_classification)
    check_conservation(totals_before, snapshot_totals(n))
    check_profile_ports(n, PROFILES)
    ev_split_report(n, PROFILES, ev_shares)
    free_capacity = extendable_clone_report(n, clone_names)
    set_report(
        suppressed_attr_warnings=suppressed_warnings,
        shares={
            "general": general_shares,
            "pv": pv_shares,
            "ev": ev_shares,
            "heat": heat_shares,
        },
        strict_classification=strict_classification,
        n_conserved_totals=len(totals_before),
        classes=classes,
        unconstrained_extendable_clones={
            f"{c}|{k}": v for (c, k), v in free_capacity.items()
        },
    )
    logger.info(
        "Applied LV profile split: %d LV buses across %s; shares=%s, "
        "pv_shares=%s, heat_shares=%s, ev_shares=%s.",
        len(low_voltage_buses),
        ", ".join(PROFILES),
        general_shares,
        pv_shares,
        heat_shares,
        ev_shares,
    )
    return n


SYSTEMS = ("urban central", "urban decentral", "rural")


CHAIN_KEYS = ("heat", "water tank", "water pit")


TOL = 1e-6


CAP_ATTRS = ("p_nom", "p_nom_min", "p_nom_max")


def _remove(n, comp, names):
    names = list(names)
    if not names:
        return
    (n.mremove if hasattr(n, "mremove") else n.remove)(comp, names)


def _check_row(name, row, profiles):
    extra = set(row) - set(profiles)
    if extra:
        raise ValueError(f"{name}: unknown profiles {sorted(extra)}")
    vals = {p: float(row.get(p, 0.0)) for p in profiles}
    if min(vals.values()) < 0:
        raise ValueError(f"{name}: negative share in {vals}")
    if abs(sum(vals.values()) - 1.0) > TOL:
        raise ValueError(f"{name}: shares must sum to 1, got {sum(vals.values()):.6g} ({vals})")
    return vals


def validate_shares(profiles, demand_shares, pv_shares, heat_shares, ev_shares):
    d = _check_row("shares", demand_shares, profiles)
    pv = _check_row("pv_shares", pv_shares, profiles)
    ev = _check_row("ev_shares", ev_shares, profiles)
    heat = {}
    for s in SYSTEMS:
        if s not in heat_shares:
            raise ValueError(f"heat_shares is missing system '{s}'")
        heat[s] = _check_row(f"heat_shares[{s}]", heat_shares[s], profiles)
    for p in profiles:
        if d[p] == 0:
            bad = [n_ for n_, v in [("pv_shares", pv[p]), ("ev_shares", ev[p])]
                   + [(f"heat_shares[{s}]", heat[s][p]) for s in SYSTEMS] if v > 0]
            if bad:
                raise ValueError(
                    f"profile '{p}' has demand share 0 (no LV grid connection) but non-zero "
                    f"{bad}: those assets would be stranded. Set them to 0 or give '{p}' demand."
                )
    return d, pv, heat, ev


def _system(carrier):
    """Heat system of a carrier, whatever its prefix ('residential ', 'services ', none).

    Same rule as _profile_split_heat_system; SYSTEMS is ordered so that 'urban central'
    and 'urban decentral' are tested before 'rural'.
    """
    c = str(carrier).lower()
    return next((s for s in SYSTEMS if s in c), None)


def pin_nonhousehold_heat_links(n, pin_profile, profiles):
    """Keep links that touch a heat bus but are not household heating techs on ONE profile."""
    import numpy as np
    if pin_profile not in profiles:
        raise ValueError(f"pin_profile '{pin_profile}' not in profiles {tuple(profiles)}")
    chain = {b for b in n.buses.index
             if _system(n.buses.at[b, "carrier"]) and b.split()[-1] in profiles
             and any(k in str(n.buses.at[b, "carrier"]) for k in CHAIN_KEYS)}
    ports = [c for c in n.links.columns if c.startswith("bus") and c[3:].isdigit()]
    groups = {}
    for l in n.links.index:
        tok = l.split()[-1]
        if tok not in profiles:
            continue
        if not any(n.links.at[l, c] in chain for c in ports):
            continue
        car = str(n.links.at[l, "carrier"])
        if any(sy in car for sy in SYSTEMS):
            continue  # household heating tech: stays split
        groups.setdefault(" ".join(l.split()[:-1]), {})[tok] = l
    removed = []
    for base, clones in groups.items():
        if pin_profile not in clones:
            raise ValueError(
                f"'{base}' has no clone on profile '{pin_profile}' (its heat share there is 0). "
                f"Give '{pin_profile}' a non-zero share for that heat system or choose another pin_profile.")
        keep = clones[pin_profile]
        for a in CAP_ATTRS:
            vals = n.links.loc[list(clones.values()), a].astype(float)
            fin = vals[np.isfinite(vals)]
            if len(fin):
                n.links.at[keep, a] = float(fin.sum())
        drop = [v for k, v in clones.items() if k != pin_profile]
        _remove(n, "Link", drop)
        removed += drop
    logger.info("pin_nonhousehold_heat_links: pinned %d link groups to '%s', removed %d clones",
                len(groups), pin_profile, len(removed))
    return removed


def _drop_buses(n, dead):
    dset = set(dead)
    ports = [c for c in n.links.columns if c.startswith("bus") and c[3:].isdigit()]
    removed = {}
    for comp, plural, cols in (("Load", "loads", ["bus"]), ("Generator", "generators", ["bus"]),
                               ("Store", "stores", ["bus"]), ("Link", "links", ports)):
        static = getattr(n, plural)
        idx = list(static.index[static[cols].isin(dset).any(axis=1)])
        if idx:
            _remove(n, comp, idx)
            removed[comp] = idx
    _remove(n, "Bus", dead)
    logger.info("drop_zero_share_profiles: removed %d buses and %s", len(dead),
                {k: len(v) for k, v in removed.items()})
    return removed


def drop_zero_share_profiles(n, profiles, demand_shares, pv_shares, heat_shares, ev_shares,
                             lv_carriers=("low voltage",), pin_profile="urban"):
    d, pv, heat, ev = validate_shares(profiles, demand_shares, pv_shares, heat_shares, ev_shares)
    dead = []
    for b in n.buses.index:
        tok = b.split()[-1]
        if tok not in profiles:
            continue
        car = str(n.buses.at[b, "carrier"])
        s = _system(car)
        if s and any(k in car for k in CHAIN_KEYS) and heat[s][tok] == 0:
            dead.append(b)
        elif car == "EV battery" and ev[tok] == 0:
            dead.append(b)
        elif car in lv_carriers and d[tok] == 0:
            dead.append(b)
        elif car == "home battery" and d[tok] == 0:
            dead.append(b)
    removed = {}
    if dead:
        removed = _drop_buses(n, dead)
            # YAML "None" is a string; accept every usual spelling of "off"
    if isinstance(pin_profile, str) and pin_profile.strip().lower() in ("", "none", "null", "false", "off"):
        pin_profile = None
        
    if pin_profile:
        pinned = pin_nonhousehold_heat_links(n, pin_profile, profiles)
        if pinned:
            removed.setdefault("Link", []).extend(pinned)
    check_profile_ports(n, profiles)
    return removed
