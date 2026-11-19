# SPDX-License-Identifier: MIT
"""Resolve month/daytype/hour profile shapes to real snapshots in local time.

The shape tables (e.g. from LPG) are written in the local clock time of the
household. PyPSA-Eur snapshots are UTC. For every country this module converts
each snapshot to that country's local time (daylight saving included), looks the
shape up there, and averages over the hours a snapshot covers (a 3H snapshot
averages three hourly values instead of sampling one).

Resolution is per country (the first two letters of a node name). A country can
be overridden in the config, e.g. ``timezones: {ES: Atlantic/Canary}``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

PROFILES = ("urban", "suburban", "rural")

COUNTRY_TIMEZONES = {
    "AL": "Europe/Tirane",
    "AT": "Europe/Vienna",
    "BA": "Europe/Sarajevo",
    "BE": "Europe/Brussels",
    "BG": "Europe/Sofia",
    "CH": "Europe/Zurich",
    "CY": "Asia/Nicosia",
    "CZ": "Europe/Prague",
    "DE": "Europe/Berlin",
    "DK": "Europe/Copenhagen",
    "EE": "Europe/Tallinn",
    "ES": "Europe/Madrid",
    "FI": "Europe/Helsinki",
    "FR": "Europe/Paris",
    "GB": "Europe/London",
    "GR": "Europe/Athens",
    "HR": "Europe/Zagreb",
    "HU": "Europe/Budapest",
    "IE": "Europe/Dublin",
    "IT": "Europe/Rome",
    "LT": "Europe/Vilnius",
    "LU": "Europe/Luxembourg",
    "LV": "Europe/Riga",
    "MD": "Europe/Chisinau",
    "ME": "Europe/Podgorica",
    "MK": "Europe/Skopje",
    "NL": "Europe/Amsterdam",
    "NO": "Europe/Oslo",
    "PL": "Europe/Warsaw",
    "PT": "Europe/Lisbon",
    "RO": "Europe/Bucharest",
    "RS": "Europe/Belgrade",
    "SE": "Europe/Stockholm",
    "SI": "Europe/Ljubljana",
    "SK": "Europe/Bratislava",
    "TR": "Europe/Istanbul",
    "UA": "Europe/Kyiv",
    "XK": "Europe/Belgrade",
}


def country_timezone(country: str, overrides: Mapping[str, str] | None = None) -> str:
    """IANA timezone of a country code (config overrides win)."""
    if overrides and country in overrides:
        return overrides[country]
    try:
        return COUNTRY_TIMEZONES[country]
    except KeyError:
        raise KeyError(
            f"No timezone known for country '{country}'. "
            f"Add it under sector.lv_split_profiles.timezones, e.g. {{{country}: Europe/Berlin}}."
        ) from None


def node_country(name: str) -> str:
    """Country code of a PyPSA-Eur node/bus name (its first two letters)."""
    return str(name)[:2]


def local_shape_lookup(
    hours_utc: pd.DatetimeIndex, shape: pd.DataFrame, tz: str
) -> np.ndarray:
    """Raw shape values for hourly UTC timestamps, read at local time in ``tz``."""
    local = hours_utc.tz_localize("UTC").tz_convert(tz)
    keys = pd.MultiIndex.from_arrays(
        [
            local.month,
            np.where(local.dayofweek >= 5, "weekend", "weekday"),
            local.hour,
        ],
        names=["month", "daytype", "hour"],
    )
    raw = shape.reindex(keys)
    if raw.isna().any().any():
        raise ValueError("Profile shape table does not cover every local hour.")
    return raw.to_numpy(dtype=float)


def resolve_shape(
    snapshots: pd.DatetimeIndex,
    durations_h: pd.Series,
    shape: pd.DataFrame,
    countries: Iterable[str],
    overrides: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Per-snapshot shape for each country: columns are (country, profile).

    A snapshot of ``d`` hours averages the hourly local-time values over its window.
    """
    if not isinstance(snapshots, pd.DatetimeIndex):
        raise NotImplementedError("Local-time shapes need plain datetime snapshots.")
    steps = np.rint(durations_h.reindex(snapshots).to_numpy(dtype=float)).astype(int)
    if (steps < 1).any():
        raise ValueError("Snapshot durations must be at least one hour.")
    # one row per covered hour, remembering which snapshot it belongs to
    owner = np.repeat(np.arange(len(snapshots)), steps)
    offset = np.concatenate([np.arange(k) for k in steps])
    hours = pd.DatetimeIndex(snapshots[owner] + pd.to_timedelta(offset, unit="h"))
    counts = steps.astype(float)[:, None]

    frames = {}
    for country in sorted(set(countries)):
        values = local_shape_lookup(hours, shape[list(PROFILES)], country_timezone(country, overrides))
        summed = np.zeros((len(snapshots), len(PROFILES)))
        np.add.at(summed, owner, values)
        frames[country] = pd.DataFrame(
            summed / counts, index=snapshots, columns=list(PROFILES)
        )
    return pd.concat(frames, axis=1)
