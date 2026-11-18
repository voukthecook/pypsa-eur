#!/usr/bin/env python
"""Remove the legacy A/B LV split from scripts/prepare_sector_network.py and move the
active three-profile split into scripts/lv_profile_split.py.

Source text of the moved code is copied verbatim (no behaviour change).

Run from the repo root:
    python clean_legacy_ab.py --dry-run     # show what would happen
    python clean_legacy_ab.py               # write files (a .bak backup is made)
"""

import argparse
import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path("scripts/prepare_sector_network.py")
DST = Path("scripts/lv_profile_split.py")

# ---- legacy A/B code to delete -------------------------------------------------
LEGACY_FUNCS = {
    "constrain_lv_links", "_lv_electric_demand", "split_low_voltage_buses",
    "_heat_split_system", "_heat_split_chain_buses", "_heat_split_norm",
    "_remove_heat_components", "_clone_heat_component", "split_heat",
    "_heat_split_totals", "_check_heat_split",
    "_transport_split_chain_buses", "split_transport",
    "_transport_split_totals", "_check_transport_split",
    "drop_zero_share_heat",
}
LEGACY_ASSIGNS = {
    "HEAT_SPLIT_SYSTEMS", "HEAT_SPLIT_HALVES", "HEAT_SPLIT_PLURAL",
    "HEAT_SPLIT_CAPACITY_ATTRS", "HEAT_SPLIT_OUTPUT_ATTRS",
    "TRANSPORT_SPLIT_HALVES", "TRANSPORT_SPLIT_LOAD_CARRIER",
    "TRANSPORT_SPLIT_STORE_CARRIER", "TRANSPORT_SPLIT_LINK_CARRIERS",
    "TRANSPORT_SPLIT_LINK_CAPACITY_ATTRS",
    "_transport_split_norm", "_remove_transport_components", "_clone_transport_component",
}
# module-level string expressions that only document the legacy / moved code
LEGACY_DOC_PREFIXES = ("Split the decentral heating chain",)
MOVED_DOC_PREFIXES = ("Zero-share safety",)
# statements in __main__ that only serve the A/B split
MAIN_LEGACY_ASSIGNS = {"ev_cfg", "heat_cfg", "lv_split_enabled", "heat_split_enabled", "ab_split_enabled"}
MAIN_COMMENT_MARKERS = (
    "# Legacy split calls intentionally remain disabled",
    "# above are retained for reference",
)

# ---- active code moved to the new module (emitted in this order) -----------------
MOVE = [
    "PROFILES", "HEAT_SYSTEMS", "PROFILE_SPLIT_OUTPUT_COLUMNS", "PROFILE_SPLIT_COMPONENTS",
    "_profile_split_shares", "_profile_split_heat_system", "apply_lv_profile_split",
    "SYSTEMS", "CHAIN_KEYS", "TOL", "CAP_ATTRS", "_remove",
    "_check_row", "validate_shares", "_system",
    "pin_nonhousehold_heat_links", "_drop_buses", "drop_zero_share_profiles",
]

NEW_IMPORT = (
    "from scripts.lv_profile_split import (\n"
    "    HEAT_SYSTEMS,\n"
    "    PROFILES,\n"
    "    apply_lv_profile_split,\n"
    "    drop_zero_share_profiles,\n"
    ")\n"
)

GUARD = '''    legacy_ab_enabled = options.get("electricity_distribution_grid_split_lv", False) or (
        options["heating"]
        and options.get("electricity_distribution_grid_split_lv_heat", {}).get(
            "enable", False
        )
    )
    if legacy_ab_enabled:
        raise ValueError(
            "The legacy A/B LV and heat splits have been removed. "
            "Use sector.lv_split_profiles instead."
        )
'''

HEADER = '''# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
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

logger = logging.getLogger(__name__)
'''


def names_of(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def is_str_expr(node):
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def span(node, lines, with_comments=True):
    """0-based [start, end) line span of a node, including directly attached comments."""
    first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    start = first - 1
    if with_comments:
        while start > 0 and lines[start - 1].lstrip().startswith("#"):
            start -= 1
    return start, node.end_lineno


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite an existing new module")
    args = ap.parse_args()

    if not SRC.exists():
        sys.exit(f"{SRC} not found. Run from the repo root.")
    if DST.exists() and not args.force and not args.dry_run:
        sys.exit(f"{DST} already exists. Use --force to overwrite.")

    text = SRC.read_text(encoding="utf-8")
    lines = text.split("\n")
    tree = ast.parse(text)

    delete = set()  # 0-based line indices
    insert_after = {}  # 0-based line index -> text to insert after that line
    report = []  # (action, name, n_lines)
    moved = {}
    logger_seen = False

    def drop(start, end, action, name):
        delete.update(range(start, end))
        report.append((action, name, end - start))

    for node in tree.body:
        nm = names_of(node)

        if is_str_expr(node) and node is not tree.body[0]:
            doc = node.value.value.strip()
            if doc.startswith(LEGACY_DOC_PREFIXES):
                drop(*span(node, lines, False), "delete", "legacy docstring")
            elif doc.startswith(MOVED_DOC_PREFIXES):
                drop(*span(node, lines, False), "delete", "old zero-share docstring (header moves)")
            continue

        if nm == ["logger"]:
            if logger_seen:
                drop(*span(node, lines, False), "delete", "duplicate logger")
            logger_seen = True
            continue

        if any(n in LEGACY_FUNCS or n in LEGACY_ASSIGNS for n in nm):
            drop(*span(node, lines), "delete", ",".join(nm))
            continue

        if len(nm) == 1 and nm[0] in MOVE:
            s, e = span(node, lines)
            moved[nm[0]] = "\n".join(lines[s:e]).rstrip()
            drop(s, e, "move", nm[0])
            continue

        # stray string expression at the end of insert_electricity_distribution_grid
        if isinstance(node, ast.FunctionDef) and node.name == "insert_electricity_distribution_grid":
            last = node.body[-1]
            if len(node.body) > 1 and is_str_expr(last):
                drop(*span(last, lines, False), "delete", "stray docstring in insert_electricity_distribution_grid")

        # __main__ block
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
            for stmt in node.body:
                snames = names_of(stmt)
                if isinstance(stmt, ast.Assign) and any(n in MAIN_LEGACY_ASSIGNS for n in snames):
                    drop(*span(stmt, lines, False), "delete", f"__main__: {','.join(snames)}")
                elif isinstance(stmt, ast.If) and "ab_split_enabled" in ast.unparse(stmt.test):
                    drop(*span(stmt, lines, False), "delete", "__main__: ab_split_enabled check")
                if isinstance(stmt, ast.Assign) and "detailed_grid_enabled" in snames:
                    insert_after[stmt.end_lineno - 1] = GUARD
                    report.append(("insert", "__main__: legacy A/B guard", GUARD.count("\n")))

    # comment lines in __main__ about the disabled legacy calls
    for i, ln in enumerate(lines):
        if ln.strip().startswith(MAIN_COMMENT_MARKERS):
            drop(i, i + 1, "delete", "__main__ comment")

    missing = [n for n in MOVE if n not in moved]
    if missing:
        sys.exit(f"Could not find these active names in {SRC}: {missing}")
    if not any(r[2] for r in report if "legacy A/B guard" in r[1]):
        sys.exit("Did not find 'detailed_grid_enabled' in __main__; cannot place the guard.")

    # ---- report ------------------------------------------------------------------
    print(f"{'action':8s} {'lines':>5s}  name")
    for action, name, n in report:
        print(f"{action:8s} {n:5d}  {name}")
    print(f"\nmoved to {DST}: {len(moved)} definitions")
    if args.dry_run:
        print("dry run: nothing written")
        return

    # ---- build the new prepare_sector_network.py --------------------------------------
    out = []
    for i, ln in enumerate(lines):
        if i not in delete:
            out.append(ln)
        if i in insert_after:
            out.append(insert_after[i].rstrip("\n"))
    new_text = "\n".join(out)

    anchor = "from scripts.prepare_network import"
    if anchor not in new_text:
        sys.exit(f"Import anchor '{anchor}' not found.")
    new_text = new_text.replace(anchor, NEW_IMPORT + anchor, 1)

    for imp, name in (("from collections.abc import Mapping", "Mapping"), ("from typing import Any", "Any")):
        if imp in new_text:
            body = new_text.replace(imp + "\n", "", 1)
            if not re.search(rf"\b{name}\b", body):
                new_text = body
    new_text = re.sub(r"\n{4,}", "\n\n\n", new_text)

    # ---- build the new module ---------------------------------------------------------
    module = HEADER + "\n\n" + "\n\n\n".join(moved[n] for n in MOVE) + "\n"

    for label, src in ((str(SRC), new_text), (str(DST), module)):
        try:
            ast.parse(src)
        except SyntaxError as exc:
            sys.exit(f"Generated {label} does not parse: {exc}")

    backup = SRC.with_suffix(".py.bak")
    shutil.copy2(SRC, backup)
    SRC.write_text(new_text, encoding="utf-8")
    DST.write_text(module, encoding="utf-8")
    print(f"backup: {backup}\nwritten: {SRC}\nwritten: {DST}")

    for target in (SRC, DST):
        try:
            r = subprocess.run([sys.executable, "-m", "pyflakes", str(target)], capture_output=True, text=True)
            if "No module named pyflakes" in r.stderr:
                print("pyflakes not installed: run `ruff check` on both files instead.")
                break
            print(f"pyflakes {target}: {r.stdout.strip() or 'clean'}")
        except Exception as exc:  # pragma: no cover
            print("lint skipped:", exc)


if __name__ == "__main__":
    main()