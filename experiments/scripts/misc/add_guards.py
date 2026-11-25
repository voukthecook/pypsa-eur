"""Wire lv_profile_split_guards into scripts/lv_profile_split.py (run from repo root).

    python add_guards.py --dry-run   # report only
    python add_guards.py             # apply (backup: lv_profile_split.py.bak2)

Every edit is an exact-text replacement that must match exactly once, so the script
refuses (and changes nothing) if the file differs from what the cleanup produced.
"""
import argparse
import shutil
import sys
from pathlib import Path

PATH = Path("scripts/lv_profile_split.py")
MARK = "lv_profile_split_guards"

EDITS = [
    (  # 1 imports
        "import pypsa\n\nlogger",
        "import pypsa\n\nfrom scripts.lv_profile_split_guards import (\n"
        "    check_conservation,\n    classify_component,\n    extendable_clone_report,\n"
        "    report_classes,\n    set_report,\n    snapshot_totals,\n)\n\nlogger",
    ),
    (  # 2 snapshot before cloning
        '    if not config.get("enable", False):\n        return n\n\n    uniform =',
        '    if not config.get("enable", False):\n        return n\n\n'
        "    totals_before = snapshot_totals(n)\n"
        '    strict_classification = bool(config.get("strict_classification", False))\n\n'
        "    uniform =",
    ),
    (  # 3 classify while planning
        "    component_plans = {}\n",
        "    component_plans = {}\n    classes = []\n",
    ),
    (
        "            plans.append((name, row, bus_columns, shares))\n",
        "            classes.append(\n"
        "                (\n                    component,\n                    name,\n"
        '                    row.get("carrier", ""),\n'
        "                    classify_component(\n"
        '                        component, row.get("carrier", ""), bool(heat_domains)\n'
        "                    ),\n                )\n            )\n"
        "            plans.append((name, row, bus_columns, shares))\n",
    ),
    (  # 4 skip zero shares, remember clone names
        "        dynamic = getattr(n, f\"{plural}_t\")\n        for name, row, bus_columns, shares in plans:\n"
        "            for profile in PROFILES:\n                share = shares[profile]\n",
        "        dynamic = getattr(n, f\"{plural}_t\")\n        clone_names[component] = []\n"
        "        for name, row, bus_columns, shares in plans:\n"
        "            for profile in PROFILES:\n                share = shares[profile]\n"
        "                if share == 0.0:\n                    continue\n",
    ),
    (
        "                n.add(component, new_name, **attrs)\n",
        "                n.add(component, new_name, **attrs)\n"
        "                clone_names[component].append(new_name)\n",
    ),
    (  # clone_names defined before the clone loop
        "    for bus, profile_names in bus_names.items():\n        row = n.buses.loc[bus].drop(",
        "    clone_names: dict[str, list[str]] = {}\n"
        "    for bus, profile_names in bus_names.items():\n        row = n.buses.loc[bus].drop(",
    ),
    (  # 5 checks after cloning
        '    _remove(n, "Bus", list(bus_domains))\n    logger.info(\n        "Applied LV profile split',
        '    _remove(n, "Bus", list(bus_domains))\n'
        "    report_classes(classes, strict=strict_classification)\n"
        "    check_conservation(totals_before, snapshot_totals(n))\n"
        "    free_capacity = extendable_clone_report(n, clone_names)\n"
        "    set_report(\n"
        "        shares={\n            \"general\": general_shares,\n            \"pv\": pv_shares,\n"
        "            \"ev\": ev_shares,\n            \"heat\": heat_shares,\n        },\n"
        "        strict_classification=strict_classification,\n"
        "        n_conserved_totals=len(totals_before),\n"
        "        classes=classes,\n"
        "        unconstrained_extendable_clones={\n"
        "            f\"{c}|{k}\": v for (c, k), v in free_capacity.items()\n        },\n"
        "    )\n"
        '    logger.info(\n        "Applied LV profile split',
    ),
]


PSN = Path("scripts/prepare_sector_network.py")
PSN_EDITS = [
    (
        "    drop_zero_share_profiles,\n)\n",
        "    drop_zero_share_profiles,\n)\nfrom scripts.lv_profile_split_guards import write_report\n",
    ),
    (
        "        drop_zero_share_profiles(\n            n,\n            profiles=profiles,",
        "        removed = drop_zero_share_profiles(\n            n,\n            profiles=profiles,",
    ),
    (
        '            pin_profile=profiles_cfg.get("pin_profile", "urban"),\n        )\n',
        '            pin_profile=profiles_cfg.get("pin_profile", "urban"),\n        )\n'
        "        write_report(snakemake.log[0], removed)\n",
    ),
]


def apply(path, edits, src):
    out = src
    for i, (old, new) in enumerate(edits, 1):
        if out.count(old) != 1:
            sys.exit(f"{path} edit {i}: anchor found {out.count(old)}x (need 1); nothing changed")
        out = out.replace(old, new)
    compile(out, str(path), "exec")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    srcs = {PATH: PATH.read_text(), PSN: PSN.read_text()}
    if MARK in srcs[PATH] or MARK in srcs[PSN]:
        sys.exit("already patched")
    outs = {PATH: apply(PATH, EDITS, srcs[PATH]), PSN: apply(PSN, PSN_EDITS, srcs[PSN])}
    print(f"{len(EDITS)} + {len(PSN_EDITS)} edits OK")
    if args.dry_run:
        return
    for path, out in outs.items():
        shutil.copy(path, path.with_suffix(".py.bak2"))
        path.write_text(out)
        print("written", path)


main()