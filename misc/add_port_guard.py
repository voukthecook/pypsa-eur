"""Wire the port guard + EV share check into scripts/lv_profile_split.py.

Usage (from the pypsa-eur folder, after add_guards.py has been applied):
    python add_port_guard.py --dry-run
    python add_port_guard.py
Also copy the new lv_profile_split_guards.py over the old one first.
Every edit is an exact-text anchor that must occur exactly once; a .bak3 backup is written.
"""
import sys
from pathlib import Path

TARGET = Path("scripts/lv_profile_split.py")

EDITS = [
    ("logger = logging.getLogger(__name__)\n",
     "from scripts.lv_profile_split_guards import (\n    check_profile_ports,\n    ev_split_report,\n)\n\n"
     "logger = logging.getLogger(__name__)\n"),
    ("    check_conservation(totals_before, snapshot_totals(n))\n",
     "    check_conservation(totals_before, snapshot_totals(n))\n"
     "    check_profile_ports(n, PROFILES)\n"
     "    ev_split_report(n, PROFILES, ev_shares)\n"),
    ("            removed.setdefault(\"Link\", []).extend(pinned)\n    return removed\n",
     "            removed.setdefault(\"Link\", []).extend(pinned)\n"
     "    check_profile_ports(n, profiles)\n"
     "    return removed\n"),
]


def main():
    dry = "--dry-run" in sys.argv
    text = TARGET.read_text()
    for i, (old, new) in enumerate(EDITS, 1):
        if text.count(old) != 1:
            sys.exit(f"edit {i}: anchor found {text.count(old)}x (need exactly 1); nothing written")
        text = text.replace(old, new)
    if dry:
        print(f"{len(EDITS)} edits OK (dry run, nothing written)")
        return
    TARGET.with_suffix(".py.bak3").write_text(TARGET.read_text())
    TARGET.write_text(text)
    print(f"{len(EDITS)} edits applied to {TARGET}; backup {TARGET}.bak3")


if __name__ == "__main__":
    main()
