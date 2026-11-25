"""Show which guard patches are present in your repo. Run from the pypsa-eur folder:
    python check_install.py
Prints OK / MISSING per piece and says which patcher fixes it."""
from pathlib import Path

CHECKS = {
    "scripts/lv_profile_split.py": [
        ("guards: silence filter installed", "silence_attr_misleading()", "add_guards.py"),
        ("guards: filter released + summary", "release_attr_misleading(attr_filter)", "add_guards.py"),
        ("guards: classification", "classify_component(", "add_guards.py"),
        ("guards: conservation check", "check_conservation(totals_before", "add_guards.py"),
        ("guards: zero-share clones skipped", "if share == 0.0:", "add_guards.py"),
        ("guards: run report data", "set_report(", "add_guards.py"),
        ("port guard (after cloning)", "check_profile_ports(n, PROFILES)", "add_port_guard.py"),
        ("EV share check", "ev_split_report(n, PROFILES, ev_shares)", "add_port_guard.py"),
        ("port guard (after drop/pin)", "check_profile_ports(n, profiles)", "add_port_guard.py"),
        ("heat-system prefix fix", "return next((s for s in SYSTEMS if s in c), None)", "fix_heat_system.py"),
    ],
    "scripts/prepare_sector_network.py": [
        ("report written to run folder", "write_report(", "add_guards.py"),
        ("drop_zero result kept", "removed = drop_zero_share_profiles(", "add_guards.py"),
    ],
    "scripts/lv_profile_split_guards.py": [
        ("new guards module (port + EV + orphan)", "def orphan_split_buses", "copy the new lv_profile_split_guards.py"),
        ("heat vent is household", '"heat vent"', "copy the new lv_profile_split_guards.py"),
    ],
}

bad = 0
for path, items in CHECKS.items():
    p = Path(path)
    if not p.exists():
        print(f"{path}: FILE NOT FOUND"); bad += 1; continue
    text = p.read_text(encoding="utf-8", errors="ignore")
    print(path)
    for label, marker, fix in items:
        ok = marker in text
        bad += not ok
        print(f"  {'OK     ' if ok else 'MISSING'} {label}" + ("" if ok else f"   -> {fix}"))
print("\nall pieces present" if not bad else f"\n{bad} piece(s) missing")