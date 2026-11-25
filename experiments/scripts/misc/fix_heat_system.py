"""Make heat-system detection prefix-agnostic in scripts/lv_profile_split.py.

Bug: _system() only stripped a leading "services ", so carriers such as
"residential rural heat" / "residential urban decentral heat" returned None. Their
zero-share profile buses were never dropped and the pin step did not see them.
Fix: look for the system name anywhere in the carrier (same rule as the split itself).

    python fix_heat_system.py --dry-run
    python fix_heat_system.py
"""
import sys
from pathlib import Path

TARGET = Path("scripts/lv_profile_split.py")
OLD = '''def _system(carrier):
    # services heat buses are merged into the residential ones by cluster_heat_buses later
    c = str(carrier).lower()
    if c.startswith("services "):
        c = c[len("services "):]
    return next((s for s in SYSTEMS if c.startswith(s)), None)
'''
NEW = '''def _system(carrier):
    """Heat system of a carrier, whatever its prefix ('residential ', 'services ', none).

    Same rule as _profile_split_heat_system; SYSTEMS is ordered so that 'urban central'
    and 'urban decentral' are tested before 'rural'.
    """
    c = str(carrier).lower()
    return next((s for s in SYSTEMS if s in c), None)
'''


def main():
    text = TARGET.read_text()
    if text.count(OLD) != 1:
        sys.exit(f"anchor found {text.count(OLD)}x (need exactly 1); nothing written")
    if "--dry-run" in sys.argv:
        print("1 edit OK (dry run, nothing written)")
        return
    TARGET.with_suffix(".py.bak4").write_text(text)
    TARGET.write_text(text.replace(OLD, NEW))
    print(f"edit applied to {TARGET}; backup {TARGET}.bak4")


if __name__ == "__main__":
    main()