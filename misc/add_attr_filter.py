"""Add the PyPSA 'attribute is misleading' warning filter to scripts/lv_profile_split.py.

For files that already have the guards (classification, conservation, report) but not the
filter. Needs the NEW lv_profile_split_guards.py (it defines silence/release_attr_misleading).

    python add_attr_filter.py --dry-run
    python add_attr_filter.py
Skips pieces that are already present; refuses if an anchor is ambiguous. Writes .bak5.
"""
import re
import sys
from pathlib import Path

TARGET = Path("scripts/lv_profile_split.py")
LOGGER_ANCHOR = "logger = logging.getLogger(__name__)\n"


def main():
    dry = "--dry-run" in sys.argv
    text = TARGET.read_text()
    done = []

    if "silence_attr_misleading()" not in text:
        imp = ("from scripts.lv_profile_split_guards import (\n"
               "    release_attr_misleading,\n    silence_attr_misleading,\n)\n\n")
        if text.count(LOGGER_ANCHOR) != 1:
            sys.exit("logger anchor not found exactly once; nothing written")
        text = text.replace(LOGGER_ANCHOR, imp + LOGGER_ANCHOR)
        done.append("import")

        m = list(re.finditer(r"^( *)totals_before = snapshot_totals\(n\)\n", text, re.M))
        if len(m) != 1:
            sys.exit(f"'totals_before = snapshot_totals(n)' found {len(m)}x (need 1); nothing written")
        ind = m[0].group(1)
        text = text[:m[0].end()] + f"{ind}attr_filter = silence_attr_misleading()\n" + text[m[0].end():]
        done.append("install filter")

        m = list(re.finditer(r"^( *)report_classes\(", text, re.M))
        if len(m) != 1:
            sys.exit(f"'report_classes(' call found {len(m)}x (need 1); nothing written")
        ind = m[0].group(1)
        text = text[:m[0].start()] + f"{ind}suppressed_warnings = release_attr_misleading(attr_filter)\n" + text[m[0].start():]
        done.append("release filter")

        if "suppressed_attr_warnings" not in text:
            m = list(re.finditer(r"^( *)set_report\(\n", text, re.M))
            if len(m) == 1:
                ind = m[0].group(1) + "    "
                text = text[:m[0].end()] + f"{ind}suppressed_attr_warnings=suppressed_warnings,\n" + text[m[0].end():]
                done.append("report field")
    else:
        print("filter already installed; nothing to do")
        return

    if dry:
        print(f"{len(done)} edits OK ({', '.join(done)}); dry run, nothing written")
        return
    TARGET.with_suffix(".py.bak5").write_text(TARGET.read_text())
    TARGET.write_text(text)
    print(f"applied: {', '.join(done)}; backup {TARGET}.bak5")


if __name__ == "__main__":
    main()
