#!/usr/bin/env python3
"""Aggregate .corrections.log files into a frequency table.

Usage:
    python scripts/corrections_report.py <directory>
"""

import re
import sys
from collections import Counter
from pathlib import Path

LINE = re.compile(r"^(.*?) => (.*?)  \((\d+)\)$")


def main(argv):
    if len(argv) < 2:
        print("usage: corrections_report.py <directory>", file=sys.stderr)
        return 2

    root = Path(argv[1])
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    totals = Counter()
    logs = list(root.rglob("*.corrections.log"))

    for path in logs:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = LINE.match(line.rstrip("\r\n"))
                if m:
                    totals[(m.group(1), m.group(2))] += int(m.group(3))

    print(f"{len(logs)} log file(s) found under {root}")
    if not totals:
        print("no corrections recorded")
        return 0

    print(f"{sum(totals.values())} replacement(s)\n")
    print(f"{'count':>6}  rule")
    print("-" * 72)
    for (frm, to), n in totals.most_common():
        print(f"{n:>6}  {frm} => {to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
