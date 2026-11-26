"""python -m bots.summarize 3col"""

import json
import sys
from pathlib import Path


def fmt(values):
    return " / ".join(f"{v:.2f}" for v in values)


def main(folder: str):
    header = ["run", "success", "success dt=0.01", "no signal", "solved"]
    rows = []
    for path in sorted(Path("results", folder).glob("*/results.json")):
        r = json.loads(path.read_text())
        rows.append([path.parent.name, fmt(r["success"]), fmt(r["success_small_dt"]), fmt(r["success_no_signal"]), str(r["solved"])])

    widths = [max(len(row[i]) for row in rows + [header]) for i in range(len(header))]
    for row in [header] + rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))
    print(f"\nsolved: {sum(row[-1] == 'True' for row in rows)} of {len(rows)}")


if __name__ == "__main__":
    main(sys.argv[1])
