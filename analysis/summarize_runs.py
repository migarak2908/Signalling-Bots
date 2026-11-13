"""One table of held-out results for every saved run.

Run from the repo root:  python -m analysis.summarize_runs [experiment filter, e.g. 3col]
"""

import json
import sys
from pathlib import Path


def main(name_filter: str = ""):
    rows = []
    for summary_path in sorted(Path("results").glob(f"*{name_filter}*/*/summary.json")):
        summary = json.loads(summary_path.read_text())
        held, ablated = summary["heldout"], summary.get("heldout_ablated")
        rows.append([
            f"{summary_path.parent.parent.name}/{summary_path.parent.name}",
            f"{held['score']:.2f}",
            f"{held['success']:.0%}",
            " / ".join(f"{s:.2f}" for s in held["score_per_colour"]),
            f"{held['success_target_first']:.0%}",
            f"{held['success_target_not_first']:.0%}",
            f"{ablated['score']:.2f}" if ablated else "-",
        ])

    header = ["run", "score", "success", "score per colour", "succ: target 1st", "succ: target not 1st", "ablated"]
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    for row in [header] + rows:
        print("  ".join(str(cell).ljust(w) for cell, w in zip(row, widths)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
