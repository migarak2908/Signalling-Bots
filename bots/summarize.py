"""%run bots/summarize.py 2col_n5_pair_fullinput"""

import json
import sys
from pathlib import Path

COLOURS = ["red", "green", "blue"]
SOLVED = 0.9
TESTS = {"success": "normal", "success_small_dt": "dt = 0.01", "success_no_signal": "no signal"}


def seed_of(path: Path) -> int:
    return int(path.parent.name.split("_")[0][1:])


def main(folder: str):
    paths = sorted(Path("results", folder).glob("s*/results.json"), key=seed_of)
    runs = [(seed_of(p), json.loads(p.read_text())) for p in paths]
    colours = COLOURS[: len(runs[0][1]["success"])]
    width = 7 * len(colours) + 3

    print("Held-out success: % of trials where the receiver ends within 0.1 of the target\n")
    print(" " * 6 + "".join(name.ljust(width) for name in TESTS.values()))
    print("seed  " + "".join("".join(f"{c:>6} " for c in colours) + "   " for _ in TESTS) + "solved")
    solved = [min(r["success"] + r["success_small_dt"]) >= SOLVED for _, r in runs]
    for (seed, r), ok in zip(runs, solved):
        cells = "".join("".join(f"{v:6.0%} " for v in r[key]) + "   " for key in TESTS)
        print(f"{seed:>4}  {cells}{'yes' if ok else 'no'}")

    print(f"\nsolved: {sum(solved)} of {len(runs)} seeds\n")
    print("normal:    best genome on 100 held-out layouts, each run with every target colour")
    print("dt = 0.01: the same test with a 10x smaller time step; should match normal")
    print("no signal: the signaller isn't told the colour; should drop to about chance")
    print("solved:    at least 90% for every colour, in both normal and dt = 0.01")


if __name__ == "__main__":
    main(sys.argv[1])
