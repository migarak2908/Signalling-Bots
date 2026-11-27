"""%run bots/compare.py 2col_n5_pair_fullinput 2col_n5_pair_plastic

Held-out success on each seed's worse colour, one column per results folder."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SOLVED = 0.9


def worst_colour(folder: str) -> np.ndarray:
    runs = [json.loads(p.read_text()) for p in Path("results", folder).glob("s*/results.json")]
    return np.array([min(r["success"]) for r in runs])


def main(folders: list[str]):
    fig, ax = plt.subplots(figsize=(1.8 * len(folders) + 1, 4))
    rng = np.random.default_rng(0)
    for i, folder in enumerate(folders):
        success = worst_colour(folder)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(success)), success, s=40, color="#2a6fdb", zorder=3)
        ax.text(i, 1.06, f"{(success >= SOLVED).sum()} / {len(success)}", ha="center")

    ax.axhline(SOLVED, color="grey", lw=1, ls="--", zorder=1)
    ax.set_xticks(range(len(folders)), [f.replace("_", "\n") for f in folders])
    ax.set_xlim(-0.5, len(folders) - 0.5)
    ax.set_ylim(-0.05, 1.12)
    ax.set_ylabel("success on the worse colour")
    ax.set_title("Held-out success per seed (top: seeds at least 90% on every colour)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("results/compare.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
