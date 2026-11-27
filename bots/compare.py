"""%run bots/compare.py 2col_n5_pair_fullinput 2col_n5_pair_plastic --out compare.png

Top: held-out success on each seed's worse colour, one column per results folder.
Bottom: best training fitness over generations, one panel per folder, one line per seed."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SOLVED = 0.9
SMOOTH = 50
BLUE = "#2a6fdb"


def worst_colour(folder: str) -> np.ndarray:
    runs = [json.loads(p.read_text()) for p in Path("results", folder).glob("s*/results.json")]
    return np.array([min(r["success"]) for r in runs])


def best_fitness(folder: str) -> np.ndarray:
    window = np.ones(SMOOTH) / SMOOTH
    histories = Path("results", folder).glob("s*/history.npz")
    return np.stack([np.convolve(np.load(p)["best"], window, mode="valid") for p in histories])


def plot_success(ax, folders: list[str]):
    rng = np.random.default_rng(0)
    for i, folder in enumerate(folders):
        success = worst_colour(folder)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(success)), success, s=40, color=BLUE, zorder=3)
        ax.text(i, 1.06, f"{(success >= SOLVED).sum()} / {len(success)}", ha="center")
    ax.axhline(SOLVED, color="grey", lw=1, ls="--", zorder=1)
    ax.set_xticks(range(len(folders)), folders, fontsize=8)
    ax.set_xlim(-0.5, len(folders) - 0.5)
    ax.set_ylim(-0.05, 1.12)
    ax.set_ylabel("success on the worse colour")
    ax.set_title("Held-out success per seed (top: seeds at least 90% on every colour)", fontsize=10)


def plot_fitness(ax, folder: str):
    best = best_fitness(folder)
    gens = np.arange(best.shape[1]) + SMOOTH
    for seed in best:
        ax.plot(gens, seed, color=BLUE, lw=0.8, alpha=0.35)
    ax.plot(gens, np.median(best, axis=0), color="black", lw=2)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("generation")
    ax.set_title(folder, fontsize=8)


def main(folders: list[str], out: str):
    n = len(folders)
    fig, axes = plt.subplot_mosaic([["success"] * n, folders], figsize=(3.2 * n + 1, 7.5), height_ratios=[1, 0.9])
    plot_success(axes["success"], folders)
    for folder in folders:
        plot_fitness(axes[folder], folder)
    axes[folders[0]].set_ylabel(f"best training fitness\n(seeds thin, median black, {SMOOTH}-gen mean)")
    for ax in axes.values():
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(Path("results", out), dpi=150)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--out", default="compare.png")
    args = parser.parse_args()
    main(args.folders, args.out)
