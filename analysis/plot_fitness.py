"""Fitness curves for an evolution run: best and mean score per generation.

Run from the repo root:  python -m analysis.plot_fitness results/2col_comm/s4_g3000
"""

import json
import sys
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from env import sample_layouts
from fitness import random_landmark_baseline

INK, MUTED, REFERENCE = "#222222", "#9a9a9a", "#6b6b6b"
SMOOTH = 50  # generations in the rolling average of the best score


def load_config(run_dir: Path) -> Config:
    raw = json.loads((run_dir / "config.json").read_text())
    return Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_fitness(run_dir: Path) -> Path:
    cfg = load_config(run_dir)
    history = np.load(run_dir / "history.npz")
    summary = json.loads((run_dir / "summary.json").read_text())
    held = sample_layouts(jax.random.key(cfg.heldout_seed), cfg.n_heldout_layouts, cfg)
    guess_score = float(random_landmark_baseline(held, cfg)[0].mean())

    best, mean = history["best"], history["mean"]
    gens = np.arange(len(best))
    window = min(SMOOTH, len(best))
    smooth_gens = gens[window - 1:]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(gens, best, color=INK, lw=0.6, alpha=0.3)
    ax.plot(smooth_gens, rolling_mean(best, window), color=INK, lw=1.8, label=f"best ({window}-gen average)")
    ax.plot(gens, mean, color=MUTED, lw=1.2, label="population mean")

    heldout = summary["heldout"]["score"]
    ax.axhline(heldout, color=REFERENCE, lw=1, ls=":", label=f"held-out score of final best ({heldout:.2f})")
    if "heldout_ablated" in summary:
        ablated = summary["heldout_ablated"]["score"]
        ax.axhline(ablated, color=REFERENCE, lw=1, ls="-.", label=f"same genome, colour hidden ({ablated:.2f})")
    ax.axhline(guess_score, color=REFERENCE, lw=1, ls="--", label=f"random-landmark guess ({guess_score:.2f})")

    ax.set_xlim(0, gens[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("generation", fontsize=9)
    ax.set_ylabel("fitness (mean trial score)", fontsize=9)
    ax.set_title(f"{run_dir.parent.name}/{run_dir.name}: fitness over generations", fontsize=11, loc="left", color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()

    path = run_dir / "fitness.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", plot_fitness(Path(sys.argv[1])))
