from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

from bots.config import Config
from bots.env import random_layouts
from bots.rollout import all_trials

COLOURS = ["red", "green", "blue"]


def plot_fitness(out: Path, history: dict):
    gens = np.arange(len(history["best"]))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gens, history["best"], color="black", lw=0.8, label="best")
    ax.plot(gens, history["mean"], color="grey", lw=0.8, label="mean")
    ax.set_xlim(0, gens[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness")
    ax.set_title(f"{out.parent.name}/{out.name}", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fitness.png", dpi=150)
    plt.close(fig)


def plot_trajectories(out: Path, genome: jax.Array, cfg: Config, n_layouts: int = 4):
    """The first few held-out layouts. Rows are layouts, columns are target colours."""
    layouts = random_layouts(jax.random.key(cfg.test_seed), cfg.test_layouts, cfg)[:n_layouts]
    positions = np.asarray(all_trials(genome, layouts, cfg))
    time = np.arange(cfg.n_steps + 1) * cfg.dt
    starts = [cfg.signaller_start, cfg.receiver_start]

    n_cols = cfg.n_colours
    fig, axes = plt.subplots(n_layouts, n_cols, figsize=(4 * n_cols, 2.2 * n_layouts), sharex=True, sharey=True)
    for i in range(n_layouts):
        for c in range(n_cols):
            ax = axes[i, c]
            for colour, x in enumerate(layouts[i]):
                is_target = colour == c
                ax.axhline(x, color=COLOURS[colour], lw=1.5 if is_target else 0.8, ls="-" if is_target else "--")
            path = np.vstack([starts, positions[i, c]])
            ax.plot(time, path[:, 0], color="grey", label="signaller")
            ax.plot(time, path[:, 1], color="black", label="receiver")
            ax.set_ylim(-0.5, 3.5)
            if i == 0:
                ax.set_title(f"target {COLOURS[c]}")

    fig.supxlabel("time")
    fig.supylabel("x")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "trajectories.png", dpi=150)
    plt.close(fig)
