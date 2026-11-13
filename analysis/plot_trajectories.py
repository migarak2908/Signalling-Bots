"""Position vs time. Without arguments: world checks (cheating receiver, random genomes). With a run folder: its best genome.

Run from the repo root:  python -m analysis.plot_trajectories
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from env import sample_layouts
from rollout import rollout, rollout_cheating, run_grid

COLOUR_NAMES = ["red", "green", "blue"]
COLOUR_HEX = ["#c8433a", "#16916a", "#3a6fc8"]  # CVD-checked; lines are also labelled and dashed
INK, MUTED, ZONE = "#222222", "#9a9a9a", "#f1f1ef"
OUT_DIR = Path("results/world_checks")


def plot_trials(positions, layouts, cfg, row_labels, title, path):
    """Grid of panels: one row per layout, one column per target colour."""
    positions, landmark_x = np.asarray(positions), np.asarray(layouts.landmark_x)
    n_rows, n_cols = positions.shape[:2]
    starts = np.array([cfg.signaller_start, cfg.receiver_start])
    time = np.arange(cfg.n_steps + 1) * cfg.dt

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 2.6 * n_rows), sharex=True, sharey=True, squeeze=False)
    for i in range(n_rows):
        for c in range(n_cols):
            ax = axes[i, c]
            ax.axhspan(*cfg.signal_zone, color=ZONE, zorder=0)
            ax.axhspan(*cfg.target_zone, color=ZONE, zorder=0)

            for colour, x in enumerate(landmark_x[i]):
                is_target = colour == c
                ax.axhline(x, color=COLOUR_HEX[colour], lw=1.8 if is_target else 1.0, ls="-" if is_target else "--")
                label = COLOUR_NAMES[colour] + (" (target)" if is_target else "")
                ax.text(time[-1] + 0.4, x, label, va="center", fontsize=8, color=INK if is_target else MUTED)

            path_xy = np.vstack([starts, positions[i, c]])  # prepend the unrecorded start
            ax.plot(time, path_xy[:, 0], color=MUTED, lw=1.2, label="signaller")
            ax.plot(time, path_xy[:, 1], color=INK, lw=1.5, label="receiver")

            ax.set_title(f"{row_labels[i]}, target {COLOUR_NAMES[c]}", fontsize=9, loc="left", color=INK)
            ax.set_ylim(-0.5, 3.5)
            ax.set_xlim(0, time[-1])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(labelsize=8, colors=MUTED)

    fig.supxlabel("time", fontsize=9)
    fig.supylabel("position x", fontsize=9)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=11, x=0.02, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def main():
    cfg = Config()
    n_examples = 3
    layouts = sample_layouts(jax.random.key(cfg.seed), n_examples, cfg)

    cheating = run_grid(lambda l, c: rollout_cheating(l, c, cfg), layouts, cfg.n_colours)
    plot_trials(
        cheating, layouts, cfg,
        row_labels=[f"layout {i}" for i in range(n_examples)],
        title="Cheating receiver: should stop at the target landmark",
        path=OUT_DIR / "cheating_receiver.png",
    )

    # One random genome per layout: row i runs genome i on layout i.
    genomes = jax.random.uniform(jax.random.key(cfg.seed + 1), (n_examples, cfg.genome_size), minval=-1, maxval=1)
    colours = jnp.arange(cfg.n_colours)
    random_runs = jax.vmap(
        lambda g, l: jax.vmap(lambda c: rollout(g, l, c, cfg))(colours)
    )(genomes, layouts)
    plot_trials(
        random_runs, layouts, cfg,
        row_labels=[f"random genome {i}" for i in range(n_examples)],
        title="Random genomes (y-axis cut at -0.5 and 3.5; runaway receivers leave the frame)",
        path=OUT_DIR / "random_genomes.png",
    )


def plot_run(run_dir: Path, n_examples: int = 4):
    """Best genome of a saved run on the first few held-out layouts."""
    from analysis.plot_fitness import load_config

    cfg = load_config(run_dir)
    genome = jnp.asarray(np.load(run_dir / "elites.npy")[0])
    layouts = sample_layouts(jax.random.key(cfg.heldout_seed), n_examples, cfg)
    positions = run_grid(lambda l, c: rollout(genome, l, c, cfg), layouts, cfg.n_colours)
    plot_trials(
        positions, layouts, cfg,
        row_labels=[f"held-out layout {i}" for i in range(n_examples)],
        title=f"{run_dir.parent.name}/{run_dir.name}: best genome",
        path=run_dir / "trajectories.png",
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        plot_run(Path(sys.argv[1]))
    else:
        main()
