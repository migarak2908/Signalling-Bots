"""What evolve.py and train_gd.py share: the experiments, held-out tests, saving and progress lines."""

import dataclasses
import json
import time
from pathlib import Path

import jax
import numpy as np

from config import Config
from env import sample_layouts
from fitness import success, trial_score
from rollout import SCRIPTED_SIGNALS, make_trial, run_grid
from analysis.plot_fitness import plot_fitness
from analysis.plot_trajectories import plot_run

# Experiments: name -> (mode, number of colours). Modes as in rollout.make_trial.
EXPERIMENTS = {
    "2col_nocomm": ("nocomm", 2),
    "2col_comm": ("comm", 2),
    "2col_scripted": ("scripted", 2),
    "2col_scripted_level": ("scripted_level", 2),
    "3col_nocomm": ("nocomm", 3),
    "3col_comm": ("comm", 3),
    "3col_scripted": ("scripted", 3),
    "3col_scripted_level": ("scripted_level", 3),
}


def setup(experiment: str, **changes) -> tuple[str, Config, int]:
    """Returns (mode, config, genome length). Scripted runs train a receiver only."""
    mode, n_colours = EXPERIMENTS[experiment]
    cfg = dataclasses.replace(Config(), n_colours=n_colours, **changes)
    n_genes = cfg.n_params_per_agent if mode in SCRIPTED_SIGNALS else cfg.genome_size
    return mode, cfg, n_genes


def progress(i: int, n: int, clock: float, text: str):
    elapsed = time.time() - clock
    left = elapsed / (i + 1) * (n - i - 1)
    print(f"{i:5d}/{n}  {text}  ({elapsed / 60:.1f} min elapsed, ~{left / 60:.1f} min left)", flush=True)


def heldout_results(genome: jax.Array, cfg: Config, mode: str) -> dict:
    held = sample_layouts(jax.random.key(cfg.heldout_seed), cfg.n_heldout_layouts, cfg)
    positions = run_grid(make_trial(genome, cfg, mode), held, cfg.n_colours)
    scores, successes = trial_score(positions, held.landmark_x, cfg), success(positions, held.landmark_x, cfg)
    target_first = held.landmark_x == held.landmark_x.min(axis=1, keepdims=True)  # target is the leftmost landmark
    return {
        "score": float(scores.mean()),
        "success": float(successes.mean()),
        "score_per_colour": scores.mean(axis=0).tolist(),
        "success_per_colour": successes.mean(axis=0).tolist(),
        "success_target_first": float(successes[target_first].mean()),
        "success_target_not_first": float(successes[~target_first].mean()),
    }


def finish_run(experiment: str, run_name: str, cfg: Config, genomes: jax.Array, history: dict) -> Path:
    """Test genomes[0] (the best) on the held-out set, save the run and plot it."""
    mode = EXPERIMENTS[experiment][0]
    summary = {"heldout": heldout_results(genomes[0], cfg, mode)}
    if mode == "comm":
        summary["heldout_ablated"] = heldout_results(genomes[0], cfg, "nocomm")

    out = Path("results") / experiment / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    np.save(out / "elites.npy", np.asarray(genomes))
    np.savez(out / "history.npz", **history)

    print(f"seed {cfg.seed}: held-out score {summary['heldout']['score']:.3f}  -> {out}", flush=True)
    plot_fitness(out)
    plot_run(out)
    return out
