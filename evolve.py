"""Evolution: truncation selection with elitism, Gaussian mutation, no crossover.

Run from the repo root:  python -m evolve 2col_comm --seeds 0 1 2 3 4 --generations 1000
Results go to results/<experiment>/s<seed>_g<generations>/.
"""

import dataclasses
import json
import time
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from config import Config
from env import Layout, sample_layouts
from fitness import fitness, success, trial_score
from rollout import rollout, run_grid
from analysis.plot_fitness import plot_fitness
from analysis.plot_trajectories import plot_run

FitnessFn = Callable[[jax.Array, Layout], jax.Array]  # (genome, layouts) -> mean score


# --- The evolver. Swap these two functions to change the algorithm.


def init_population(key: jax.Array, n_genes: int, cfg: Config) -> jax.Array:
    return jax.random.uniform(key, (cfg.pop_size, n_genes), minval=-1.0, maxval=1.0)


def next_generation(key: jax.Array, population: jax.Array, scores: jax.Array, cfg: Config) -> jax.Array:
    """Elites first (best-first, unchanged), then mutated copies of them, each elite equally often."""
    n_elite = round(cfg.pop_size * cfg.elite_fraction)
    elites = population[jnp.argsort(-scores)[:n_elite]]
    parents = elites[jnp.arange(cfg.pop_size - n_elite) % n_elite]
    children = parents + cfg.mutation_std * jax.random.normal(key, parents.shape)
    return jnp.concatenate([elites, jnp.clip(children, -1.0, 1.0)])


# --- The loop.


def evolve(
    fitness_fn: FitnessFn, n_genes: int, cfg: Config, seeds: list[int], log_every: int = 50
) -> tuple[jax.Array, dict]:
    """Evolve one independent population per seed, all at once (vmap over seeds).

    Returns (elites, shape (n_seeds, n_elite, n_genes), best-first;
             history with "best" and "mean" scores, shape (n_seeds, n_generations)).
    """

    def start(seed):
        key, init_key = jax.random.split(jax.random.key(seed))
        return key, init_population(init_key, n_genes, cfg)

    def generation(key, population):
        key, gen_key = jax.random.split(key)
        layout_key, mutate_key = jax.random.split(gen_key)
        layouts = sample_layouts(layout_key, cfg.n_layouts, cfg)  # same trials for every genome in this population
        scores = jax.vmap(fitness_fn, in_axes=(0, None))(population, layouts)
        return key, next_generation(mutate_key, population, scores, cfg), scores

    keys, populations = jax.vmap(start)(jnp.array(seeds))
    step_all = jax.jit(jax.vmap(generation))

    best, mean = [], []
    print(f"{len(seeds)} seed(s), {cfg.n_generations} generations; compiling (first generation is slow)...", flush=True)
    clock = time.time()
    for gen in range(cfg.n_generations):
        keys, populations, scores = step_all(keys, populations)
        best.append(scores.max(axis=1))
        mean.append(scores.mean(axis=1))
        if gen % log_every == 0 or gen == cfg.n_generations - 1:
            per_seed = " ".join(f"{b:.2f}" for b in best[-1])
            elapsed = time.time() - clock
            left = elapsed / (gen + 1) * (cfg.n_generations - gen - 1)
            print(f"gen {gen:4d}/{cfg.n_generations}  best per seed: {per_seed}  "
                  f"({elapsed / 60:.1f} min elapsed, ~{left / 60:.1f} min left)", flush=True)

    n_elite = round(cfg.pop_size * cfg.elite_fraction)
    history = {"best": np.stack(best, axis=1), "mean": np.stack(mean, axis=1)}
    return populations[:, :n_elite], history


def save_run(out: Path, cfg: Config, elites: jax.Array, history: dict, summary: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    np.save(out / "elites.npy", np.asarray(elites))
    np.savez(out / "history.npz", **history)
    return out


def heldout_results(genome: jax.Array, cfg: Config, hide_colour: bool) -> dict:
    held = sample_layouts(jax.random.key(cfg.heldout_seed), cfg.n_heldout_layouts, cfg)
    positions = run_grid(lambda l, c: rollout(genome, l, c, cfg, hide_colour), held, cfg.n_colours)
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


# Experiments: name -> (hide the signaller's colour inputs?, number of colours)
EXPERIMENTS = {
    "2col_nocomm": (True, 2),
    "2col_comm": (False, 2),
    "3col_nocomm": (True, 3),
    "3col_comm": (False, 3),
}


def main(name: str, seeds: list[int], n_generations: int):
    hide, n_colours = EXPERIMENTS[name]
    cfg = dataclasses.replace(Config(), n_generations=n_generations, n_colours=n_colours)
    elites, history = evolve(lambda genome, layouts: fitness(genome, layouts, cfg, hide), cfg.genome_size, cfg, seeds)

    for i, seed in enumerate(seeds):
        run_cfg = dataclasses.replace(cfg, seed=seed)
        best = elites[i, 0]
        summary = {"heldout": heldout_results(best, run_cfg, hide)}
        if not hide:
            summary["heldout_ablated"] = heldout_results(best, run_cfg, hide_colour=True)
        run_history = {k: v[i] for k, v in history.items()}
        out = save_run(Path("results") / name / f"s{seed}_g{n_generations}", run_cfg, elites[i], run_history, summary)
        print(f"seed {seed}: held-out score {summary['heldout']['score']:.3f}  -> {out}", flush=True)
        plot_fitness(out)
        plot_run(out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[Config.seed])
    parser.add_argument("--generations", type=int, default=Config.n_generations)
    args = parser.parse_args()
    main(args.experiment, args.seeds, args.generations)
