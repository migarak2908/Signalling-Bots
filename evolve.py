"""The evolution loop. The algorithm itself is a swappable file in evolvers/ (ga.py, neat.py).

Run from the repo root:  python -m evolve 2col_comm --algorithm ga --neurons 6 --seeds 0 1 2 --generations 1000
Results go to results/<experiment>/<algorithm>_n<neurons>_s<seed>_g<generations>/.
"""

import dataclasses
import importlib
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


# --- The loop.


def evolve(
    fitness_fn: FitnessFn, n_genes: int, cfg: Config, seeds: list[int], log_every: int = 50
) -> tuple[jax.Array, dict]:
    """Evolve one independent population per seed, all at once (vmap over seeds).

    Each evolver (evolvers/<cfg.algorithm>.py) provides init, population and next_generation.
    Returns (best genomes of the last scored generation, shape (n_seeds, n_elite, n_genes), best-first;
             history with "best" and "mean" scores, shape (n_seeds, n_generations)).
    """
    evolver = importlib.import_module(f"evolvers.{cfg.algorithm}")

    def start(seed):
        key, init_key = jax.random.split(jax.random.key(seed))
        return key, evolver.init(init_key, n_genes, cfg)

    def generation(key, state):
        key, gen_key = jax.random.split(key)
        layout_key, mutate_key = jax.random.split(gen_key)
        layouts = sample_layouts(layout_key, cfg.n_layouts, cfg)  # same trials for every genome in this population
        population = evolver.population(state)
        scores = jax.vmap(fitness_fn, in_axes=(0, None))(population, layouts)
        return key, evolver.next_generation(mutate_key, state, scores, cfg), population, scores

    keys, states = jax.vmap(start)(jnp.array(seeds))
    step_all = jax.jit(jax.vmap(generation))

    best, mean = [], []
    print(f"{len(seeds)} seed(s), {cfg.n_generations} generations; compiling (first generation is slow)...", flush=True)
    clock = time.time()
    for gen in range(cfg.n_generations):
        keys, states, populations, scores = step_all(keys, states)
        best.append(scores.max(axis=1))
        mean.append(scores.mean(axis=1))
        if gen % log_every == 0 or gen == cfg.n_generations - 1:
            per_seed = " ".join(f"{b:.2f}" for b in best[-1])
            elapsed = time.time() - clock
            left = elapsed / (gen + 1) * (cfg.n_generations - gen - 1)
            print(f"gen {gen:4d}/{cfg.n_generations}  best per seed: {per_seed}  "
                  f"({elapsed / 60:.1f} min elapsed, ~{left / 60:.1f} min left)", flush=True)

    n_elite = round(cfg.pop_size * cfg.elite_fraction)
    order = jnp.argsort(-scores, axis=1)[:, :n_elite]
    history = {"best": np.stack(best, axis=1), "mean": np.stack(mean, axis=1)}
    return jnp.take_along_axis(populations, order[:, :, None], axis=1), history


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


def main(name: str, algorithm: str, n_neurons: int, seeds: list[int], n_generations: int):
    hide, n_colours = EXPERIMENTS[name]
    cfg = dataclasses.replace(
        Config(), n_generations=n_generations, n_colours=n_colours, algorithm=algorithm, n_neurons=n_neurons
    )
    elites, history = evolve(lambda genome, layouts: fitness(genome, layouts, cfg, hide), cfg.genome_size, cfg, seeds)

    for i, seed in enumerate(seeds):
        run_cfg = dataclasses.replace(cfg, seed=seed)
        best = elites[i, 0]
        summary = {"heldout": heldout_results(best, run_cfg, hide)}
        if not hide:
            summary["heldout_ablated"] = heldout_results(best, run_cfg, hide_colour=True)
        run_history = {k: v[i] for k, v in history.items()}
        run_name = f"{algorithm}_n{n_neurons}_s{seed}_g{n_generations}"
        out = save_run(Path("results") / name / run_name, run_cfg, elites[i], run_history, summary)
        print(f"seed {seed}: held-out score {summary['heldout']['score']:.3f}  -> {out}", flush=True)
        plot_fitness(out)
        plot_run(out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--algorithm", choices=["ga", "neat"], default=Config.algorithm)
    parser.add_argument("--neurons", type=int, default=Config.n_neurons)
    parser.add_argument("--seeds", type=int, nargs="+", default=[Config.seed])
    parser.add_argument("--generations", type=int, default=Config.n_generations)
    args = parser.parse_args()
    main(args.experiment, args.algorithm, args.neurons, args.seeds, args.generations)
