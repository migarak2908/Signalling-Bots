"""The evolution loop. The algorithm itself is a swappable file in evolvers/ (ga.py, neat.py).

Run from the repo root:  python -m evolve 2col_comm --algorithm ga --neurons 6 --seeds 0 1 2 --generations 1000
Results go to results/<experiment>/<algorithm>_n<neurons>_s<seed>_g<generations>/.
"""

import dataclasses
import importlib
import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from config import Config
from env import Layout, sample_layouts
from experiments import EXPERIMENTS, finish_run, progress, setup
from fitness import fitness

FitnessFn = Callable[[jax.Array, Layout], jax.Array]  # (genome, layouts) -> mean score


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
            progress(gen, cfg.n_generations, clock, "best per seed: " + " ".join(f"{b:.2f}" for b in best[-1]))

    n_elite = round(cfg.pop_size * cfg.elite_fraction)
    order = jnp.argsort(-scores, axis=1)[:, :n_elite]
    history = {"best": np.stack(best, axis=1), "mean": np.stack(mean, axis=1)}
    return jnp.take_along_axis(populations, order[:, :, None], axis=1), history


def main(experiment: str, algorithm: str, n_neurons: int, seeds: list[int], n_generations: int):
    mode, cfg, n_genes = setup(experiment, algorithm=algorithm, n_neurons=n_neurons, n_generations=n_generations)
    elites, history = evolve(lambda genome, layouts: fitness(genome, layouts, cfg, mode), n_genes, cfg, seeds)
    for i, seed in enumerate(seeds):
        run_history = {k: v[i] for k, v in history.items()}
        run_cfg = dataclasses.replace(cfg, seed=seed)
        finish_run(experiment, f"{algorithm}_n{n_neurons}_s{seed}_g{n_generations}", run_cfg, elites[i], run_history)


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
