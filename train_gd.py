"""Gradient descent on the same genome the GA evolves: backprop through the whole trial, with Adam.

Run from the repo root:  python -m train_gd 3col_scripted --neurons 6 --seeds 10 11 --steps 3000
Results go to results/<experiment>/gd_n<neurons>_s<seed>_steps<steps>/.
"""

import dataclasses
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from config import Config
from env import sample_layouts
from experiments import EXPERIMENTS, finish_run, progress, setup
from fitness import distance_loss, trial_score
from rollout import make_trial, run_grid


def train(mode: str, n_genes: int, cfg: Config, seeds: list[int], log_every: int = 50) -> tuple[jax.Array, dict]:
    """Train one genome per seed, all at once (vmap over seeds). Fresh layouts every step.

    Returns (final genomes, shape (n_seeds, n_genes); history with "best" (the training score) and "loss",
    shape (n_seeds, n_gd_steps)).
    """
    optimiser = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.learning_rate))

    def loss_and_score(genes, layouts):
        positions = run_grid(make_trial(genes, cfg, mode), layouts, cfg.n_colours)
        return distance_loss(positions, layouts.landmark_x, cfg), trial_score(positions, layouts.landmark_x, cfg).mean()

    def start(seed):
        key, init_key = jax.random.split(jax.random.key(seed))
        genes = jax.random.uniform(init_key, (n_genes,), minval=-1.0, maxval=1.0)
        return key, genes, optimiser.init(genes)

    def update(key, genes, opt_state):
        key, layout_key = jax.random.split(key)
        layouts = sample_layouts(layout_key, cfg.n_layouts, cfg)
        (loss, score), grads = jax.value_and_grad(loss_and_score, has_aux=True)(genes, layouts)
        changes, opt_state = optimiser.update(grads, opt_state)
        genes = jnp.clip(optax.apply_updates(genes, changes), -1.0, 1.0)  # stay a valid genome
        return key, genes, opt_state, loss, score

    keys, genes, opt_states = jax.vmap(start)(jnp.array(seeds))
    update_all = jax.jit(jax.vmap(update))

    scores, losses = [], []
    print(f"{len(seeds)} seed(s), {cfg.n_gd_steps} steps; compiling (first step is slow)...", flush=True)
    clock = time.time()
    for i in range(cfg.n_gd_steps):
        keys, genes, opt_states, loss, score = update_all(keys, genes, opt_states)
        scores.append(score)
        losses.append(loss)
        if i % log_every == 0 or i == cfg.n_gd_steps - 1:
            progress(i, cfg.n_gd_steps, clock, "score per seed: " + " ".join(f"{s:.2f}" for s in score))

    return genes, {"best": np.stack(scores, axis=1), "loss": np.stack(losses, axis=1)}


def main(experiment: str, n_neurons: int, seeds: list[int], n_steps: int):
    mode, cfg, n_genes = setup(experiment, algorithm="gd", n_neurons=n_neurons, n_gd_steps=n_steps)
    genes, history = train(mode, n_genes, cfg, seeds)
    for i, seed in enumerate(seeds):
        run_history = {k: v[i] for k, v in history.items()}
        run_cfg = dataclasses.replace(cfg, seed=seed)
        finish_run(experiment, f"gd_n{n_neurons}_s{seed}_steps{n_steps}", run_cfg, genes[i : i + 1], run_history)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--neurons", type=int, default=Config.n_neurons)
    parser.add_argument("--seeds", type=int, nargs="+", default=[Config.seed])
    parser.add_argument("--steps", type=int, default=Config.n_gd_steps)
    args = parser.parse_args()
    main(args.experiment, args.neurons, args.seeds, args.steps)
