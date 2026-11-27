"""python -m bots.evolve --colours 2 --seeds 0 1 2 3 4 5 6 7 8 9"""

import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from bots.config import Config
from bots.ctrnn import scale, split_genome, unpack
from bots.env import training_layouts
from bots.fitness import fitness, heldout_results
from bots.plots import plot_fitness, plot_trajectories


def initial_population(key: jax.Array, cfg: Config) -> jax.Array:
    n = cfg.n_neurons
    w, theta, tau = slice(0, n * n), slice(n * n, n * n + n), slice(n * n + n, n * n + 2 * n)
    genes = jax.random.uniform(key, (cfg.pop_size, 2, cfg.params_per_agent), minval=-1.0, maxval=1.0)
    rule = slice(n * n + 2 * n + cfg.n_input_weights, None)
    genes = genes * jnp.full(cfg.params_per_agent, cfg.init_range).at[tau].set(1.0).at[rule].set(1.0)
    if cfg.centre:
        weights = scale(genes[..., w], cfg.param_range).reshape(*genes.shape[:2], n, n)
        lo, hi = cfg.param_range
        centred = -weights.sum(axis=-2) / 2
        genes = genes.at[..., theta].set(jnp.clip((centred - lo) / (hi - lo) * 2 - 1, -1.0, 1.0))
    return genes.reshape(cfg.pop_size, cfg.genome_size)


def next_generation(key: jax.Array, pop: jax.Array, scores: jax.Array, cfg: Config) -> jax.Array:
    """Elites kept unchanged (best first), the rest are mutated copies of them."""
    n_elite = round(cfg.pop_size * cfg.elite_fraction)
    elites = pop[jnp.argsort(-scores)[:n_elite]]
    parents = elites[jnp.arange(cfg.pop_size - n_elite) % n_elite]
    children = parents + cfg.mutation_std * jax.random.normal(key, parents.shape)
    return jnp.concatenate([elites, jnp.clip(children, -1.0, 1.0)])


def evolve(cfg: Config, seeds: list[int], log_every: int = 50):
    """One population per seed, all run together. Returns best genomes (n_seeds, genome_size) and history."""

    def start(seed):
        key, init_key = jax.random.split(jax.random.key(seed))
        return key, initial_population(init_key, cfg)

    def generation(key, pop):
        key, layout_key, mutate_key = jax.random.split(key, 3)
        layouts = training_layouts(layout_key, cfg)
        scores = jax.vmap(lambda g: fitness(g, layouts, cfg))(pop)
        return key, next_generation(mutate_key, pop, scores, cfg), scores

    keys, pops = jax.vmap(start)(jnp.array(seeds))
    print("compiling...", flush=True)
    step = jax.jit(jax.vmap(generation)).lower(keys, pops).compile()

    best, mean = [], []
    clock = time.time()
    for gen in range(cfg.generations):
        keys, pops, scores = step(keys, pops)
        best.append(scores.max(axis=1))
        mean.append(scores.mean(axis=1))
        if gen % log_every == 0 or gen == cfg.generations - 1:
            elapsed = time.time() - clock
            left = elapsed / (gen + 1) * (cfg.generations - gen - 1)
            line = " ".join(f"{b:.2f}" for b in best[-1])
            print(f"{gen:5d}/{cfg.generations}  best: {line}  ({elapsed / 60:.1f} min, ~{left / 60:.1f} min left)", flush=True)

    history = {"best": np.stack(best, axis=1), "mean": np.stack(mean, axis=1)}
    return pops[:, 0], history


def save_run(out: Path, cfg: Config, genome: jax.Array, history: dict):
    out.mkdir(parents=True, exist_ok=True)
    results = heldout_results(genome, cfg)
    if cfg.plastic:
        agents = zip(["signaller", "receiver"], split_genome(genome, cfg))
        results["rule"] = {name: dict(zip(["eta", "A", "B", "C", "D"], unpack(genes, cfg).rule.tolist())) for name, genes in agents}
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    (out / "results.json").write_text(json.dumps(results, indent=2))
    np.save(out / "best_genome.npy", np.asarray(genome))
    np.savez(out / "history.npz", **history)
    success = " / ".join(f"{s:.2f}" for s in results["success"])
    print(f"{out}: success {success}, solved {results['solved']}", flush=True)
    plot_fitness(out, history)
    plot_trajectories(out, genome, cfg)


def run_folder(cfg: Config) -> str:
    options = {
        "pair": cfg.motor == "pair",
        "fullinput": cfg.full_input,
        "plastic": cfg.plastic,
        f"init{cfg.init_range:g}": cfg.init_range != 1.0,
        "centre": cfg.centre,
    }
    return f"{cfg.n_colours}col_n{cfg.n_neurons}" + "".join(f"_{name}" for name, on in options.items() if on)


def main(cfg: Config, seeds: list[int]):
    genomes, history = evolve(cfg, seeds)
    for i, seed in enumerate(seeds):
        out = Path("results") / run_folder(cfg) / f"s{seed}_g{cfg.generations}"
        save_run(out, cfg, genomes[i], {k: v[i] for k, v in history.items()})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--colours", type=int, choices=[2, 3], default=Config.n_colours)
    parser.add_argument("--extra", type=int, default=Config.n_extra)
    parser.add_argument("--full-input", action="store_true")
    parser.add_argument("--motor", choices=["single", "pair"], default=Config.motor)
    parser.add_argument("--plastic", action="store_true")
    parser.add_argument("--init", type=float, default=Config.init_range)
    parser.add_argument("--centre", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--generations", type=int, default=Config.generations)
    args = parser.parse_args()
    cfg = Config(
        n_colours=args.colours,
        n_extra=args.extra,
        full_input=args.full_input,
        motor=args.motor,
        plastic=args.plastic,
        init_range=args.init,
        centre=args.centre,
        generations=args.generations,
    )
    main(cfg, args.seeds)
