"""Genetic algorithm: truncation selection with elitism, Gaussian mutation, no crossover. The state is the population."""

import jax
import jax.numpy as jnp

from config import Config


def init(key: jax.Array, n_genes: int, cfg: Config) -> jax.Array:
    return jax.random.uniform(key, (cfg.pop_size, n_genes), minval=-1.0, maxval=1.0)


def population(state: jax.Array) -> jax.Array:
    return state


def next_generation(key: jax.Array, state: jax.Array, scores: jax.Array, cfg: Config) -> jax.Array:
    """Elites first (best-first, unchanged), then mutated copies of them, each elite equally often."""
    n_elite = round(cfg.pop_size * cfg.elite_fraction)
    elites = state[jnp.argsort(-scores)[:n_elite]]
    parents = elites[jnp.arange(cfg.pop_size - n_elite) % n_elite]
    children = parents + cfg.mutation_std * jax.random.normal(key, parents.shape)
    return jnp.concatenate([elites, jnp.clip(children, -1.0, 1.0)])
