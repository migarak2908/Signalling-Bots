"""NEAT-style evolution: networks start small and grow, and speciation protects new ideas.

Same genome format as the GA. A connection weight gene of exactly 0 means "no connection".
Simplified from NEAT: fixed neuron slots (no node-splitting or innovation numbers), a fixed number of
species (each genome joins its nearest representative), no crossover.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from config import Config
from ctrnn import connection_genes, input_to_motor_genes


class State(NamedTuple):
    population: jax.Array  # (pop_size, n_genes)
    representatives: jax.Array  # (n_species, n_genes): one genome per species, used to assign species


def init(key: jax.Array, n_genes: int, cfg: Config) -> State:
    """Random biases and taus; the only connections are input -> motor."""
    gene_key, rep_key = jax.random.split(key)
    genes = jax.random.uniform(gene_key, (cfg.pop_size, n_genes), minval=-1.0, maxval=1.0)
    keep = ~connection_genes(cfg) | input_to_motor_genes(cfg)
    population = jnp.where(keep, genes, 0.0)
    representatives = population[jax.random.choice(rep_key, cfg.pop_size, (cfg.n_species,), replace=False)]
    return State(population, representatives)


def population(state: State) -> jax.Array:
    return state.population


def distance(a: jax.Array, b: jax.Array, conn: jax.Array) -> jax.Array:
    """Fraction of connection slots where one genome has a connection and the other doesn't, plus mean gene difference."""
    structure = jnp.sum(((a != 0) != (b != 0)) & conn) / jnp.sum(conn)
    return structure + jnp.mean(jnp.abs(a - b))


def _mutate(key: jax.Array, genes: jax.Array, conn: jax.Array, cfg: Config) -> jax.Array:
    """Perturb existing genes; maybe switch one connection on; maybe switch one off."""
    noise_key, add_key, add_pick, add_value, remove_key, remove_pick = jax.random.split(key, 6)
    active = ~conn | (genes != 0)
    genes = jnp.where(active, genes + cfg.mutation_std * jax.random.normal(noise_key, genes.shape), genes)

    off = conn & (genes == 0)
    add = jax.random.bernoulli(add_key, cfg.neat_add_connection) & off.any()
    slot = jax.random.categorical(add_pick, jnp.where(off, 0.0, -jnp.inf))
    genes = jnp.where(add, genes.at[slot].set(jax.random.uniform(add_value, minval=-1.0, maxval=1.0)), genes)

    on = conn & (genes != 0)
    remove = jax.random.bernoulli(remove_key, cfg.neat_remove_connection) & on.any()
    slot = jax.random.categorical(remove_pick, jnp.where(on, 0.0, -jnp.inf))
    genes = jnp.where(remove, genes.at[slot].set(0.0), genes)
    return jnp.clip(genes, -1.0, 1.0)


def next_generation(key: jax.Array, state: State, scores: jax.Array, cfg: Config) -> State:
    pop, n_species = cfg.pop_size, cfg.n_species
    conn = connection_genes(cfg)
    parent_key, mutate_key = jax.random.split(key)

    # 1. Speciate: each genome joins the species of its nearest representative.
    dist = jax.vmap(lambda g: jax.vmap(lambda r: distance(g, r, conn))(state.representatives))(state.population)
    species = jnp.argmin(dist, axis=1)
    member = species[:, None] == jnp.arange(n_species)  # (pop, n_species)
    size = member.sum(axis=0)

    # 2. Offspring per species, in proportion to the species' mean score (empty species get none).
    mean_score = (member * scores[:, None]).sum(axis=0) / jnp.maximum(size, 1)
    share = jnp.where(size > 0, mean_score + 1e-6, 0.0)
    share = share / share.sum()
    quota = jnp.floor(share * pop).astype(int)
    quota = quota.at[jnp.argmax(share)].add(pop - quota.sum())

    # 3. Parents: each child slot belongs to a species and picks a parent from that species' top members.
    slot_species = jnp.repeat(jnp.arange(n_species), quota, total_repeat_length=pop)
    same = species[:, None] == species[None, :]
    rank = jnp.sum(same & (scores[None, :] > scores[:, None]), axis=1)  # 0 = best in its species
    can_parent = rank < jnp.maximum(1, jnp.ceil(cfg.neat_survival * size[species]))
    logits = jnp.where((species[None, :] == slot_species[:, None]) & can_parent[None, :], 0.0, -jnp.inf)
    parents = jax.random.categorical(parent_key, logits)  # (pop,)

    # 4. Children are mutated copies, except each species' first slot, which keeps its champion unchanged.
    champion = jnp.argmax(jnp.where(member, scores[:, None], -jnp.inf), axis=0)  # (n_species,)
    first_slot = jnp.arange(pop) == (jnp.cumsum(quota) - quota)[slot_species]
    children = jax.vmap(lambda k, g: _mutate(k, g, conn, cfg))(jax.random.split(mutate_key, pop), state.population[parents])
    new_population = jnp.where(first_slot[:, None], state.population[champion[slot_species]], children)

    # 5. New representatives: each species' champion (empty species keep their old one).
    representatives = jnp.where((size > 0)[:, None], state.population[champion], state.representatives)
    return State(new_population, representatives)
