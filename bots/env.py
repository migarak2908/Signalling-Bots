from itertools import permutations

import jax
import jax.numpy as jnp

from bots.config import Config


def place_landmarks(key: jax.Array, order: jax.Array, cfg: Config) -> jax.Array:
    """order[i] is the colour of the i-th landmark from the left. Returns x[c], the position of colour c."""
    n = cfg.n_colours
    lo, hi = cfg.target_zone
    slack = hi - lo - (n - 1) * cfg.min_gap
    x = lo + jnp.sort(jax.random.uniform(key, (n,), maxval=slack)) + cfg.min_gap * jnp.arange(n)
    return jnp.zeros(n).at[order].set(x)


def training_layouts(key: jax.Array, cfg: Config) -> jax.Array:
    """layouts_per_order layouts for every colour order. Shape (n_orders * layouts_per_order, n_colours)."""
    orders = jnp.array(list(permutations(range(cfg.n_colours))))
    orders = jnp.repeat(orders, cfg.layouts_per_order, axis=0)
    keys = jax.random.split(key, len(orders))
    return jax.vmap(lambda k, o: place_landmarks(k, o, cfg))(keys, orders)


def random_layouts(key: jax.Array, n_layouts: int, cfg: Config) -> jax.Array:
    def one(k):
        k_pos, k_order = jax.random.split(k)
        return place_landmarks(k_pos, jax.random.permutation(k_order, cfg.n_colours), cfg)

    return jax.vmap(one)(jax.random.split(key, n_layouts))


def falloff(distance: jax.Array, sensor_range: float) -> jax.Array:
    return jnp.maximum(0.0, 1.0 - distance / sensor_range)


def sense(sig_x, rec_x, landmark_x, target, cfg: Config) -> tuple[jax.Array, jax.Array]:
    """Inputs for each network: the colour sensors, then proximity."""
    prox = falloff(jnp.abs(sig_x - rec_x), 1.0)
    landmarks = falloff(jnp.abs(rec_x - landmark_x), cfg.landmark_range)
    return jnp.append(target, prox), jnp.append(landmarks, prox)
