"""The world: landmark layouts, sensors and movement. All functions handle a single trial."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from config import Config


class Layout(NamedTuple):
    landmark_x: jax.Array  # shape (n_colours,); landmark_x[c] is the position of colour c
    signaller_x0: jax.Array
    receiver_x0: jax.Array


def sample_layout(key: jax.Array, cfg: Config) -> Layout:
    key_pos, key_perm = jax.random.split(key)
    n, gap = cfg.n_colours, cfg.min_landmark_spacing
    lo, hi = cfg.target_zone

    # Sorted points in the leftover space, then push each one a gap further right.
    slack = (hi - lo) - (n - 1) * gap
    offsets = jnp.sort(jax.random.uniform(key_pos, (n,), maxval=slack))
    positions = lo + offsets + gap * jnp.arange(n)
    landmark_x = jax.random.permutation(key_perm, positions)

    return Layout(landmark_x, jnp.asarray(cfg.signaller_start), jnp.asarray(cfg.receiver_start))


def sample_layouts(key: jax.Array, n: int, cfg: Config) -> Layout:
    """A batch of n layouts; each field gets a leading axis of size n."""
    return jax.vmap(lambda k: sample_layout(k, cfg))(jax.random.split(key, n))


def _falloff(distance: jax.Array, sensor_range: float) -> jax.Array:
    return jnp.maximum(0.0, 1.0 - distance / sensor_range)


def proximity(signaller_x: jax.Array, receiver_x: jax.Array, cfg: Config) -> jax.Array:
    return _falloff(jnp.abs(signaller_x - receiver_x), cfg.proximity_range)


def landmark_sensors(receiver_x: jax.Array, landmark_x: jax.Array, cfg: Config) -> jax.Array:
    return _falloff(jnp.abs(receiver_x - landmark_x), cfg.landmark_range)


def target_onehot(colour: jax.Array, cfg: Config) -> jax.Array:
    return jax.nn.one_hot(colour, cfg.n_colours)


def move_signaller(x: jax.Array, velocity: jax.Array, cfg: Config) -> jax.Array:
    return jnp.clip(x + velocity * cfg.dt, *cfg.signal_zone)


def move_receiver(x: jax.Array, velocity: jax.Array, cfg: Config) -> jax.Array:
    return x + velocity * cfg.dt
