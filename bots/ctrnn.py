from typing import NamedTuple

import jax
import jax.numpy as jnp

from bots.config import Config


class Params(NamedTuple):
    w: jax.Array  # (n, n), w[j, i] is the weight from neuron j to neuron i
    theta: jax.Array
    tau: jax.Array
    g: jax.Array  # g[i] is the weight of input i, which feeds neuron i


def scale(genes: jax.Array, bounds: tuple[float, float]) -> jax.Array:
    lo, hi = bounds
    return lo + (genes + 1) / 2 * (hi - lo)


def unpack(genes: jax.Array, cfg: Config) -> Params:
    n = cfg.n_neurons
    w, theta, tau, g = jnp.split(genes, [n * n, n * n + n, n * n + 2 * n])
    return Params(
        w=scale(w.reshape(n, n), cfg.param_range),
        theta=scale(theta, cfg.param_range),
        tau=scale(tau, cfg.tau_range),
        g=scale(g, cfg.param_range),
    )


def split_genome(genome: jax.Array, cfg: Config) -> tuple[jax.Array, jax.Array]:
    n = cfg.params_per_agent
    return genome[:n], genome[n:]


def step(p: Params, y: jax.Array, inputs: jax.Array, cfg: Config) -> jax.Array:
    external = jnp.pad(p.g * inputs, (0, cfg.n_extra))
    dy = -y + jax.nn.sigmoid(y + p.theta) @ p.w + external
    return y + cfg.dt / p.tau * dy


def velocity(p: Params, y: jax.Array, cfg: Config) -> jax.Array:
    m = cfg.motor_neuron
    return 2 * jax.nn.sigmoid(y[m] + p.theta[m]) - 1
