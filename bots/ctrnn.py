from typing import NamedTuple

import jax
import jax.numpy as jnp

from bots.config import Config


class Params(NamedTuple):
    w: jax.Array  # (n, n), w[j, i] is the weight from neuron j to neuron i
    theta: jax.Array
    tau: jax.Array
    g: jax.Array  # g[i] is the weight of input i, which feeds neuron i
    rule: jax.Array  # [eta, A, B, C, D] if plastic, else empty


def scale(genes: jax.Array, bounds: tuple[float, float]) -> jax.Array:
    lo, hi = bounds
    return lo + (genes + 1) / 2 * (hi - lo)


def unpack(genes: jax.Array, cfg: Config) -> Params:
    n = cfg.n_neurons
    w, theta, tau, g, rule = jnp.split(genes, [n * n, n * n + n, n * n + 2 * n, n * n + 2 * n + cfg.n_inputs])
    return Params(
        w=scale(w.reshape(n, n), cfg.param_range),
        theta=scale(theta, cfg.param_range),
        tau=scale(tau, cfg.tau_range),
        g=scale(g, cfg.param_range),
        rule=jnp.concatenate([scale(rule[:1], (0.0, 1.0)), rule[1:]]),
    )


def split_genome(genome: jax.Array, cfg: Config) -> tuple[jax.Array, jax.Array]:
    n = cfg.params_per_agent
    return genome[:n], genome[n:]


def step(p: Params, w: jax.Array, y: jax.Array, inputs: jax.Array, cfg: Config) -> jax.Array:
    external = jnp.pad(p.g * inputs, (0, cfg.n_extra))
    dy = -y + jax.nn.sigmoid(y + p.theta) @ w + external
    return y + cfg.dt / p.tau * dy


def hebbian(rule: jax.Array, w: jax.Array, o: jax.Array, cfg: Config) -> jax.Array:
    """One weight update from outputs o. o[:, None] is the presynaptic side (row j), o[None, :] the postsynaptic."""
    eta, a, b, c, d = rule
    dw = eta * (a * jnp.outer(o, o) + b * o[:, None] + c * o[None, :] + d) * cfg.dt
    return jnp.clip(w + dw, *cfg.param_range)


def learn(p: Params, w: jax.Array, y: jax.Array, cfg: Config) -> jax.Array:
    return hebbian(p.rule, w, jax.nn.sigmoid(y + p.theta), cfg)


def velocity(p: Params, y: jax.Array, cfg: Config) -> jax.Array:
    m = cfg.motor_neuron
    return 2 * jax.nn.sigmoid(y[m] + p.theta[m]) - 1
