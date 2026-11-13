from typing import NamedTuple

import jax
import jax.numpy as jnp


from config import Config


class Params(NamedTuple):
    w: jax.Array  # shape (n_neurons, n_neurons); w[j, i] is the weight from neuron j to neuron i
    theta: jax.Array  # shape (n_neurons,)
    tau: jax.Array  # shape (n_neurons,)
    g: jax.Array  # shape (n_inputs, n_neurons); g[k, i] is the weight from input k to neuron i


def _linear(genes: jax.Array, bounds: tuple[float, float]) -> jax.Array:
    lo, hi = bounds
    return lo + (genes + 1) / 2 * (hi - lo)


def _log(genes: jax.Array, bounds: tuple[float, float]) -> jax.Array:
    # Log scale spreads tau evenly between fast and slow neurons. Needs lo > 0.
    lo, hi = bounds
    return lo * (hi / lo) ** ((genes + 1) / 2)


def unpack(genes: jax.Array, cfg: Config) -> Params:
    """Turn one agent's genes (each in [-1, 1]) into network parameters."""
    n, k = cfg.n_neurons, cfg.n_inputs
    w_end = n * n
    theta_end = w_end + n
    tau_end = theta_end + n

    return Params(
        w=_linear(genes[:w_end].reshape(n, n), cfg.weight_range),
        theta=_linear(genes[w_end:theta_end], cfg.bias_range),
        tau=_log(genes[theta_end:tau_end], cfg.tau_range),
        g=_linear(genes[tau_end:].reshape(k, n), cfg.weight_range),
    )


def split_genome(genome: jax.Array, cfg: Config) -> tuple[jax.Array, jax.Array]:

    n = cfg.n_params_per_agent
    signaller_genes = genome[..., :n]
    receiver_genes = genome[..., n:]

    return (signaller_genes, receiver_genes)

def step(params: Params, state: jax.Array, inputs: jax.Array, cfg: Config) -> jax.Array:
    """One Euler step of the CTRNN. Vectors go on the left: weights are stored (from, to)."""

    firing = jax.nn.sigmoid(state + params.theta)
    drive = -state + firing @ params.w + inputs @ params.g
    new_state = state + cfg.dt / params.tau * drive
    return new_state


def velocity(params: Params, state: jax.Array, cfg: Config) -> jax.Array:
    m = cfg.motor_index
    motor_output = 2 * jax.nn.sigmoid(state[m] + params.theta[m]) - 1
    vel = cfg.v_max * motor_output
    return vel




# --- Which genes are connections (used by evolvers that grow networks). A weight gene of 0 = no connection.


def _gene_mask(cfg: Config, w: jax.Array, g: jax.Array) -> jax.Array:
    """Full-genome mask from one agent's (n, n) recurrent mask and (k, n) input mask. Biases and taus: False."""
    agent = jnp.concatenate([w.ravel(), jnp.zeros(2 * cfg.n_neurons, bool), g.ravel()])
    return jnp.tile(agent, 2)


def connection_genes(cfg: Config) -> jax.Array:
    n, k = cfg.n_neurons, cfg.n_inputs
    return _gene_mask(cfg, jnp.ones((n, n), bool), jnp.ones((k, n), bool))


def input_to_motor_genes(cfg: Config) -> jax.Array:
    n, k = cfg.n_neurons, cfg.n_inputs
    return _gene_mask(cfg, jnp.zeros((n, n), bool), jnp.zeros((k, n), bool).at[:, cfg.motor_index].set(True))
