"""Run trials with lax.scan. Every rollout returns positions of shape (n_steps, 2): [signaller, receiver].

Each step: sense -> think -> decide -> move -> record.
"""

from typing import Callable

import jax
import jax.numpy as jnp

from config import Config
from ctrnn import split_genome, step, unpack, velocity
from env import (
    Layout,
    landmark_sensors,
    move_receiver,
    move_signaller,
    proximity,
    target_onehot,
)
from scripted import cheating_receiver, scripted_signal


def _inputs(prox: jax.Array, colour_inputs: jax.Array) -> jax.Array:
    return jnp.concatenate([jnp.atleast_1d(prox), colour_inputs])


def rollout(
    genome: jax.Array, layout: Layout, colour: jax.Array, cfg: Config, hide_colour: bool = False
) -> jax.Array:
    """The real task: signaller CTRNN and receiver CTRNN.

    hide_colour=True zeroes the signaller's target-colour inputs (no-communication baseline, signal ablation).
    """
    sig_params, rec_params = (unpack(genes, cfg) for genes in split_genome(genome, cfg))
    target = jnp.where(hide_colour, 0.0, target_onehot(colour, cfg))

    def body(carry, _):
        sig_x, rec_x, sig_state, rec_state = carry

        prox = proximity(sig_x, rec_x, cfg)

        sig_state = step(sig_params, sig_state, _inputs(prox, target), cfg)
        rec_state = step(rec_params, rec_state, _inputs(prox, landmark_sensors(rec_x, layout.landmark_x, cfg)), cfg)

        sig_x = move_signaller(sig_x, velocity(sig_params, sig_state, cfg), cfg)
        rec_x = move_receiver(rec_x, velocity(rec_params, rec_state, cfg), cfg)

        return (sig_x, rec_x, sig_state, rec_state), jnp.stack([sig_x, rec_x])

    state0 = jnp.zeros(cfg.n_neurons)
    init = (layout.signaller_x0, layout.receiver_x0, state0, state0)
    _, positions = jax.lax.scan(body, init, length=cfg.n_steps)
    return positions


def rollout_cheating(layout: Layout, colour: jax.Array, cfg: Config) -> jax.Array:
    """Cheating receiver; the signaller stays at its start."""

    def body(carry, _):
        rec_x, stopped = carry
        v, stopped = cheating_receiver(stopped, rec_x, layout.landmark_x, colour, cfg)
        rec_x = move_receiver(rec_x, v, cfg)
        return (rec_x, stopped), jnp.stack([layout.signaller_x0, rec_x])

    init = (layout.receiver_x0, jnp.array(False))
    _, positions = jax.lax.scan(body, init, length=cfg.n_steps)
    return positions


def rollout_scripted(receiver_genes: jax.Array, layout: Layout, colour: jax.Array, cfg: Config) -> jax.Array:
    """Receiver CTRNN fed the scripted signal. No signaller: its column stays at the start position."""
    params = unpack(receiver_genes, cfg)

    def body(carry, k):
        rec_x, rec_state = carry

        prox = scripted_signal(k, colour, cfg)

        rec_state = step(params, rec_state, _inputs(prox, landmark_sensors(rec_x, layout.landmark_x, cfg)), cfg)
        rec_x = move_receiver(rec_x, velocity(params, rec_state, cfg), cfg)

        return (rec_x, rec_state), jnp.stack([layout.signaller_x0, rec_x])

    init = (layout.receiver_x0, jnp.zeros(cfg.n_neurons))
    _, positions = jax.lax.scan(body, init, jnp.arange(cfg.n_steps))
    return positions


def run_grid(trial: Callable[[Layout, jax.Array], jax.Array], layouts: Layout, n_colours: int) -> jax.Array:
    """Run trial(layout, colour) for every layout with every colour.

    `layouts` is a batch of layouts. Returns shape (n_layouts, n_colours, n_steps, 2).
    """
    colours = jnp.arange(n_colours)
    over_colours = jax.vmap(trial, in_axes=(None, 0))
    return jax.vmap(over_colours, in_axes=(0, None))(layouts, colours)
