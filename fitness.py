"""Scoring. All functions work on one trial or any batch of trials.

For run_grid output, the targets are layouts.landmark_x: entry [i, c] is where colour c is in layout i.
"""

import jax
import jax.numpy as jnp

from config import Config
from env import Layout
from rollout import rollout, rollout_cheating, run_grid


def trial_score(positions: jax.Array, target_x: jax.Array, cfg: Config) -> jax.Array:
    """Mean of max(0, 1 - distance / score_range) over the last score_steps."""
    receiver_x = positions[..., -cfg.score_steps:, 1]
    distance = jnp.abs(receiver_x - jnp.asarray(target_x)[..., None])
    return jnp.maximum(0.0, 1.0 - distance / cfg.score_range).mean(axis=-1)


def success(positions: jax.Array, target_x: jax.Array, cfg: Config) -> jax.Array:
    """True if the receiver ends within success_radius of the target. For reporting only."""
    return jnp.abs(positions[..., -1, 1] - target_x) <= cfg.success_radius


def fitness(genome: jax.Array, layouts: Layout, cfg: Config, hide_colour: bool = False) -> jax.Array:
    """Mean trial score over every layout with every colour."""
    trial = lambda layout, colour: rollout(genome, layout, colour, cfg, hide_colour)
    positions = run_grid(trial, layouts, cfg.n_colours)
    return trial_score(positions, layouts.landmark_x, cfg).mean()


# --- Baselines: receivers that don't know the target. Each returns (scores, successes), shape (n_layouts, n_colours).


def park_baseline(layouts: Layout, cfg: Config) -> tuple[jax.Array, jax.Array]:
    """Receiver drives right at full speed and parks in the middle of the target zone."""
    park_x = sum(cfg.target_zone) / 2
    steps = jnp.arange(1, cfg.n_steps + 1)
    receiver_x = jnp.minimum(cfg.receiver_start + cfg.v_max * cfg.dt * steps, park_x)
    positions = jnp.stack([jnp.full_like(receiver_x, cfg.signaller_start), receiver_x], axis=-1)
    return trial_score(positions, layouts.landmark_x, cfg), success(positions, layouts.landmark_x, cfg)


def random_landmark_baseline(layouts: Layout, cfg: Config) -> tuple[jax.Array, jax.Array]:
    """Receiver goes to a landmark picked at random. Exact average over every possible pick, so no key needed."""
    positions = run_grid(lambda layout, pick: rollout_cheating(layout, pick, cfg), layouts, cfg.n_colours)
    positions = positions[:, :, None]  # (layout, pick, 1, step, agent)
    targets = layouts.landmark_x[:, None, :]  # (layout, 1, target)
    scores = trial_score(positions, targets, cfg)  # (layout, pick, target)
    successes = success(positions, targets, cfg)
    return scores.mean(axis=1), successes.mean(axis=1)
