import jax
import jax.numpy as jnp

from bots.config import Config
from bots.env import random_layouts
from bots.rollout import all_trials


def score(positions: jax.Array, landmark_x: jax.Array, colour: jax.Array, cfg: Config) -> jax.Array:
    rec_x = positions[-cfg.score_steps:, 1]
    distances = jnp.abs(rec_x[:, None] - landmark_x)
    to_target = distances[:, colour]
    wrong_nearer = (distances < to_target[:, None]).any(axis=1)
    return jnp.where(wrong_nearer, 0.0, jnp.maximum(0.0, 1.0 - to_target / cfg.score_range)).mean()


def success(positions: jax.Array, landmark_x: jax.Array, colour: jax.Array, cfg: Config) -> jax.Array:
    return jnp.abs(positions[-1, 1] - landmark_x[colour]) <= cfg.success_radius


def evaluate(genome: jax.Array, layouts: jax.Array, cfg: Config, hide_colour: bool = False):
    """Scores and successes, each shape (n_layouts, n_colours)."""
    positions = all_trials(genome, layouts, cfg, hide_colour)
    colours = jnp.arange(cfg.n_colours)

    def grid(fn):
        per_colour = jax.vmap(lambda p, c, x: fn(p, x, c, cfg), (0, 0, None))
        return jax.vmap(per_colour, (0, None, 0))(positions, colours, layouts)

    return grid(score), grid(success)


def fitness(genome: jax.Array, layouts: jax.Array, cfg: Config) -> jax.Array:
    return evaluate(genome, layouts, cfg)[0].mean()


def heldout_success(genome: jax.Array, cfg: Config, hide_colour: bool = False) -> list[float]:
    """Success per colour on the fixed held-out layouts, each run with every target colour."""
    layouts = random_layouts(jax.random.key(cfg.test_seed), cfg.test_layouts, cfg)
    return evaluate(genome, layouts, cfg, hide_colour)[1].mean(axis=0).tolist()


def heldout_results(genome: jax.Array, cfg: Config) -> dict:
    success = heldout_success(genome, cfg)
    return {
        "success": success,
        "success_no_signal": heldout_success(genome, cfg, hide_colour=True),
        "solved": min(success) >= cfg.solved_success,
    }
