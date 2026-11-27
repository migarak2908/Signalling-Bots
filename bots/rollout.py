import jax
import jax.numpy as jnp

from bots.config import Config
from bots.ctrnn import learn, split_genome, step, unpack, velocity
from bots.env import sense


def trial(
    genome: jax.Array, landmark_x: jax.Array, colour: jax.Array, cfg: Config, hide_colour: bool = False, last: int | None = None
) -> jax.Array:
    """Signaller and receiver positions after every step, shape (n_steps, 2), or only the last `last` steps."""
    sig, rec = (unpack(genes, cfg) for genes in split_genome(genome, cfg))
    target = jax.nn.one_hot(colour, cfg.n_colours) * (not hide_colour)

    def body(carry, _):
        sig_x, rec_x, sig_y, rec_y, sig_w, rec_w = carry
        sig_in, rec_in = sense(sig_x, rec_x, landmark_x, target, cfg)
        sig_y = step(sig, sig_w, sig_y, sig_in, cfg)
        rec_y = step(rec, rec_w, rec_y, rec_in, cfg)
        if cfg.plastic:
            sig_w = learn(sig, sig_w, sig_y, cfg)
            rec_w = learn(rec, rec_w, rec_y, cfg)
        sig_x = jnp.clip(sig_x + velocity(sig, sig_y, cfg) * cfg.dt, *cfg.signal_zone)
        rec_x = rec_x + velocity(rec, rec_y, cfg) * cfg.dt
        return (sig_x, rec_x, sig_y, rec_y, sig_w, rec_w), jnp.stack([sig_x, rec_x])

    y0 = jnp.zeros(cfg.n_neurons)
    start = (jnp.asarray(cfg.signaller_start), jnp.asarray(cfg.receiver_start), y0, y0, sig.w, rec.w)
    def unrecorded(carry, x):
        return body(carry, x)[0], None

    keep = cfg.n_steps if last is None else last
    carry, _ = jax.lax.scan(unrecorded, start, length=cfg.n_steps - keep, unroll=cfg.unroll)
    _, positions = jax.lax.scan(body, carry, length=keep, unroll=cfg.unroll)
    return positions


def all_trials(
    genome: jax.Array, layouts: jax.Array, cfg: Config, hide_colour: bool = False, last: int | None = None
) -> jax.Array:
    """Every layout with every target colour. Shape (n_layouts, n_colours, steps, 2)."""
    def run(landmark_x, colour):
        return trial(genome, landmark_x, colour, cfg, hide_colour, last)

    colours = jnp.arange(cfg.n_colours)
    return jax.vmap(jax.vmap(run, (None, 0)), (0, None))(layouts, colours)
