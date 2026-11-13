"""Scripted test components: the cheating receiver and the scripted signal. Not results."""

import jax
import jax.numpy as jnp

from config import Config
from env import landmark_sensors


def cheating_receiver(
    stopped: jax.Array, receiver_x: jax.Array, landmark_x: jax.Array, colour: jax.Array, cfg: Config
) -> tuple[jax.Array, jax.Array]:
    """Tests the world code: a hand-written receiver that is told the target colour.

    Moves right at full speed and stops for good once the target's sensor passes the
    threshold (within 0.1). It should score 0.8 or more; if not, the world code has a bug.
    Returns (velocity, stopped).
    """
    target_reading = landmark_sensors(receiver_x, landmark_x, cfg)[colour]
    stopped = stopped | (target_reading > cfg.cheat_stop_threshold)
    return jnp.where(stopped, 0.0, cfg.v_max), stopped


def scripted_signal(k: jax.Array, colour: jax.Array, cfg: Config) -> jax.Array:
    """Tests whether a receiver can learn from a clean signal.

    Replaces the signaller: gives the receiver's proximity input at time step k as
    colour + 1 pulses (red 1, green 2, blue 3), then 0.
    """
    since_start = k - cfg.signal_start_step
    pulse = since_start // cfg.pulse_period_steps
    in_pulse = since_start % cfg.pulse_period_steps < cfg.pulse_on_steps
    return jnp.where((since_start >= 0) & (pulse <= colour) & in_pulse, 1.0, 0.0)
