from collections import Counter

import jax
import jax.numpy as jnp
import pytest

from bots.config import Config
from bots.ctrnn import step, unpack
from bots.env import sense, training_layouts
from bots.fitness import score

CFG = Config(n_colours=3)


def test_genome_length():
    assert Config(n_colours=3).genome_size == 56
    assert Config(n_colours=2).genome_size == 36
    assert Config(n_colours=2, n_extra=1).genome_size == 54
    assert Config(n_colours=3, n_extra=1).genome_size == 78


def no_recurrence(cfg):
    """One agent's params with every neuron-to-neuron weight and bias 0, input weights 10."""
    n = cfg.n_neurons
    genes = jnp.concatenate([jnp.zeros(n * n + 2 * n), jnp.ones(cfg.n_inputs)])
    return unpack(genes, cfg)


@pytest.mark.parametrize("n_colours", [2, 3])
@pytest.mark.parametrize("n_extra", [0, 1, 2])
def test_wiring(n_colours, n_extra):
    cfg = Config(n_colours=n_colours, n_extra=n_extra)
    params = no_recurrence(cfg)
    landmarks = 2.0 + 0.5 * jnp.arange(n_colours)
    far, prox_neuron = 5.0, n_colours

    def changed(inputs):
        y = step(params, jnp.zeros(cfg.n_neurons), inputs, cfg)
        return set(jnp.flatnonzero(y).tolist())

    for colour in range(n_colours):
        target = jax.nn.one_hot(colour, n_colours)
        sig_in, _ = sense(0.0, far, landmarks, target, cfg)
        assert changed(sig_in) == {colour}
        _, rec_in = sense(0.0, landmarks[colour], landmarks, jnp.zeros(n_colours), cfg)
        assert changed(rec_in) == {colour}

    sig_in, rec_in = sense(0.5, 0.5, landmarks, jnp.zeros(n_colours), cfg)
    assert changed(sig_in) == {prox_neuron}
    assert changed(rec_in) == {prox_neuron}


def parked(x):
    return jnp.stack([jnp.zeros(CFG.n_steps), jnp.full(CFG.n_steps, x)], axis=1)


def test_score_on_target():
    landmarks = jnp.array([2.0, 2.5, 3.0])
    assert score(parked(2.5), landmarks, 1, CFG) == pytest.approx(1.0)


def test_score_nearer_wrong_landmark():
    landmarks = jnp.array([2.0, 2.5, 3.0])
    assert score(parked(2.3), landmarks, 0, CFG) == 0.0


def test_every_order_three_times():
    layouts = training_layouts(jax.random.key(0), CFG)
    orders = Counter(tuple(jnp.argsort(x).tolist()) for x in layouts)
    assert len(orders) == 6
    assert set(orders.values()) == {CFG.layouts_per_order}

    lo, hi = CFG.target_zone
    sorted_x = jnp.sort(layouts, axis=1)
    assert (sorted_x >= lo).all() and (sorted_x <= hi).all()
    assert (jnp.diff(sorted_x, axis=1) >= CFG.min_gap - 1e-6).all()
