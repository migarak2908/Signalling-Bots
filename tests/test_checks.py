from collections import Counter

import jax
import jax.numpy as jnp
import pytest

from bots.config import Config
from bots.ctrnn import hebbian, step, unpack, velocity
from bots.env import sense, training_layouts
from bots.fitness import combine, score
from bots.rollout import trial

CFG = Config(n_colours=3)


def test_genome_length():
    assert Config(n_colours=3).genome_size == 56
    assert Config(n_colours=2).genome_size == 36
    assert Config(n_colours=2, n_extra=1).genome_size == 54
    assert Config(n_colours=3, n_extra=1).genome_size == 78


def no_recurrence(cfg):
    """One agent's params with every neuron-to-neuron weight and bias 0, input weights 10."""
    n = cfg.n_neurons
    genes = jnp.concatenate([jnp.zeros(n * n + 2 * n), jnp.ones(cfg.n_input_weights)])
    return unpack(genes, cfg)


@pytest.mark.parametrize("n_colours", [2, 3])
@pytest.mark.parametrize("n_extra", [0, 1, 2])
@pytest.mark.parametrize("motor", ["single", "pair"])
def test_wiring(n_colours, n_extra, motor):
    cfg = Config(n_colours=n_colours, n_extra=n_extra, motor=motor)
    params = no_recurrence(cfg)
    landmarks = 2.0 + 0.5 * jnp.arange(n_colours)
    far, prox_neuron = 5.0, n_colours

    def changed(inputs):
        y = step(params, params.w, jnp.zeros(cfg.n_neurons), inputs, cfg)
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


def test_plastic_genome_length():
    assert Config(n_colours=2, plastic=True).genome_size == 46
    assert Config(n_colours=3, plastic=True).genome_size == 66
    assert Config(n_colours=2, n_extra=1, plastic=True).genome_size == 64


def test_plastic_with_eta_zero_matches_static():
    static, plastic = Config(n_colours=3), Config(n_colours=3, plastic=True)
    agents = jax.random.uniform(jax.random.key(1), (2, static.params_per_agent), minval=-1, maxval=1)
    rule = jnp.array([-1.0, 0.7, -0.3, 0.5, 0.9])  # eta gene -1 -> eta 0
    plastic_genome = jnp.concatenate([agents[0], rule, agents[1], rule])
    landmarks = jnp.array([2.0, 2.5, 3.0])
    for colour in range(3):
        a = trial(agents.ravel(), landmarks, colour, static)
        b = trial(plastic_genome, landmarks, colour, plastic)
        assert (a == b).all()


ACTIVE = jnp.array([1.0, 0.0, 1.0, 0.0])  # only neurons 0 and 2 active


def changed_weights(rule):
    w = hebbian(jnp.array(rule), jnp.zeros((4, 4)), ACTIVE, CFG)
    return {tuple(ji) for ji in jnp.argwhere(w != 0).tolist()}


def test_hebbian_term_orientation():
    assert changed_weights([1.0, 1.0, 0.0, 0.0, 0.0]) == {(0, 0), (0, 2), (2, 0), (2, 2)}


def test_presynaptic_term_changes_rows():
    assert changed_weights([1.0, 0.0, 1.0, 0.0, 0.0]) == {(j, i) for j in (0, 2) for i in range(4)}


def test_weights_stay_in_range():
    for d in (1.0, -1.0):
        rule = jnp.array([1.0, 1.0, d, d, d])
        w = jax.lax.fori_loop(0, 1000, lambda _, w: hebbian(rule, w, ACTIVE, CFG), jnp.full((4, 4), 9.0 * d))
        assert (jnp.abs(w) <= 10.0).all()


def test_full_input_genome_length():
    assert Config(n_colours=2, n_extra=1, full_input=True).genome_size == 72
    assert Config(n_colours=3, full_input=True).genome_size == 80


@pytest.mark.parametrize("n_extra", [0, 1])
@pytest.mark.parametrize("motor", ["single", "pair"])
def test_full_input_reaches_every_neuron(n_extra, motor):
    cfg = Config(n_colours=2, n_extra=n_extra, motor=motor, full_input=True)
    params = no_recurrence(cfg)
    for k in range(cfg.n_inputs):
        y = step(params, params.w, jnp.zeros(cfg.n_neurons), jax.nn.one_hot(k, cfg.n_inputs), cfg)
        assert (y != 0).all()


def test_worst_colour_fitness():
    one_colour = jnp.array([[1.0, 0.0]] * 6)  # always right for red, never for green
    assert combine(one_colour, Config(n_colours=2)) == pytest.approx(0.5)
    assert combine(one_colour, Config(n_colours=2, worst_weight=0.5)) == pytest.approx(0.25)
    uneven = jnp.array([[1.0, 0.6], [0.8, 0.2]])  # colour means 0.9 and 0.4
    assert combine(uneven, Config(n_colours=2, worst_weight=0.5)) == pytest.approx(0.5 * 0.65 + 0.5 * 0.4)


def test_pair_genome_length():
    assert Config(n_colours=2, motor="pair").genome_size == 2 * (25 + 10 + 3)
    assert Config(n_colours=3, motor="pair").genome_size == 2 * (36 + 12 + 4)
    assert Config(n_colours=2, n_extra=1, motor="pair", full_input=True).genome_size == 2 * (36 + 12 + 18)


def test_pair_velocity():
    cfg = Config(n_colours=2, motor="pair")
    params = no_recurrence(cfg)
    y = jnp.zeros(cfg.n_neurons)
    assert velocity(params, y, cfg) == 0.0
    assert velocity(params, y.at[-2:].set(jnp.array([20.0, -20.0])), cfg) > 0.99
    assert velocity(params, y.at[-2:].set(jnp.array([-20.0, 20.0])), cfg) < -0.99
