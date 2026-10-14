"""Unit tests for scoring: pool weights, family metrics and the example reward."""
import numpy as np

from fewshot_dlms.scoring import example_reward
from fewshot_dlms.scoring import family_metrics
from fewshot_dlms.scoring import pool_weights

L1 = np.array([-1.0, -2.0, -3.0])
L0 = np.array([-2.0, -2.0, -1.0])


def test_beta_zero_is_uniform():
    assert np.allclose(pool_weights(L1, L0, 0.0), np.ones(3) / 3)


def test_beta_half_matches_closed_form():
    w = pool_weights(L1, L0, 0.5)
    R = L1 - L0
    assert np.isclose(w[0] / w[1], np.exp(0.5 * (R[0] - R[1])))


def test_beta_half_matches_explicit_softmax():
    # R = [1, 0, -2], so the weights are softmax(0.5 R) = softmax([0.5, 0, -1]).
    expected = np.exp([0.5, 0.0, -1.0])
    expected = expected / expected.sum()
    assert np.allclose(pool_weights(L1, L0, 0.5), expected, rtol=1e-12, atol=0.0)


def test_sharpening_two_thirds_is_p1_half():
    w = pool_weights(L1, L0, 0.0, tau=2 / 3)
    assert np.isclose(w[0] / w[1], np.exp(0.5 * (L1[0] - L1[1])))


def test_tau_and_beta_combine_additively_in_the_exponent():
    w = pool_weights(L1, L0, 0.25, tau=0.5)
    exponent = (1 / 0.5 - 1) * L1 + 0.25 * (L1 - L0)
    expected = np.exp(exponent - exponent.max())
    assert np.allclose(w, expected / expected.sum(), rtol=1e-12, atol=0.0)


def test_weights_are_a_distribution_even_for_very_negative_log_likelihoods():
    ell1 = np.array([-12000.0, -12003.0, -12010.0])
    ell0 = np.array([-12001.0, -12001.0, -12001.0])
    w = pool_weights(ell1, ell0, 1.0)
    assert np.all(np.isfinite(w))
    assert np.isclose(w.sum(), 1.0)
    assert w.argmax() == 0


def test_weights_accept_lists():
    assert np.allclose(pool_weights([-1.0, -2.0], [-2.0, -2.0], 0.0), [0.5, 0.5])


def test_family_metrics():
    c = np.array([[1, 1], [1, 0], [0, 0]])
    w = np.array([0.5, 0.25, 0.25])
    m = family_metrics(c, w)
    assert np.isclose(m["A_all"], 0.5)
    assert np.isclose(m["A_target"], 0.5 * 1 + 0.25 * 0.5)
    assert np.isclose(m["ess"], 1 / (w ** 2).sum())
    assert np.isclose(m["max_w"], 0.5)


def test_family_metrics_keys_and_types():
    m = family_metrics(np.ones((3, 1)), np.ones(3) / 3)
    assert set(m) == {"A_target", "A_all", "ess", "max_w"}
    assert all(type(v) is float for v in m.values())
    assert np.isclose(m["ess"], 3.0)


def test_reward():
    assert np.allclose(example_reward(L1, L0), L1 - L0)


def test_reward_accepts_lists():
    assert np.allclose(example_reward([-1, -2], [-2, -2]), [1.0, 0.0])
