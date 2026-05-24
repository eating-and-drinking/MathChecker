from __future__ import annotations

import math

import pytest

from mathchecker.prism.posterior import Posterior, length_prior, uniform_prior


def test_uniform_prior_sums_to_one():
    probs = uniform_prior(5)
    assert len(probs) == 6
    assert math.isclose(sum(probs), 1.0)


def test_length_prior_respects_no_error_mass():
    probs = length_prior(5, p_no_error=0.3)
    assert math.isclose(probs[-1], 0.3, abs_tol=1e-9)
    assert math.isclose(sum(probs), 1.0, abs_tol=1e-9)


def test_posterior_bayes_update_normalizes():
    pi = Posterior(num_steps=3, floor_weight=0.0)
    assert math.isclose(sum(pi.probs), 1.0)
    pi.bayes_update([0.1, 0.9, 0.1, 0.1])
    assert math.isclose(sum(pi.probs), 1.0, abs_tol=1e-9)
    assert pi.argmax_index() == 1


def test_posterior_argmax_returns_minus_one_for_no_error():
    pi = Posterior(num_steps=4, floor_weight=0.0)
    pi.bayes_update([0.05, 0.05, 0.05, 0.05, 0.95])
    assert pi.argmax_index() == -1
    assert pi.p_no_error() > 0.5


def test_phi_invariant_concentrates_on_first_contradiction():
    pi = Posterior(num_steps=4)
    for t in range(4):
        pi.mark_observed(t)
    pi.apply_phi_invariant(contradiction_strength_at=[0.05, 0.05, 0.95, 0.95])
    assert pi.argmax_index() == 2


def test_phi_invariant_zero_contradiction_concentrates_on_no_error():
    pi = Posterior(num_steps=3)
    for t in range(3):
        pi.mark_observed(t)
    pi.apply_phi_invariant(contradiction_strength_at=[0.02, 0.02, 0.02])
    assert pi.argmax_index() == -1


def test_phi_invariant_unobserved_steps_remain_in_play():
    pi = Posterior(num_steps=4)
    pi.mark_observed(0)
    pi.apply_phi_invariant(contradiction_strength_at=[0.02, 0.0, 0.0, 0.0])
    for tau in range(1, 4):
        assert pi.p_first_mistake_at(tau) > 0.05
    assert pi.p_no_error() > 0.05


def test_bayes_update_with_zero_likelihood_eliminates_hypothesis():
    pi = Posterior(num_steps=2, floor_weight=0.0)
    pi.bayes_update([0.0, 1.0, 1.0])
    assert math.isclose(pi.probs[0], 0.0, abs_tol=1e-9)
    assert math.isclose(sum(pi.probs), 1.0, abs_tol=1e-9)


def test_entropy_is_max_for_uniform():
    pi = Posterior(num_steps=3, floor_weight=0.0)
    assert math.isclose(pi.entropy(), math.log(4), abs_tol=1e-9)


def test_copy_is_independent():
    pi = Posterior(num_steps=2, floor_weight=0.0)
    clone = pi.copy()
    clone.bayes_update([0.0, 1.0, 0.0])
    assert clone.probs != pi.probs


def test_posterior_floor_prevents_zero_mass():
    pi = Posterior(num_steps=3, floor_weight=1e-3)
    pi.bayes_update([1e-20, 1.0, 1.0, 1.0])
    expected_floor = 1e-3 / pi.num_hypotheses
    assert pi.probs[0] >= expected_floor * 0.5


def test_posterior_can_recover_from_near_total_crush():
    pi = Posterior(num_steps=3, floor_weight=1e-3)
    for _ in range(20):
        pi.bayes_update([0.01, 1.0, 1.0, 1.0])
    mass_before = pi.probs[0]
    pi.bayes_update([100.0, 0.01, 0.01, 0.01])
    mass_after = pi.probs[0]
    assert mass_after > mass_before * 10.0


def test_floor_zero_recovers_old_behavior():
    pi = Posterior(num_steps=2, floor_weight=0.0)
    pi.bayes_update([0.0, 1.0, 0.0])
    assert pi.probs[0] == 0.0


def test_floor_weight_out_of_range_rejected():
    with pytest.raises(ValueError):
        Posterior(num_steps=2, floor_weight=1.0)
    with pytest.raises(ValueError):
        Posterior(num_steps=2, floor_weight=-0.1)
