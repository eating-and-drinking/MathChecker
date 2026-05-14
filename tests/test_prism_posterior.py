from __future__ import annotations

import math

from mathchecker.prism.posterior import Posterior, length_prior, uniform_prior


def test_uniform_prior_sums_to_one() -> None:
    probs = uniform_prior(5)
    assert len(probs) == 6
    assert math.isclose(sum(probs), 1.0)


def test_length_prior_respects_no_error_mass() -> None:
    probs = length_prior(5, p_no_error=0.3)
    assert math.isclose(probs[-1], 0.3, abs_tol=1e-9)
    assert math.isclose(sum(probs), 1.0, abs_tol=1e-9)


def test_posterior_bayes_update_normalizes() -> None:
    pi = Posterior(num_steps=3)
    assert math.isclose(sum(pi.probs), 1.0)
    likelihood = [0.1, 0.9, 0.1, 0.1]
    pi.bayes_update(likelihood)
    assert math.isclose(sum(pi.probs), 1.0, abs_tol=1e-9)
    assert pi.argmax_index() == 1


def test_posterior_argmax_returns_minus_one_for_no_error() -> None:
    pi = Posterior(num_steps=4)
    likelihood = [0.05, 0.05, 0.05, 0.05, 0.95]
    pi.bayes_update(likelihood)
    assert pi.argmax_index() == -1
    assert pi.p_no_error() > 0.5


def test_phi_invariant_concentrates_on_first_contradiction() -> None:
    pi = Posterior(num_steps=4)
    for t in range(4):
        pi.mark_observed(t)
    pi.apply_phi_invariant(contradiction_strength_at=[0.05, 0.05, 0.95, 0.95])
    assert pi.argmax_index() == 2


def test_phi_invariant_zero_contradiction_concentrates_on_no_error() -> None:
    pi = Posterior(num_steps=3)
    for t in range(3):
        pi.mark_observed(t)
    pi.apply_phi_invariant(contradiction_strength_at=[0.02, 0.02, 0.02])
    assert pi.argmax_index() == -1


def test_phi_invariant_unobserved_steps_remain_in_play() -> None:
    pi = Posterior(num_steps=4)
    pi.mark_observed(0)
    pi.apply_phi_invariant(contradiction_strength_at=[0.02, 0.0, 0.0, 0.0])
    for tau in range(1, 4):
        assert pi.p_first_mistake_at(tau) > 0.05
    assert pi.p_no_error() > 0.05


def test_bayes_update_with_zero_likelihood_eliminates_hypothesis() -> None:
    pi = Posterior(num_steps=2)
    pi.bayes_update([0.0, 1.0, 1.0])
    assert math.isclose(pi.probs[0], 0.0, abs_tol=1e-9)
    assert math.isclose(sum(pi.probs), 1.0, abs_tol=1e-9)


def test_entropy_is_max_for_uniform() -> None:
    pi = Posterior(num_steps=3)
    expected = math.log(4)
    assert math.isclose(pi.entropy(), expected, abs_tol=1e-9)


def test_gibbs_refine_handles_empty_logits() -> None:
    pi = Posterior(num_steps=2)
    pi.gibbs_refine(
        per_step_label_logits=[[], []],
        contradiction_label_index=3,
    )
    assert math.isclose(sum(pi.probs), 1.0, abs_tol=1e-9)


def test_copy_is_independent() -> None:
    pi = Posterior(num_steps=2)
    clone = pi.copy()
    clone.bayes_update([0.0, 1.0, 0.0])
    assert clone.probs != pi.probs
