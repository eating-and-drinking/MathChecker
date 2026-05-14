from __future__ import annotations

from mathchecker.prism.eig import (
    DEFAULT_SPECIALIST_CANDIDATES,
    SpecialistCandidate,
    expected_information_gain,
    greedy_select,
)
from mathchecker.prism.posterior import Posterior, length_prior


def test_eig_is_non_negative() -> None:
    pi = Posterior(num_steps=3)
    cand = SpecialistCandidate(name="alternative_route_verifier_tool", cost=1.0, sensitivity=0.75)
    eig = expected_information_gain(posterior=pi, step_index=1, candidate=cand)
    assert eig >= 0.0


def test_eig_is_zero_when_posterior_is_certain() -> None:
    pi = Posterior(num_steps=3)
    # Concentrate all mass on tau = 1.
    pi.bayes_update([1e-9, 1.0, 1e-9, 1e-9])
    cand = SpecialistCandidate(name="alternative_route_verifier_tool", cost=1.0, sensitivity=0.75)
    eig = expected_information_gain(posterior=pi, step_index=1, candidate=cand)
    # Already certain; no further information to gain.
    assert eig < 0.05


def test_eig_decreases_as_sensitivity_decreases() -> None:
    pi = Posterior(num_steps=3)
    high_sens = SpecialistCandidate(name="x", cost=1.0, sensitivity=0.95)
    low_sens = SpecialistCandidate(name="x", cost=1.0, sensitivity=0.1)
    high = expected_information_gain(posterior=pi, step_index=1, candidate=high_sens)
    low = expected_information_gain(posterior=pi, step_index=1, candidate=low_sens)
    assert high >= low


def test_greedy_select_respects_budget() -> None:
    pi = Posterior(num_steps=4, probs=length_prior(4, p_no_error=0.4))
    selected, scores = greedy_select(
        posterior=pi,
        step_index=2,
        candidates=DEFAULT_SPECIALIST_CANDIDATES,
        budget=2,
        lam=0.0,
    )
    assert len(selected) <= 2
    for cand in DEFAULT_SPECIALIST_CANDIDATES:
        assert cand.name in scores


def test_greedy_select_skips_when_cost_exceeds_value() -> None:
    pi = Posterior(num_steps=3)
    # Force EIG to be much smaller than lam * cost by concentrating mass.
    pi.bayes_update([1e-9, 1.0, 1e-9, 1e-9])
    selected, _scores = greedy_select(
        posterior=pi,
        step_index=1,
        candidates=DEFAULT_SPECIALIST_CANDIDATES,
        budget=3,
        lam=1.0,  # heavy cost penalty
    )
    # Nothing profitable to invoke when posterior is already certain.
    assert selected == []


def test_eig_submodular_diminishing_returns_property() -> None:
    """Marginal EIG should be non-increasing as we add more specialists.

    This is the submodularity property invoked by Theorem 1 in PRISM_ALGORITHM.md.
    We don't claim strict diminishing returns on every pair (small numerical
    noise can flip ties), but the *sum* of marginal gains under greedy must be
    at most the sum of standalone EIGs (sublinear, not super-linear).
    """
    pi = Posterior(num_steps=4, probs=length_prior(4, p_no_error=0.4))
    # Standalone EIG sum.
    standalone_sum = sum(
        expected_information_gain(posterior=pi, step_index=2, candidate=c)
        for c in DEFAULT_SPECIALIST_CANDIDATES
    )
    # Greedy selection with no cost.
    selected, _ = greedy_select(
        posterior=pi,
        step_index=2,
        candidates=DEFAULT_SPECIALIST_CANDIDATES,
        budget=len(DEFAULT_SPECIALIST_CANDIDATES),
        lam=0.0,
    )
    # Replay the greedy gains on a fresh posterior using true expected-emission update.
    working = pi.copy()
    from mathchecker.prism.eig import _apply_expected_emission

    greedy_sum = 0.0
    for cand in selected:
        gain = expected_information_gain(posterior=working, step_index=2, candidate=cand)
        greedy_sum += gain
        _apply_expected_emission(working, 2, cand)

    # Submodular f gives sum of marginal gains <= sum of standalone gains.
    assert greedy_sum <= standalone_sum + 1e-6
