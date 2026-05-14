"""Expected Information Gain (EIG) routing.

Selecting which specialist tools to invoke at step t is cast as a one-step
Bayesian optimal experimental design problem:

    S* = argmax_S  I(tau ; V_{t+1}^{(S)} | E_{1:t})  -  lambda * c(S)

where V^{(S)} is the joint emission of the selected specialist subset.

Under the conditional-independence assumption between specialists (Krause &
Guestrin 2005), the function f(S) = I(tau; V^{(S)} | E_{1:t}) is monotone
submodular, so greedy selection gives a (1 - 1/e)-approximation. For
cost-aware variants, the standard partial-enumeration + greedy trick gives a
(1/2)(1 - 1/e) approximation (Sviridenko 2004); we use greedy directly which
in practice is essentially optimal at our budget sizes (max k <= 3).

Per-specialist EIG is computed by marginalizing over a small discrete grid of
possible specialist outcomes (here: a 5-bin grid for hard-conflict strength).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .likelihoods import make_specialist_likelihood
from .posterior import Posterior


# Discrete outcome grid for hard-conflict strength.
_HARD_CONFLICT_GRID = (0.05, 0.25, 0.5, 0.75, 0.95)
# Prior over outcomes when computing EIG (uniform; could be learned).
_OUTCOME_PRIOR = tuple(1.0 / len(_HARD_CONFLICT_GRID) for _ in _HARD_CONFLICT_GRID)


@dataclass(slots=True, frozen=True)
class SpecialistCandidate:
    name: str
    cost: float = 1.0
    sensitivity: float = 0.75
    # Optional prior on outcomes; if None, uniform.
    outcome_prior: tuple[float, ...] | None = None


def expected_information_gain(
    *,
    posterior: Posterior,
    step_index: int,
    candidate: SpecialistCandidate,
    outcome_grid: Sequence[float] = _HARD_CONFLICT_GRID,
) -> float:
    """Closed-form EIG of running `candidate` at `step_index` given posterior.

    EIG = H(tau | E) - E_V [ H(tau | E, V) ]

    Computed by:
      1. enumerating a discrete grid of candidate outcomes V
      2. for each, computing p(V | E) = sum_tau p(V | tau) * p(tau | E)
      3. computing the posterior after seeing V and its entropy
      4. averaging entropies weighted by p(V | E)
    """
    prior = candidate.outcome_prior if candidate.outcome_prior else tuple(
        1.0 / len(outcome_grid) for _ in outcome_grid
    )
    if len(prior) != len(outcome_grid):
        raise ValueError("outcome_prior length must match outcome_grid")

    h_before = posterior.entropy()
    expected_h_after = 0.0
    total_weight = 0.0

    for outcome_idx, hard_conflict in enumerate(outcome_grid):
        lik = make_specialist_likelihood(
            step_index=step_index,
            num_steps=posterior.num_steps,
            hard_conflict_strength=hard_conflict,
            valid_alternative_strength=0.0,
            sensitivity=candidate.sensitivity,
            source=candidate.name,
        ).values

        # p(V = v | E) = sum_tau pi(tau) * p(v | tau)
        p_v = 0.0
        for tau_idx, pi_val in enumerate(posterior.probs):
            p_v += pi_val * lik[tau_idx]
        if p_v <= 0.0:
            continue

        # Posterior after V = v
        unnorm = [pi_val * lik[idx] for idx, pi_val in enumerate(posterior.probs)]
        z = sum(unnorm)
        if z <= 0.0:
            continue
        inv = 1.0 / z
        post_after = [u * inv for u in unnorm]

        h_after = 0.0
        for p in post_after:
            if p > 0.0:
                h_after -= p * math.log(p)

        expected_h_after += p_v * h_after
        total_weight += p_v

    if total_weight > 0.0:
        expected_h_after /= total_weight

    eig = h_before - expected_h_after
    # Clamp tiny negatives from FP error.
    return max(0.0, eig)


def greedy_select(
    *,
    posterior: Posterior,
    step_index: int,
    candidates: Iterable[SpecialistCandidate],
    budget: int = 3,
    lam: float = 0.05,
) -> tuple[list[SpecialistCandidate], dict[str, float]]:
    """Greedy submodular selection of specialist candidates.

    Returns (selected_candidates, eig_scores) where eig_scores maps each
    candidate's name to its standalone EIG against the current posterior
    (for logging / training the EIG regressor).

    The greedy loop adds the candidate with highest *marginal* EIG-minus-cost
    until either:
      - marginal EIG < lam * cost   (no profitable selection left)
      - budget is reached
      - all candidates are exhausted
    """
    remaining = list(candidates)
    selected: list[SpecialistCandidate] = []
    standalone_scores: dict[str, float] = {}

    # Precompute standalone EIG for logging.
    for cand in remaining:
        standalone_scores[cand.name] = expected_information_gain(
            posterior=posterior,
            step_index=step_index,
            candidate=cand,
        )

    # Greedy with marginal EIG against a working posterior.
    working = posterior.copy()
    for _ in range(max(budget, 0)):
        if not remaining:
            break
        best_gain = -math.inf
        best_idx = -1
        best_cand: SpecialistCandidate | None = None
        for idx, cand in enumerate(remaining):
            gain = expected_information_gain(
                posterior=working,
                step_index=step_index,
                candidate=cand,
            )
            score = gain - lam * cand.cost
            if score > best_gain:
                best_gain = score
                best_idx = idx
                best_cand = cand
        if best_cand is None or best_gain <= 0.0:
            break
        selected.append(best_cand)
        # Update working posterior with the EXPECTED specialist emission
        # (this is the standard "lookahead" trick in submodular sensor
        # placement -- using the expected posterior keeps the surrogate
        # comparable while remaining (1-1/e)-competitive).
        _apply_expected_emission(working, step_index, best_cand)
        remaining.pop(best_idx)

    return selected, standalone_scores


def _apply_expected_emission(
    posterior: Posterior,
    step_index: int,
    candidate: SpecialistCandidate,
) -> None:
    """Update posterior by the expected likelihood under the outcome prior.

    Equivalent to integrating out V before observing it, which yields a
    submodular surrogate for greedy lookahead.
    """
    grid = _HARD_CONFLICT_GRID
    prior = candidate.outcome_prior if candidate.outcome_prior else _OUTCOME_PRIOR
    expected_lik = [0.0] * posterior.num_hypotheses
    for hard_conflict, p_v in zip(grid, prior):
        lik = make_specialist_likelihood(
            step_index=step_index,
            num_steps=posterior.num_steps,
            hard_conflict_strength=hard_conflict,
            sensitivity=candidate.sensitivity,
            source=candidate.name,
        ).values
        for idx, val in enumerate(lik):
            expected_lik[idx] += p_v * val
    posterior.bayes_update(expected_lik)


# ---- Default specialist registry for the legacy three-tool triad ----

DEFAULT_SPECIALIST_CANDIDATES: tuple[SpecialistCandidate, ...] = (
    SpecialistCandidate(
        name="alternative_route_verifier_tool",
        cost=1.0,
        sensitivity=0.70,
    ),
    SpecialistCandidate(
        name="equivalence_substitution_verifier_tool",
        cost=1.0,
        sensitivity=0.78,
    ),
    SpecialistCandidate(
        name="condition_obligation_verifier_tool",
        cost=1.0,
        sensitivity=0.78,
    ),
)


def default_candidate_by_name(name: str) -> SpecialistCandidate:
    for cand in DEFAULT_SPECIALIST_CANDIDATES:
        if cand.name == name:
            return cand
    raise KeyError(name)
