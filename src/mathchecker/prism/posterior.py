"""Posterior pi_t(tau) over the first-mistake index tau in {1, ..., T, infty}.

Implementation notes
--------------------
- We use a length-(T+1) probability vector indexed 0..T. Position 0..T-1 means
  "the first mistake is at step (index+1)" (one-indexed semantics so it matches
  the existing pred_first_mistake_index convention, which is also 1-indexed
  externally but 0-indexed in TracePrediction.pred_first_mistake_index --
  callers should translate). Position T is the "no error" / tau = infty bucket.
- All math is plain Python; no numpy dependency to avoid bloating the wheel.
- A Phi-invariant flag per index records whether that tau-hypothesis is still
  compatible with the observed step-label evidence so far. When invariant is
  violated, the posterior mass on that index is zeroed and the remaining mass
  is renormalized. This is the Bayesian incarnation of the old deterministic
  fallback layer.

Performance: O(T) per Bayesian update; total inference cost is O(T^2), trivially
fast compared to the LLM calls that produce evidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


_TAU_INFTY = -1  # sentinel returned by argmax_index when tau = infty wins


@dataclass(slots=True)
class Posterior:
    """A discrete distribution over tau in {0, 1, ..., T-1, infty}.

    Index k in [0, T) means "first mistake at step k" (0-indexed). Index T is
    the "no error" hypothesis. Invariant: probabilities are non-negative and
    sum to 1 up to numerical tolerance.
    """

    num_steps: int
    probs: list[float] = field(default_factory=list)
    # Index t is True iff the trace has been observed (an evidence update has
    # been applied for that step). Used by Phi-invariant projection: once we
    # observe step t and it looks consistent, hypotheses tau > t cannot have
    # been "blocked" by step t.
    observed_up_to: int = 0  # number of steps for which evidence has been seen
    # Mixture weight for the uniform-prior floor. After each update we
    # interpolate pi <- (1 - floor_weight) * pi + floor_weight * uniform.
    # This guarantees no hypothesis is ever crushed to 0, so a posterior
    # collapse from early multiplicative updates can be recovered if later
    # evidence contradicts. Equivalent to a Bayesian prior of weight
    # `floor_weight` against perfect-confidence in any single hypothesis.
    floor_weight: float = 1e-4

    def __post_init__(self) -> None:
        if self.num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if not 0.0 <= self.floor_weight < 1.0:
            raise ValueError("floor_weight must be in [0, 1)")
        if not self.probs:
            self.probs = uniform_prior(self.num_steps)
        elif len(self.probs) != self.num_steps + 1:
            raise ValueError(
                f"probs length {len(self.probs)} does not match T+1 = {self.num_steps + 1}"
            )
        self._renormalize()
        self._apply_floor()

    # ---- accessors ----

    @property
    def num_hypotheses(self) -> int:
        return self.num_steps + 1

    def p_no_error(self) -> float:
        return self.probs[self.num_steps]

    def p_first_mistake_at(self, step_index: int) -> float:
        if step_index < 0 or step_index >= self.num_steps:
            raise IndexError(step_index)
        return self.probs[step_index]

    def entropy(self) -> float:
        """Shannon entropy of pi in nats."""
        h = 0.0
        for p in self.probs:
            if p > 0.0:
                h -= p * math.log(p)
        return h

    def max_mass(self) -> float:
        return max(self.probs)

    def argmax_index(self) -> int:
        """Return the MAP index. -1 means tau = infty (no error)."""
        best_idx = 0
        best_val = self.probs[0]
        for idx in range(1, self.num_hypotheses):
            if self.probs[idx] > best_val:
                best_val = self.probs[idx]
                best_idx = idx
        return _TAU_INFTY if best_idx == self.num_steps else best_idx

    def copy(self) -> "Posterior":
        return Posterior(
            num_steps=self.num_steps,
            probs=list(self.probs),
            observed_up_to=self.observed_up_to,
        )

    # ---- updates ----

    def bayes_update(self, likelihood: Sequence[float]) -> None:
        """Multiplicative posterior update.

        likelihood[k] = p(evidence | tau = k) for k in 0..T-1
        likelihood[T] = p(evidence | tau = infty)

        Likelihood entries must be non-negative. Zeros are allowed (they
        eliminate that hypothesis). The result is renormalized.
        """
        if len(likelihood) != self.num_hypotheses:
            raise ValueError(
                f"likelihood length {len(likelihood)} != T+1 = {self.num_hypotheses}"
            )
        if any(value < 0.0 for value in likelihood):
            raise ValueError("likelihood entries must be non-negative")
        new_probs = [p * lik for p, lik in zip(self.probs, likelihood)]
        total = sum(new_probs)
        if total <= 0.0:
            # Degenerate: evidence is impossible under every hypothesis. Fall
            # back to keeping the prior unchanged rather than producing NaNs.
            return
        inv = 1.0 / total
        self.probs = [p * inv for p in new_probs]
        self._apply_floor()

    def mark_observed(self, step_index: int) -> None:
        """Record that step `step_index` has been processed."""
        if step_index + 1 > self.observed_up_to:
            self.observed_up_to = step_index + 1

    def apply_phi_invariant(
        self,
        *,
        contradiction_strength_at: Sequence[float],
    ) -> None:
        """Project posterior onto the Phi-invariant support.

        Phi requires:
          (a) tau >= some index k iff step k did not exhibit contradiction
              evidence.
          (b) tau = infty requires NO step to show contradiction evidence.
          (c) tau = k requires step k to be the FIRST step with contradiction
              evidence (so all steps j < k look clean).

        `contradiction_strength_at[t]` is a soft score in [0, 1] indicating how
        strongly step t looked like a hard contradiction. We turn this into a
        soft multiplier rather than a hard 0/1 cut, so it remains a Bayesian
        update under a noisy-channel model.

        We only enforce Phi over OBSERVED steps (0 <= t < observed_up_to).
        Unobserved future steps contribute a neutral multiplier of 1 so that
        hypotheses tau = (unobserved k) and tau = infty remain in play until
        we have actually examined those steps.
        """
        if len(contradiction_strength_at) != self.num_steps:
            raise ValueError("contradiction_strength_at length must equal num_steps")

        eps = 1e-6
        multipliers = [1.0] * self.num_hypotheses
        running_clean = 1.0
        observed_T = max(0, min(self.observed_up_to, self.num_steps))

        # Over observed steps, the invariant tightens the support.
        for k in range(observed_T):
            c_k = max(0.0, min(1.0, float(contradiction_strength_at[k])))
            multipliers[k] = running_clean * max(c_k, eps)
            running_clean *= max(1.0 - c_k, eps)

        # For unobserved steps and tau = infty: multiplier inherits running_clean
        # (consistency over what we *have* seen) without further damping. This
        # leaves the relative balance among unexamined hypotheses unchanged.
        for k in range(observed_T, self.num_steps):
            multipliers[k] = running_clean
        multipliers[self.num_steps] = running_clean

        self.bayes_update(multipliers)

    # ---- helpers ----

    def _renormalize(self) -> None:
        total = sum(self.probs)
        if total <= 0.0:
            self.probs = uniform_prior(self.num_steps)
            return
        inv = 1.0 / total
        self.probs = [max(0.0, p * inv) for p in self.probs]

    def _apply_floor(self) -> None:
        """Interpolate the posterior with a uniform prior of weight `floor_weight`.

        pi' = (1 - w) * pi + w * uniform

        This guarantees the smallest posterior mass is at least
        `floor_weight / num_hypotheses`, so no hypothesis is irrecoverably
        crushed to 0 by multiplicative Bayes updates. Mathematically
        equivalent to maintaining a (1-w):w mixture between the data-driven
        belief and the uniform reference prior.
        """
        w = self.floor_weight
        if w <= 0.0:
            return
        n = self.num_hypotheses
        if n == 0:
            return
        uniform_mass = 1.0 / n
        self.probs = [(1.0 - w) * p + w * uniform_mass for p in self.probs]

    def as_dict(self) -> dict:
        return {
            "num_steps": self.num_steps,
            "probs": list(self.probs),
            "observed_up_to": self.observed_up_to,
            "p_no_error": self.p_no_error(),
            "max_mass": self.max_mass(),
            "entropy_nats": self.entropy(),
            "argmax_index": self.argmax_index(),
            "floor_weight": self.floor_weight,
        }


# ---- factories ----

def uniform_prior(num_steps: int) -> list[float]:
    """Uniform prior over {0, ..., T-1, infty}."""
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")
    n = num_steps + 1
    return [1.0 / n] * n


def length_prior(num_steps: int, *, p_no_error: float = 0.4) -> list[float]:
    """Trace-length-aware prior.

    Concentrates "first mistake" mass on the middle of the trajectory
    (errors at the start are easier to catch by humans, errors at the
    very end are rarer in practice) and reserves p_no_error for tau=infty.
    """
    if not 0.0 <= p_no_error <= 1.0:
        raise ValueError("p_no_error must be in [0, 1]")
    if num_steps == 0:
        return [1.0]
    weights = []
    half = (num_steps - 1) / 2.0 if num_steps > 1 else 0.0
    for k in range(num_steps):
        if num_steps == 1:
            weights.append(1.0)
        else:
            weights.append(1.0 - abs(k - half) / (half + 1.0))
    total = sum(weights)
    if total <= 0.0:
        weights = [1.0] * num_steps
        total = float(num_steps)
    mass_for_errors = 1.0 - p_no_error
    probs = [(w / total) * mass_for_errors for w in weights]
    probs.append(p_no_error)
    return probs
