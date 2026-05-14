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

    def __post_init__(self) -> None:
        if self.num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if not self.probs:
            self.probs = uniform_prior(self.num_steps)
        elif len(self.probs) != self.num_steps + 1:
            raise ValueError(
                f"probs length {len(self.probs)} does not match T+1 = {self.num_steps + 1}"
            )
        self._renormalize()

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

    def gibbs_refine(
        self,
        *,
        per_step_label_logits: Sequence[Sequence[float]],
        contradiction_label_index: int,
    ) -> None:
        """One Gibbs step that re-fuses the per-step label posterior into pi.

        This subsumes the legacy stage2_review + stage2_specialist_review
        path. Given calibrated per-step label logits, we compute the marginal
        probability of "this step is a hard contradiction" and re-apply the
        Phi-invariant softly.

        per_step_label_logits[t] is a logit vector over PRINCIPLE_LABELS for
        step t. We extract the softmax probability of the contradiction class
        and pass it through apply_phi_invariant.
        """
        if len(per_step_label_logits) != self.num_steps:
            raise ValueError("per_step_label_logits length must equal num_steps")
        contradiction_scores: list[float] = []
        for logits in per_step_label_logits:
            if not logits:
                contradiction_scores.append(0.0)
                continue
            m = max(logits)
            exps = [math.exp(v - m) for v in logits]
            denom = sum(exps)
            if denom <= 0.0:
                contradiction_scores.append(0.0)
                continue
            idx = contradiction_label_index
            if idx < 0 or idx >= len(exps):
                contradiction_scores.append(0.0)
                continue
            contradiction_scores.append(exps[idx] / denom)
        self.apply_phi_invariant(contradiction_strength_at=contradiction_scores)

    # ---- helpers ----

    def _renormalize(self) -> None:
        total = sum(self.probs)
        if total <= 0.0:
            self.probs = uniform_prior(self.num_steps)
            return
        inv = 1.0 / total
        self.probs = [max(0.0, p * inv) for p in self.probs]

    def as_dict(self) -> dict:
        return {
            "num_steps": self.num_steps,
            "probs": list(self.probs),
            "observed_up_to": self.observed_up_to,
            "p_no_error": self.p_no_error(),
            "max_mass": self.max_mass(),
            "entropy_nats": self.entropy(),
            "argmax_index": self.argmax_index(),
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
    # Tent-shaped weighting over step indices.
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
