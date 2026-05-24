"""Anytime-valid stopping via e-processes for PRISM.

Replaces (or runs alongside) the split-conformal Bonferroni schedule with a
test-martingale construction that provides anytime-valid coverage without
requiring a calibration set.

Theoretical foundation
----------------------
Let pi_0 be a reference prior over tau in {0, ..., T-1, infty} and let
{pi_t}_{t=0}^{T} be the posterior trajectory PRISM maintains. For each
hypothesis k define

    M_t(k) = pi_t(k) / pi_0(k).

Because pi_t is obtained from pi_0 via a sequence of multiplicative Bayes
updates with non-negative likelihoods, {M_t(k)}_{t >= 0} is a non-negative
process with E[M_t(k)] = 1 under the null

    H_0^k : "the data-generating mechanism gives the same expected likelihood
             to tau = k as the reference prior puts on it,
             i.e. tau != k is observationally indistinguishable from tau = k
             on average."

This makes {M_t(k)} a (super-)martingale w.r.t. its natural filtration, and
Ville's inequality (Ville 1939; see Howard, Ramdas, McAuliffe & Sekhon 2021
for the modern treatment) gives the time-uniform bound

    P( sup_{t >= 0} M_t(k) >= 1/delta | H_0^k )  <=  delta.

Anytime-valid commit rule
-------------------------
At step t, let k_t = argmax_k pi_t(k). PRISM commits to k_t iff

    M_t(k_t) >= tau_delta.

Two guarantee modes are available:

  - "per_target" (default, tau_delta = 1/delta).
        Per-hypothesis anytime-valid miscoverage bound:
            for any fixed k,  P_{tau != k}( commit to k at any time ) <= delta.
        Trace-level miscoverage (commit to ANY wrong k) is at most (T+1) delta
        via union bound -- matching conformal Bonferroni's rate without
        needing calibration data.

  - "bonferroni" (tau_delta = (T+1)/delta).
        Trace-level anytime-valid: P( commit to any wrong k ) <= delta.
        Strictly more conservative; rarely practical at moderate delta.

  - "none" : alias for "per_target" (kept for back-compat with early drafts).

Advantages over split-conformal
-------------------------------
1. ANYTIME VALID. The bound holds for any data-adaptive stopping rule, not
   just the fixed schedule used during calibration.
2. CALIBRATION-FREE. The threshold is determined analytically from delta and
   the prior; no held-out calibration set is required.
3. PRIOR-AWARE. The threshold tightens automatically for hypotheses that
   are rare under the prior (small pi_0(k) -> larger e-value for the same
   posterior mass). Conformal floors at 1 - delta regardless.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EProcessSchedule:
    """Anytime-valid stopping rule for PRISM based on test-martingale e-processes.

    Parameters
    ----------
    delta : float in (0, 1)
        Target miscoverage. With multiplicity="per_target" this is a
        per-hypothesis bound. With multiplicity="bonferroni" it is a
        trace-level bound.
    prior : list[float]
        Reference prior pi_0 over {0, ..., T-1, infty}. Length must equal
        T+1. Non-negative and normalized in __post_init__.
    num_steps : int
        T (number of reasoning steps; pi has T+1 entries including tau=infty).
    multiplicity : "per_target" (default), "bonferroni", or "none"
        See module docstring for semantics.
    """

    delta: float
    prior: list[float]
    num_steps: int
    multiplicity: str = "per_target"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if self.num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if len(self.prior) != self.num_steps + 1:
            raise ValueError(
                f"prior length {len(self.prior)} != T+1 = {self.num_steps + 1}"
            )
        if any(p < 0.0 for p in self.prior):
            raise ValueError("prior entries must be non-negative")
        total = sum(self.prior)
        if total <= 0.0:
            raise ValueError("prior must have positive total mass")
        self.prior = [p / total for p in self.prior]
        if self.multiplicity not in ("bonferroni", "none", "per_target"):
            raise ValueError(f"unknown multiplicity correction: {self.multiplicity}")

    @property
    def threshold(self) -> float:
        """Required e-value M_t(k_t) to commit (under chosen multiplicity)."""
        base = 1.0 / self.delta
        if self.multiplicity == "bonferroni":
            return float(self.num_steps + 1) * base
        return base  # per_target and none

    def evalue_at(
        self,
        *,
        posterior_probs: list[float],
        hypothesis_index: int,
    ) -> float:
        """M_t(k) = pi_t(k) / pi_0(k)."""
        if len(posterior_probs) != len(self.prior):
            raise ValueError("posterior length must equal prior length")
        if hypothesis_index < 0 or hypothesis_index >= len(self.prior):
            raise IndexError(hypothesis_index)
        p0 = self.prior[hypothesis_index]
        if p0 <= 0.0:
            return float("inf") if posterior_probs[hypothesis_index] > 0 else 0.0
        return posterior_probs[hypothesis_index] / p0

    def evalues(self, *, posterior_probs: list[float]) -> list[float]:
        if len(posterior_probs) != len(self.prior):
            raise ValueError("posterior length must equal prior length")
        out: list[float] = []
        for k, p0 in enumerate(self.prior):
            if p0 > 0.0:
                out.append(posterior_probs[k] / p0)
            else:
                out.append(float("inf") if posterior_probs[k] > 0 else 0.0)
        return out

    def should_commit(
        self,
        *,
        step_index: int = -1,
        max_posterior_mass: float = 0.0,
        posterior_probs: list[float] | None = None,
        argmax_index: int = -1,
    ) -> bool:
        """Anytime-valid commit predicate.

        argmax_index follows Posterior.argmax_index() convention:
          -1 -> tau = infty (absolute index T)
           k -> tau = step k
        step_index and max_posterior_mass are accepted but unused (for
        protocol parity with ConformalSchedule.should_commit).
        """
        if posterior_probs is None:
            raise ValueError("EProcessSchedule.should_commit requires posterior_probs")
        if len(posterior_probs) != len(self.prior):
            raise ValueError("posterior length must equal prior length")
        absolute = (len(self.prior) - 1) if argmax_index == -1 else int(argmax_index)
        m = self.evalue_at(posterior_probs=posterior_probs, hypothesis_index=absolute)
        return m >= self.threshold

    def to_dict(self) -> dict:
        return {
            "type": "eprocess",
            "delta": self.delta,
            "prior": list(self.prior),
            "num_steps": self.num_steps,
            "multiplicity": self.multiplicity,
            "threshold": self.threshold,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "EProcessSchedule":
        return cls(
            delta=float(payload.get("delta", 0.1)),
            prior=list(payload.get("prior", [])),
            num_steps=int(payload.get("num_steps", 0)),
            multiplicity=str(payload.get("multiplicity", "per_target")),
            meta=dict(payload.get("meta", {})),
        )


def default_eprocess(
    *,
    num_steps: int,
    delta: float = 0.1,
    p_no_error: float = 0.4,
) -> EProcessSchedule:
    """Length-aware default e-process schedule."""
    if num_steps <= 0:
        return EProcessSchedule(
            delta=delta,
            prior=[1.0],
            num_steps=0,
            multiplicity="per_target",
            meta={"type": "default_eprocess", "trivial": True},
        )
    from .posterior import length_prior  # local import to avoid cycle

    prior = length_prior(num_steps, p_no_error=p_no_error)
    return EProcessSchedule(
        delta=delta,
        prior=prior,
        num_steps=num_steps,
        multiplicity="per_target",
        meta={"type": "default_eprocess", "p_no_error": p_no_error},
    )
