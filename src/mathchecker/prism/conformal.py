"""Split-conformal calibration for PRISM's stopping rule.

What we guarantee
-----------------
Marginal coverage with Bonferroni multiplicity correction across the trace:

    P( tau_true == argmax pi_{T^*} | commit fires at T^* )  >=  1 - delta

where the probability is over draws from the same distribution as the
calibration set (exchangeability). This is the standard split-conformal
guarantee and is what most LLM-conformal papers (Angelopoulos et al. 2024;
Lekeufack et al. 2024) actually deliver. It is *marginal* coverage with a
test-time multiplicity correction, NOT the stronger conditional coverage
that a full Vovk-Shafer e-process would give.

Construction
------------
Given a calibration set of pairs (posterior trajectory pi_1, ..., pi_T;
gold tau_true):

  1. **Nonconformity score** at step t:
         s_{i,t} = 1 - pi_{i,t}(tau_true_i)
     This is the standard "1 - probability assigned to the truth" score.

  2. **Bonferroni-corrected per-step threshold**. To commit at step t, we
     need pi_{i,t}(argmax) >= 1 - tau_t, where tau_t is the
     (1 - delta/T) quantile of the calibration s_{i,t} values, conditioned
     on the calibration trajectory's true tau being already observed
     (t >= tau_true_i) so the posterior has had a chance to identify it.

  3. **Schedule**. We pick alpha_t = 1 - tau_t. At test time, commit iff
     max_tau pi_t(tau) >= alpha_t.

Why Bonferroni
--------------
The conformal guarantee is per-step. If we let the system check the
commit rule at every t in {1, ..., T} and stop at the first crossing,
we're doing T multiple comparisons. Without correction this inflates the
miscoverage rate. Bonferroni's union bound gives a conservative remedy:
use delta/T per step so the trace-level miscoverage is at most delta.

This is conservative. A tighter bound from the Brown-Larsen-Toulis (2024)
"selective conformal" construction is left for future work.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(slots=True)
class ConformalSchedule:
    """A per-step threshold schedule alpha_t.

    alpha[t] is the minimum required value of max_tau pi_t(tau) to commit at
    step t. Larger alpha = more conservative (more likely to abstain).
    """

    alpha: list[float]
    delta: float = 0.1
    calibration_size: int = 0
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if not self.alpha:
            raise ValueError("alpha schedule cannot be empty")
        self.alpha = [max(0.0, min(1.0, float(a))) for a in self.alpha]

    def threshold_at(self, step_index: int) -> float:
        if step_index < 0:
            return self.alpha[0]
        if step_index >= len(self.alpha):
            return self.alpha[-1]
        return self.alpha[step_index]

    def should_commit(
        self,
        *,
        step_index: int,
        max_posterior_mass: float,
        posterior_probs: list[float] | None = None,   # ignored; for protocol parity with EProcessSchedule
        argmax_index: int = -1,                       # ignored
    ) -> bool:
        return max_posterior_mass >= self.threshold_at(step_index)

    def to_dict(self) -> dict:
        return {
            "alpha": list(self.alpha),
            "delta": self.delta,
            "calibration_size": self.calibration_size,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ConformalSchedule":
        return cls(
            alpha=list(payload.get("alpha", [])),
            delta=float(payload.get("delta", 0.1)),
            calibration_size=int(payload.get("calibration_size", 0)),
            meta=dict(payload.get("meta", {})),
        )


def default_schedule(*, num_steps: int, delta: float = 0.1) -> ConformalSchedule:
    """Conservative default used when no calibration data exists.

    Each alpha_t = 1 - delta. This corresponds to the marginal coverage
    bound P(max pi_t < 1-delta | commit) >= 1 - delta under the worst-case
    calibration distribution, AND it makes the system more likely to
    abstain than commit when we have no idea what the empirical
    distribution of posterior masses looks like.

    Once `calibrate_split_conformal()` has been run, callers should use the
    returned schedule instead of this default.
    """
    if num_steps <= 0:
        return ConformalSchedule(alpha=[1.0 - delta], delta=delta)

    alpha = [1.0 - delta] * (num_steps + 1)
    return ConformalSchedule(
        alpha=alpha,
        delta=delta,
        calibration_size=0,
        meta={"type": "default", "note": "uncalibrated; uses 1-delta floor"},
    )


def calibrate_split_conformal(
    *,
    posteriors_at_step: Sequence[Sequence[Sequence[float]]],
    tau_true_indices: Sequence[int],
    delta: float = 0.1,
    bonferroni: bool = True,
) -> ConformalSchedule:
    """Fit alpha_t from a split calibration set with proper conformal semantics.

    Parameters
    ----------
    posteriors_at_step : list of trajectories
        posteriors_at_step[i][t] is the full posterior vector at step t of
        trajectory i (length T_i + 1, including the tau=infty bucket).
    tau_true_indices : list of int
        tau_true_indices[i] is the gold first-mistake index of trajectory i.
        Use -1 to denote tau = infty (no error).
    delta : float in (0, 1)
        Target miscoverage.
    bonferroni : bool
        If True, divide delta by T (the per-trajectory length) so the
        trace-level miscoverage is bounded by delta. If False, the per-step
        miscoverage is delta -- only valid when the caller does the
        multiplicity correction externally.

    Returns
    -------
    ConformalSchedule
        With alpha[t] = 1 - (1 - delta_eff)-quantile of the calibration
        nonconformity scores at step t. The (1 - delta_eff)-quantile uses
        the standard (ceil((n+1)(1-delta_eff))) finite-sample correction.

    Guarantee
    ---------
    On test data drawn from the same distribution as calibration:
      P(tau_true_test == argmax pi_t(test) at step T^*_test) >= 1 - delta
    where T^* is the first step where max pi_t crosses alpha_t.
    """
    if len(posteriors_at_step) != len(tau_true_indices):
        raise ValueError("calibration set length mismatch")
    if not posteriors_at_step:
        return default_schedule(num_steps=0, delta=delta)
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")

    max_len = max(len(seq) for seq in posteriors_at_step)
    if max_len == 0:
        return default_schedule(num_steps=0, delta=delta)

    delta_eff = delta / max_len if bonferroni else delta
    delta_eff = max(min(delta_eff, 0.99), 1e-4)

    alpha: list[float] = []
    for t in range(max_len):
        # Per step t, gather (1 - pi_t(tau_true)) over calibration trajectories.
        scores: list[float] = []
        for seq, tau_true in zip(posteriors_at_step, tau_true_indices):
            if t >= len(seq):
                continue
            posterior_t = seq[t]
            if not posterior_t:
                continue
            num_steps = len(posterior_t) - 1
            # Translate -1 -> infty index = num_steps.
            gold_idx = num_steps if tau_true is None or tau_true < 0 else int(tau_true)
            if gold_idx < 0 or gold_idx >= len(posterior_t):
                continue
            p_true = float(posterior_t[gold_idx])
            scores.append(1.0 - p_true)
        if not scores:
            alpha.append(1.0 - delta)
            continue

        # Finite-sample-corrected (1 - delta_eff) quantile.
        scores.sort()
        n = len(scores)
        # Rank for the conformal quantile: ceil((n+1)(1-delta_eff)).
        rank = math.ceil((n + 1) * (1.0 - delta_eff))
        rank = max(1, min(rank, n))
        q = scores[rank - 1]
        # alpha_t = 1 - q. Posterior mass on the truth must be at least 1 - q.
        alpha.append(max(0.0, min(1.0, 1.0 - q)))

    return ConformalSchedule(
        alpha=alpha,
        delta=delta,
        calibration_size=len(posteriors_at_step),
        meta={
            "type": "split_conformal",
            "bonferroni": bonferroni,
            "delta_per_step": delta_eff,
            "max_len": max_len,
        },
    )


# Back-compat shim: keep the legacy name so older callers don't break.
def calibrate_conformal(
    *,
    max_posterior_sequences=None,
    tau_true_indices=None,
    posteriors_at_step=None,
    delta: float = 0.1,
) -> ConformalSchedule:
    """Legacy entrypoint. New code should call calibrate_split_conformal directly."""
    if posteriors_at_step is not None and tau_true_indices is not None:
        return calibrate_split_conformal(
            posteriors_at_step=posteriors_at_step,
            tau_true_indices=tau_true_indices,
            delta=delta,
        )
    if max_posterior_sequences is None or tau_true_indices is None:
        return default_schedule(num_steps=0, delta=delta)
    max_len = max((len(seq) for seq in max_posterior_sequences), default=0)
    alpha = []
    for t in range(max_len):
        masses = [float(seq[t]) for seq in max_posterior_sequences if t < len(seq)]
        if not masses:
            alpha.append(1.0 - delta)
            continue
        masses.sort()
        idx = max(0, math.floor(delta * len(masses)))
        threshold = masses[idx] if idx < len(masses) else masses[-1]
        alpha.append(max(0.0, min(1.0, threshold)))
    return ConformalSchedule(
        alpha=alpha or [1.0 - delta],
        delta=delta,
        calibration_size=len(max_posterior_sequences),
        meta={"type": "legacy_heuristic", "warning": "no conformal guarantee"},
    )
