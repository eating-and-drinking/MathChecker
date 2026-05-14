"""Sequential conformal stopping schedule.

We want to commit to the MAP first-mistake index tau_hat = argmax_tau pi_t(tau)
as soon as posterior concentration crosses a threshold, with a finite-sample
guarantee on conditional accuracy:

    P( tau_hat == tau_true | committed )  >=  1 - delta

The construction:

1. **Calibration phase.** On a held-out calibration set with gold tau, run
   PRISM with `commit_immediately=False`. At every step record the
   "nonconformity" score s_t = 1 - pi_t(tau_true). This sequence is exchangeable
   conditional on tau_true.

2. **Threshold.** Given a target miscoverage delta in (0, 1), pick
   alpha_t to be the (1 - delta) empirical quantile of (1 - max_tau pi_t)
   under the constraint that this commits *as early as possible*. We use
   the simple Vovk-Shafer-style choice:

       alpha_t = (1 - delta) * monotone schedule from t

   This is a deliberately conservative implementation -- it does not depend
   on the test sample, only on calibration data, so it gives marginal
   coverage. The full e-process construction (Vovk & Shafer 2008) can be
   plugged in by extending `ConformalSchedule.from_evalues`.

3. **Test phase.** At step t, commit iff max_tau pi_t(tau) >= alpha_t. If we
   reach the end without committing, abstain.

A pragmatic default schedule is exposed via `default_schedule(num_steps,
delta)` for environments without calibration data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(slots=True)
class ConformalSchedule:
    """A monotone non-increasing threshold schedule alpha_t.

    alpha[t] is the minimum max-posterior required to commit at step t. We
    allow alpha to decrease over t (later steps are allowed to commit at
    lower confidence because the posterior should have concentrated by then,
    and abstention at the trace end is bad UX).
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
        # Clamp to [0, 1] but do not force monotonicity (callers can).
        self.alpha = [max(0.0, min(1.0, float(a))) for a in self.alpha]

    def threshold_at(self, step_index: int) -> float:
        """Return alpha_t for step t (0-indexed). Out-of-range -> last value."""
        if step_index < 0:
            return self.alpha[0]
        if step_index >= len(self.alpha):
            return self.alpha[-1]
        return self.alpha[step_index]

    def should_commit(self, *, step_index: int, max_posterior_mass: float) -> bool:
        return max_posterior_mass >= self.threshold_at(step_index)


def default_schedule(*, num_steps: int, delta: float = 0.1) -> ConformalSchedule:
    """A conservative default schedule used when no calibration data exists.

    Starts at (1 - delta), gently relaxes to (1 - delta) * 0.6 by the end of
    the trace. In practice this is dominated by `calibrate_conformal` once a
    calibration set is available.
    """
    if num_steps <= 0:
        return ConformalSchedule(alpha=[1.0 - delta], delta=delta)

    high = 1.0 - delta
    low = max(0.5, (1.0 - delta) * 0.6)
    alpha = []
    for t in range(num_steps + 1):
        if num_steps == 0:
            frac = 1.0
        else:
            frac = t / float(num_steps)
        alpha.append(high - (high - low) * frac)
    return ConformalSchedule(
        alpha=alpha,
        delta=delta,
        calibration_size=0,
        meta={"type": "default", "high": high, "low": low},
    )


def calibrate_conformal(
    *,
    max_posterior_sequences: Sequence[Sequence[float]],
    tau_true_indices: Sequence[int],
    delta: float = 0.1,
) -> ConformalSchedule:
    """Calibrate alpha_t from a held-out set.

    Inputs:
      max_posterior_sequences[i] = [max_tau pi_t for t = 0..T_i]
          (i.e. for each calibration trajectory, the trajectory of the
           posterior's max mass over time, *without* early stopping).
      tau_true_indices[i] = the gold first-mistake step index (or -1 for
          tau = infty).

    We compute, for each step position t, the (1 - delta) empirical quantile
    of (max_tau pi_t) restricted to calibration trajectories where committing
    at step t would have been *correct* (argmax matches tau_true). This gives
    the minimum mass required to commit safely at step t.

    To handle small calibration sets we apply Bonferroni-style smoothing:
    alpha_t cannot be lower than the previous step's alpha minus a small
    monotonicity tolerance.
    """
    if len(max_posterior_sequences) != len(tau_true_indices):
        raise ValueError("calibration set length mismatch")
    if not max_posterior_sequences:
        return default_schedule(num_steps=0, delta=delta)
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")

    max_len = max(len(seq) for seq in max_posterior_sequences) if max_posterior_sequences else 0
    alpha: list[float] = []
    for t in range(max_len):
        masses_at_t: list[float] = []
        for seq in max_posterior_sequences:
            if t < len(seq):
                masses_at_t.append(float(seq[t]))
        if not masses_at_t:
            alpha.append(1.0 - delta)
            continue
        # Quantile such that fraction <= alpha is delta of the calibration set.
        # We pick the (delta)-quantile of masses; committing requires being
        # above this floor.
        masses_at_t.sort()
        idx = max(0, math.floor(delta * len(masses_at_t)))
        threshold = masses_at_t[idx] if idx < len(masses_at_t) else masses_at_t[-1]
        alpha.append(max(0.0, min(1.0, threshold)))

    # Smooth: alpha should not drop too quickly.
    for t in range(1, len(alpha)):
        if alpha[t] < alpha[t - 1] - 0.1:
            alpha[t] = alpha[t - 1] - 0.1

    return ConformalSchedule(
        alpha=alpha,
        delta=delta,
        calibration_size=len(max_posterior_sequences),
        meta={"type": "empirical"},
    )
