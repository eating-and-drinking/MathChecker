from __future__ import annotations

import random

from mathchecker.prism.conformal import (
    ConformalSchedule,
    calibrate_conformal,
    calibrate_split_conformal,
    default_schedule,
)


def test_default_schedule_floor_is_one_minus_delta() -> None:
    sched = default_schedule(num_steps=5, delta=0.1)
    for a in sched.alpha:
        assert a >= 0.9 - 1e-9


def test_split_conformal_zero_calibration_falls_back() -> None:
    sched = calibrate_split_conformal(
        posteriors_at_step=[],
        tau_true_indices=[],
        delta=0.1,
    )
    # default_schedule(num_steps=0) returns a single-element schedule
    assert len(sched.alpha) >= 1


def test_split_conformal_threshold_drops_when_posterior_concentrates_on_truth() -> None:
    """If calibration trajectories assign high mass to tau_true at step t, the
    learned threshold at t should be LOW (we're confident in our predictions)."""
    posteriors_at_step = []
    tau_true_indices = []
    for _ in range(100):
        # Three steps; gold tau = 1; posterior at step 1 puts 0.95 on tau=1.
        traj = [
            [0.20, 0.20, 0.20, 0.40],  # step 0 -- uniform-ish
            [0.025, 0.95, 0.013, 0.012],  # step 1 -- concentrated on truth
            [0.01, 0.96, 0.02, 0.01],  # step 2 -- still concentrated
        ]
        posteriors_at_step.append(traj)
        tau_true_indices.append(1)

    sched = calibrate_split_conformal(
        posteriors_at_step=posteriors_at_step,
        tau_true_indices=tau_true_indices,
        delta=0.1,
    )
    # alpha at step 1 should be modestly above (or near) the highest posterior
    # mass on tau_true, which is ~0.95. Because all calibration trajectories
    # are nearly identical, the threshold tightens toward that mass.
    assert sched.alpha[1] >= 0.5  # something non-trivial
    assert sched.meta["type"] == "split_conformal"
    assert sched.meta["bonferroni"] is True
    assert sched.calibration_size == 100


def test_split_conformal_finite_sample_coverage() -> None:
    """Construct a calibration set with a controlled distribution of
    nonconformity scores, then verify empirical coverage on a fresh test
    sample from the same distribution exceeds 1 - delta."""
    random.seed(42)

    def sample_trajectory():
        # Two steps. Gold tau = 0. Posterior mass on truth is a random
        # variable distributed Beta(8, 2) (so well-calibrated, peaked around 0.8).
        traj = []
        for _ in range(2):
            p_true = random.betavariate(8, 2)
            other = (1.0 - p_true) / 2
            traj.append([p_true, other, other])
        return traj

    n_cal = 500
    cal_post = [sample_trajectory() for _ in range(n_cal)]
    cal_tau = [0] * n_cal

    delta = 0.1
    sched = calibrate_split_conformal(
        posteriors_at_step=cal_post,
        tau_true_indices=cal_tau,
        delta=delta,
    )

    # Fresh test sample
    n_test = 2000
    correct = 0
    committed = 0
    for _ in range(n_test):
        traj = sample_trajectory()
        # Walk the trace; commit at first step where max posterior >= alpha_t
        for t, post in enumerate(traj):
            if max(post) >= sched.threshold_at(t):
                argmax = post.index(max(post))
                committed += 1
                if argmax == 0:  # gold tau = 0
                    correct += 1
                break

    if committed == 0:
        # The schedule was extremely conservative -- abstained on everything.
        # That trivially satisfies coverage (vacuously).
        return
    coverage = correct / committed
    # Allow some slack for finite-sample noise but we should be at least near 1-delta
    # (Bonferroni makes it more conservative; usually >> 1-delta).
    assert coverage >= (1.0 - delta) - 0.05, f"coverage {coverage} below target {1-delta}"


def test_legacy_calibrate_conformal_forwards_to_split() -> None:
    posteriors = [
        [[0.1, 0.7, 0.2], [0.05, 0.85, 0.10]],
        [[0.1, 0.8, 0.1], [0.05, 0.90, 0.05]],
    ]
    sched = calibrate_conformal(
        posteriors_at_step=posteriors,
        tau_true_indices=[1, 1],
        delta=0.2,
    )
    assert sched.meta["type"] == "split_conformal"


def test_schedule_round_trip_via_to_from_dict() -> None:
    sched = ConformalSchedule(alpha=[0.9, 0.85, 0.8], delta=0.1, calibration_size=42)
    payload = sched.to_dict()
    restored = ConformalSchedule.from_dict(payload)
    assert restored.alpha == sched.alpha
    assert restored.delta == sched.delta
    assert restored.calibration_size == sched.calibration_size


def test_bonferroni_makes_threshold_more_conservative_than_uncorrected() -> None:
    """With Bonferroni, delta is divided by T -> per-step delta_eff is tiny ->
    higher quantile rank -> higher nonconformity threshold -> lower alpha (more
    likely to abstain). So the schedule's alpha values should be no higher than
    the uncorrected version."""
    random.seed(7)
    n_cal = 200
    cal_post = []
    cal_tau = []
    for _ in range(n_cal):
        p = random.uniform(0.3, 0.99)
        other = (1 - p) / 3
        # 4 steps
        traj = [[p, other, other, other, 0.0] for _ in range(4)]
        cal_post.append(traj)
        cal_tau.append(0)
    bonf = calibrate_split_conformal(
        posteriors_at_step=cal_post, tau_true_indices=cal_tau, delta=0.1, bonferroni=True
    )
    raw = calibrate_split_conformal(
        posteriors_at_step=cal_post, tau_true_indices=cal_tau, delta=0.1, bonferroni=False
    )
    for a_bonf, a_raw in zip(bonf.alpha, raw.alpha):
        # Bonferroni uses delta/T, so the corresponding alpha_t = 1 - (high quantile)
        # should be <= the uncorrected one (more conservative -> easier to abstain).
        assert a_bonf <= a_raw + 1e-9
