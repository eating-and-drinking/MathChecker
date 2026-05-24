"""Tests for joint-likelihood calibration via per-channel temperature scaling."""
from __future__ import annotations

import math
import random

from mathchecker.prism.infer import PrismEvidence, prism_infer
from mathchecker.prism.joint_calibration import (
    CalibrationStep,
    CalibrationTrajectory,
    TemperatureMixer,
    calibration_nll,
    fit_temperatures,
)
from mathchecker.prism.posterior import length_prior


# ---- TemperatureMixer semantics ----

def test_neutral_mixer_is_identity() -> None:
    mixer = TemperatureMixer.neutral()
    lik = [0.1, 0.5, 0.9, 0.2]
    tempered = mixer.temper(likelihood=lik, channel="any_channel")
    for a, b in zip(lik, tempered):
        assert a == b


def test_unspecified_channel_defaults_to_T_one() -> None:
    mixer = TemperatureMixer(temperatures={"stage1": 2.0})
    lik = [0.1, 0.5, 0.9, 0.2]
    out = mixer.temper(likelihood=lik, channel="stage2")
    for a, b in zip(lik, out):
        assert a == b


def test_temperature_two_softens() -> None:
    mixer = TemperatureMixer(temperatures={"x": 2.0})
    lik = [0.01, 0.99]
    out = mixer.temper(likelihood=lik, channel="x")
    assert math.isclose(out[0], 0.1, abs_tol=1e-6)
    assert math.isclose(out[1], math.sqrt(0.99), abs_tol=1e-6)


def test_temperature_half_sharpens() -> None:
    mixer = TemperatureMixer(temperatures={"x": 0.5})
    lik = [0.2, 0.8]
    out = mixer.temper(likelihood=lik, channel="x")
    assert math.isclose(out[0], 0.04, abs_tol=1e-6)
    assert math.isclose(out[1], 0.64, abs_tol=1e-6)


def test_mixer_rejects_non_positive_temperature() -> None:
    for bad in (-1.0, 0.0):
        try:
            TemperatureMixer(temperatures={"x": bad})
        except ValueError:
            continue
        raise AssertionError(f"temperature {bad} should have raised")


def test_mixer_round_trip_via_to_from_dict() -> None:
    mixer = TemperatureMixer(temperatures={"stage1": 1.7, "stage2": 0.8}, meta={"k": "v"})
    payload = mixer.to_dict()
    restored = TemperatureMixer.from_dict(payload)
    assert math.isclose(restored.temperatures["stage1"], 1.7)
    assert math.isclose(restored.temperatures["stage2"], 0.8)


# ---- helper ----

def _make_likelihood(*, num_steps: int, peak_at: int, noise: float, sharpness: float = 0.85) -> tuple[float, ...]:
    base = 1.0 - sharpness
    vals = []
    for k in range(num_steps + 1):
        if k == peak_at:
            vals.append(base + sharpness * (1.0 - noise))
        else:
            vals.append(base + sharpness * noise + 1e-6)
    return tuple(vals)


# ---- fitting ----

def test_fit_temperatures_compensates_for_correlated_detector_errors() -> None:
    """When two channels emit IDENTICAL sharp likelihoods that sometimes
    point at the wrong hypothesis (perfectly correlated errors), independent
    product fusion (T=1 for both) double-counts the evidence. Fitting must
    reduce NLL below the baseline AND pull the total channel weight
    1/T_a + 1/T_b below 2.0 (the no-tempering total)."""
    random.seed(31)
    num_steps = 4
    n_traj = 300
    hyps = [0, 1, 2, 3, num_steps]
    prior = tuple(length_prior(num_steps, p_no_error=0.4))

    trajectories: list[CalibrationTrajectory] = []
    for _ in range(n_traj):
        gold_tau = random.choice(hyps)
        steps = []
        for _t in range(num_steps):
            if random.random() < 0.6:
                peak = gold_tau
            else:
                wrong = [k for k in hyps if k != gold_tau]
                peak = random.choice(wrong)
            shared = _make_likelihood(num_steps=num_steps, peak_at=peak, noise=0.08)
            steps.append(CalibrationStep(likelihoods={"ch_a": shared, "ch_b": shared}))
        trajectories.append(
            CalibrationTrajectory(prior=prior, steps=tuple(steps), gold_tau=gold_tau)
        )

    baseline_nll = calibration_nll(
        trajectories=trajectories, temperatures={"ch_a": 1.0, "ch_b": 1.0},
    )
    fitted = fit_temperatures(
        trajectories=trajectories, channels=["ch_a", "ch_b"], n_rounds=5,
    )
    fitted_nll = calibration_nll(
        trajectories=trajectories, temperatures=fitted.temperatures,
    )

    assert fitted_nll < baseline_nll
    w_total = 1.0 / fitted.temperatures["ch_a"] + 1.0 / fitted.temperatures["ch_b"]
    assert w_total < 2.0, f"total weight {w_total:.3f} did not drop below 2.0; T={fitted.temperatures}"


def test_fit_temperatures_returns_finite_positive_for_single_channel() -> None:
    """Single non-redundant channel: fitted T should at least be finite and
    positive. The double-counting compensation test (above) is the real
    correctness check."""
    random.seed(99)
    num_steps = 3
    n_traj = 150
    prior = tuple(length_prior(num_steps, p_no_error=0.4))
    trajectories: list[CalibrationTrajectory] = []
    for _ in range(n_traj):
        gold_tau = random.choice([0, 1, 2, num_steps])
        steps = []
        for _t in range(num_steps):
            lik = _make_likelihood(num_steps=num_steps, peak_at=gold_tau, noise=0.2)
            steps.append(CalibrationStep(likelihoods={"only_channel": lik}))
        trajectories.append(
            CalibrationTrajectory(prior=prior, steps=tuple(steps), gold_tau=gold_tau)
        )

    fitted = fit_temperatures(
        trajectories=trajectories, channels=["only_channel"], n_rounds=4,
    )
    T = fitted.temperatures["only_channel"]
    assert T > 0.05 and T < 20.0, f"temperature drifted unreasonably: {T}"


def test_calibration_nll_decreases_with_sharper_calibration_data() -> None:
    num_steps = 2
    prior = tuple(length_prior(num_steps, p_no_error=0.4))
    weak = CalibrationTrajectory(
        prior=prior,
        steps=(CalibrationStep(likelihoods={"x": _make_likelihood(num_steps=num_steps, peak_at=1, noise=0.45)}),),
        gold_tau=1,
    )
    strong = CalibrationTrajectory(
        prior=prior,
        steps=(CalibrationStep(likelihoods={"x": _make_likelihood(num_steps=num_steps, peak_at=1, noise=0.05)}),),
        gold_tau=1,
    )
    assert calibration_nll(trajectories=[strong], temperatures={"x": 1.0}) < calibration_nll(trajectories=[weak], temperatures={"x": 1.0})


# ---- end-to-end wiring ----

def test_prism_infer_neutral_mixer_matches_no_mixer() -> None:
    num_steps = 4

    def evidence(t: int) -> PrismEvidence:
        if t == 2:
            return PrismEvidence(
                step_index=t,
                principle_labels=("contradiction-found",) * 3,
                specialist_emissions={"alternative_route_verifier_tool": (0.93, 0.05)},
            )
        return PrismEvidence(step_index=t, principle_labels=("correct-and-aligned",) * 3)

    no_mixer = prism_infer(num_steps=num_steps, evidence_at_step=evidence)
    neutral = prism_infer(num_steps=num_steps, evidence_at_step=evidence, mixer=TemperatureMixer.neutral())
    assert no_mixer.pred_first_mistake_index == neutral.pred_first_mistake_index


def test_prism_infer_soft_mixer_dampens_step2_posterior() -> None:
    """Compare snapshot at the SAME step under neutral vs soft mixer, with
    early stopping DISABLED so both runs traverse all steps."""
    num_steps = 4

    def evidence(t: int) -> PrismEvidence:
        if t == 2:
            return PrismEvidence(
                step_index=t,
                principle_labels=("contradiction-found",) * 3,
                specialist_emissions={"alternative_route_verifier_tool": (0.9, 0.05)},
            )
        return PrismEvidence(step_index=t, principle_labels=("correct-and-aligned",) * 3)

    neutral = prism_infer(num_steps=num_steps, evidence_at_step=evidence, early_stop=False)
    soft = prism_infer(
        num_steps=num_steps,
        evidence_at_step=evidence,
        early_stop=False,
        mixer=TemperatureMixer(temperatures={
            "stage1": 3.0,
            "stage2": 3.0,
            "alternative_route_verifier_tool": 3.0,
            "equivalence_substitution_verifier_tool": 3.0,
            "condition_obligation_verifier_tool": 3.0,
        }),
    )
    n2 = neutral.step_traces[2].posterior_snapshot
    s2 = soft.step_traces[2].posterior_snapshot
    assert max(s2) < max(n2), f"soft step-2 peak {max(s2):.4f} >= neutral {max(n2):.4f}"
