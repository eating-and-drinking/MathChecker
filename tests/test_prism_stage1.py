from __future__ import annotations

from mathchecker.prism.conformal import default_schedule
from mathchecker.prism.infer import PrismEvidence, prism_infer
from mathchecker.prism.stage1_consistency import extract_stage1_inconsistency


# ---- extractor tests ----

def test_stage1_extractor_detects_numeric_mismatch() -> None:
    result = extract_stage1_inconsistency(
        stage1_calculations="Expected: 5 * 20 = 100, sum of the odd integers.",
        current_step="Multiply: 5 × 20 = 110.",
    )
    assert result.inconsistency_strength >= 0.8
    assert result.source == "numeric_mismatch"
    assert len(result.contradictions) >= 1


def test_stage1_extractor_returns_zero_on_consistent_arithmetic() -> None:
    result = extract_stage1_inconsistency(
        stage1_calculations="The product 5 * 20 = 100 gives the answer.",
        current_step="Compute: 5 * 20 = 100.",
    )
    assert result.inconsistency_strength == 0.0
    assert result.source == "no_signal"


def test_stage1_extractor_returns_zero_when_nothing_parseable() -> None:
    result = extract_stage1_inconsistency(
        stage1_calculations="Use a clever identity for arithmetic sequences.",
        current_step="Therefore the answer is 100.",
    )
    assert result.inconsistency_strength == 0.0


def test_stage1_extractor_flags_self_inconsistent_step_even_without_stage1() -> None:
    """Even with no concrete stage1 prediction, an arithmetic-self-contradictory
    step should be caught (5 * 20 != 110)."""
    result = extract_stage1_inconsistency(
        stage1_calculations="Apply the pairing identity.",
        current_step="Multiply: 5 * 20 = 110.",
    )
    assert result.inconsistency_strength >= 0.8


def test_stage1_extractor_missing_inputs_is_safe() -> None:
    assert extract_stage1_inconsistency(
        stage1_calculations=None, current_step=None
    ).inconsistency_strength == 0.0
    assert extract_stage1_inconsistency(
        stage1_calculations="anything", current_step=""
    ).inconsistency_strength == 0.0


def test_stage1_extractor_detects_inequality_violation() -> None:
    """Stage1 predicts answer >= 0; step asserts a negative answer."""
    r = extract_stage1_inconsistency(
        stage1_calculations="The result must satisfy x >= 0.",
        current_step="So the answer is -5.",
    )
    assert r.inconsistency_strength >= 0.8
    assert any("expected value" in c for c in r.contradictions)


def test_stage1_extractor_inequality_consistent_returns_no_signal() -> None:
    r = extract_stage1_inconsistency(
        stage1_calculations="The result must satisfy x >= 0.",
        current_step="So the answer is 7.",
    )
    assert r.inconsistency_strength == 0.0


def test_stage1_extractor_detects_domain_violation_integer() -> None:
    r = extract_stage1_inconsistency(
        stage1_calculations="The answer must be a positive integer.",
        current_step="Therefore the answer is 2.5.",
    )
    assert r.inconsistency_strength >= 0.8


def test_stage1_extractor_detects_domain_violation_negative() -> None:
    r = extract_stage1_inconsistency(
        stage1_calculations="The result is non-negative.",
        current_step="So the answer is -3.",
    )
    assert r.inconsistency_strength >= 0.8


def test_stage1_extractor_domain_consistent_is_no_signal() -> None:
    r = extract_stage1_inconsistency(
        stage1_calculations="The answer should be an integer.",
        current_step="Therefore the answer is 17.",
    )
    assert r.inconsistency_strength == 0.0


# ---- inference-loop integration ----

def _clean_step(t: int) -> PrismEvidence:
    return PrismEvidence(
        step_index=t,
        principle_labels=("correct-and-aligned",) * 3,
    )


def test_prism_infer_stage1_inconsistency_pulls_posterior_to_step() -> None:
    """When stage1 reports a numeric mismatch at step k but stage2 labels are
    clean and no specialist is invoked, the stage1 channel alone should still
    move the posterior toward tau = k."""
    num_steps = 4

    def evidence(t: int) -> PrismEvidence:
        if t == 2:
            return PrismEvidence(
                step_index=t,
                principle_labels=("correct-and-aligned",) * 3,
                stage1_inconsistency=0.92,
            )
        return _clean_step(t)

    result = prism_infer(
        num_steps=num_steps,
        evidence_at_step=evidence,
        schedule=default_schedule(num_steps=num_steps, delta=0.1),
    )
    assert result.pred_first_mistake_index == 2


def test_prism_infer_stage1_no_signal_is_a_noop() -> None:
    """An evidence bundle with stage1_inconsistency=0 must behave identically to
    one that omits the field -- i.e. the channel is silent."""
    num_steps = 4

    def evidence_clean(t):
        return PrismEvidence(
            step_index=t,
            principle_labels=("correct-and-aligned",) * 3,
            stage1_inconsistency=0.0,
        )

    result = prism_infer(
        num_steps=num_steps,
        evidence_at_step=evidence_clean,
        schedule=default_schedule(num_steps=num_steps, delta=0.1),
    )
    assert result.pred_first_mistake_index is None


def test_prism_infer_valid_alternative_suppresses_stage1_signal() -> None:
    """If specialist reports valid_alternative=high, the stage1 numeric mismatch
    should be SUPPRESSED."""
    num_steps = 4

    def evidence(t):
        if t == 2:
            return PrismEvidence(
                step_index=t,
                principle_labels=("correct-and-aligned",) * 3,
                stage1_inconsistency=0.90,
                specialist_emissions={
                    "alternative_route_verifier_tool": (0.05, 0.95),
                    "equivalence_substitution_verifier_tool": (0.05, 0.95),
                },
            )
        return _clean_step(t)

    result = prism_infer(
        num_steps=num_steps,
        evidence_at_step=evidence,
        schedule=default_schedule(num_steps=num_steps, delta=0.1),
    )
    assert result.pred_first_mistake_index != 2
