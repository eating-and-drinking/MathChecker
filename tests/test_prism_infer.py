from __future__ import annotations

from mathchecker.prism.conformal import default_schedule
from mathchecker.prism.infer import PrismEvidence, prism_infer


def _clean_step(t: int) -> PrismEvidence:
    return PrismEvidence(
        step_index=t,
        principle_labels=("correct-and-aligned", "correct-and-aligned", "correct-and-aligned"),
    )


def _contradiction_step(t: int) -> PrismEvidence:
    return PrismEvidence(
        step_index=t,
        principle_labels=("contradiction-found", "correct-and-aligned", "contradiction-found"),
    )


def test_prism_infer_clean_trace_picks_no_error() -> None:
    num_steps = 4
    result = prism_infer(
        num_steps=num_steps,
        evidence_at_step=lambda t: _clean_step(t),
        schedule=default_schedule(num_steps=num_steps, delta=0.1),
    )
    assert result.pred_first_mistake_index is None
    assert result.final_posterior.p_no_error() > 0.5


def test_prism_infer_finds_first_contradiction() -> None:
    num_steps = 5

    def evidence(t: int) -> PrismEvidence:
        if t == 2:
            return _contradiction_step(t)
        return _clean_step(t)

    result = prism_infer(
        num_steps=num_steps,
        evidence_at_step=evidence,
        schedule=default_schedule(num_steps=num_steps, delta=0.1),
    )
    assert result.pred_first_mistake_index == 2
    assert result.committed is True


def test_prism_infer_specialist_evidence_overrides_clean_labels() -> None:
    """When specialist tools report a hard conflict at step k, posterior moves
    onto tau = k even if the stage2 labels look clean.

    This is the PRISM equivalent of the deterministic adjustment 'upgrade
    contradiction-found from hard tool evidence' rule -- now expressed
    as a Bayesian update."""
    num_steps = 4

    def evidence(t: int) -> PrismEvidence:
        if t == 1:
            return PrismEvidence(
                step_index=t,
                principle_labels=("correct-and-aligned", "correct-and-aligned", "correct-and-aligned"),
                specialist_emissions={
                    "alternative_route_verifier_tool": (0.95, 0.05),
                    "equivalence_substitution_verifier_tool": (0.95, 0.05),
                },
            )
        return _clean_step(t)

    result = prism_infer(
        num_steps=num_steps,
        evidence_at_step=evidence,
        schedule=default_schedule(num_steps=num_steps, delta=0.1),
    )
    # Strong specialist evidence at step 1 should pull MAP to step 1.
    assert result.pred_first_mistake_index == 1


def test_prism_infer_specialist_valid_alternative_suppresses_contradiction() -> None:
    """When labels say contradiction but specialists report 'valid alternative',
    the contradiction should be downgraded -- posterior shouldn't commit to
    tau = step. This is the PRISM equivalent of the legacy
    'downgrade_contradiction_from_valid_alternative' rule."""
    num_steps = 4

    def evidence(t: int) -> PrismEvidence:
        if t == 1:
            return PrismEvidence(
                step_index=t,
                principle_labels=(
                    "contradiction-found",
                    "correct-and-aligned",
                    "correct-and-aligned",
                ),
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
    # Step 1 should NOT win because valid-alternative suppresses the conflict.
    assert result.pred_first_mistake_index != 1


def test_prism_infer_records_specialist_calls_in_traces() -> None:
    num_steps = 3

    def evidence(t: int) -> PrismEvidence:
        return PrismEvidence(
            step_index=t,
            principle_labels=("correct-and-aligned", "correct-and-aligned", "correct-and-aligned"),
            specialist_emissions={"alternative_route_verifier_tool": (0.10, 0.20)},
        )

    result = prism_infer(
        num_steps=num_steps,
        evidence_at_step=evidence,
        schedule=default_schedule(num_steps=num_steps, delta=0.1),
    )
    assert len(result.step_traces) >= 1
    first = result.step_traces[0]
    assert "alternative_route_verifier_tool" in first.eig_scores


def test_prism_infer_zero_steps() -> None:
    result = prism_infer(
        num_steps=0,
        evidence_at_step=lambda t: _clean_step(t),
        schedule=default_schedule(num_steps=0, delta=0.1),
    )
    assert result.pred_first_mistake_index is None
