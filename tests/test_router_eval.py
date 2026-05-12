from __future__ import annotations

from pathlib import Path

from pedcot.core.constants import DATASET_BIG_BENCH_MISTAKE
from pedcot.core.models import TraceExample
from pedcot.data.stores import StageCacheStore
from pedcot.pipeline.predictor import PedCoTPredictor
from pedcot.pipeline.router import build_route_decision_from_scores
from pedcot.pipeline.router_dataset import build_router_export_row
from pedcot.pipeline.router_eval import (
    build_route_ablation_summary,
    calibrate_per_label_thresholds,
    evaluate_router_predictions,
)


class FakeReviewClient:
    def generate(
        self,
        *,
        model: str,
        prompt: str,
        metadata: dict | None = None,
        tools: list[dict] | None = None,
        tool_handlers: dict | None = None,
        required_tool_names: list[str] | None = None,
    ) -> tuple[str, dict]:
        del model
        del prompt
        del tools
        del tool_handlers

        assert metadata is not None
        stage = metadata["stage"]
        if stage == "stage1":
            return (
                """1. Mathematical Concepts to Apply:
Use substitution carefully.
2. Key Analyses for the Next Step:
Substitute the known variable value into the expression.
3. Mathematical Expressions to Compute:
x = 3, so evaluate x + 2.
""",
                {"tool_trace": [], "tool_errors": []},
            )

        if stage == "stage2_step_type":
            return (
                """{
  "step_type": "substitution",
  "reasoning": "The current step plugs a known value into an expression.",
  "risk_flags": ["substitution_risk", "equivalence_risk"],
  "confidence": 0.92
}""",
                {},
            )

        if stage == "stage2":
            trace = []
            for tool_name in required_tool_names or []:
                result = {
                    "status": "ok",
                    "verification_type": "generic",
                    "preferred_dimension": "calculations",
                    "evidence": [],
                }
                if tool_name == "alternative_route_verifier_tool":
                    result = {
                        "status": "alternative_route_verified",
                        "verification_type": "alternative_route",
                        "valid_alternative": True,
                        "hard_contradiction": False,
                        "preferred_dimension": "key_analyses",
                        "evidence": ["The step follows a different but valid route."],
                    }
                elif tool_name == "equivalence_substitution_verifier_tool":
                    result = {
                        "status": "equivalent",
                        "verification_type": "equivalence_substitution",
                        "valid_equivalent_transformation": True,
                        "hard_contradiction": False,
                        "preferred_dimension": "calculations",
                        "evidence": ["The substitution is locally equivalent."],
                    }
                elif tool_name == "condition_obligation_verifier_tool":
                    result = {
                        "status": "condition_checked",
                        "verification_type": "condition_obligation",
                        "hard_contradiction": False,
                        "preferred_dimension": "key_analyses",
                        "evidence": ["No condition conflict detected."],
                    }
                trace.append({"tool_name": tool_name, "result": result})

            return (
                """1. Mathematical Concepts to Apply:
The step differs from the expected reference path.
Label: contradiction-found
2. Key Analyses for the Next Step:
The substitution is treated as conflicting with the reference derivation.
Label: contradiction-found
3. Mathematical Expressions to Compute:
The rewritten expression appears different from the reference form.
Label: contradiction-found
""",
                {"tool_trace": trace, "tool_errors": []},
            )

        if stage == "stage2_review":
            return (
                """1. Mathematical Concepts to Apply:
The step still appears to conflict with the expected reference path.
Label: contradiction-found
2. Key Analyses for the Next Step:
The reasoning is judged as contradictory to the reference solution.
Label: contradiction-found
3. Mathematical Expressions to Compute:
The transformed expression is marked as conflicting.
Label: contradiction-found
""",
                {},
            )

        if stage == "stage2_specialist_review":
            return (
                """1. Mathematical Concepts to Apply:
The step uses a valid substitution-based alternative route.
Label: reasonable-but-incomplete
2. Key Analyses for the Next Step:
The reasoning differs from the reference path but remains mathematically compatible.
Label: reasonable-but-incomplete
3. Mathematical Expressions to Compute:
The local transformation is supported by specialist evidence.
Label: reasonable-but-incomplete
""",
                {},
            )

        raise AssertionError(f"Unexpected stage: {stage}")


def test_router_export_row_supports_expected_gain_targets(tmp_path: Path) -> None:
    cache_store = StageCacheStore(tmp_path / "stage-cache.jsonl")
    predictor = PedCoTPredictor(
        client=FakeReviewClient(),
        cache_store=cache_store,
        stage1_tools_mode="none",
        stage2_tools_mode="triad",
    )
    example = TraceExample(
        example_id="router-benefit-1",
        dataset=DATASET_BIG_BENCH_MISTAKE,
        question="If x = 3, compute x + 2.",
        steps=["Substitute x = 3 into x + 2 and simplify."],
        gold_answer="5",
        model_answer="x + 2 = 3 + 2",
        gold_first_mistake_index=None,
    )

    prediction = predictor.predict_trace(example, model="fake-model")
    row = build_router_export_row(example, prediction.steps[0], label_strategy="expected-gain")

    assert row is not None
    assert row["labels"] == row["expected_gain_targets"]
    assert "Substitute the known variable value into the expression." in (row["stage1_key_analyses"] or "")
    assert row["expected_gain_targets"]["use_alternative_route_verifier_tool"] > 0.75
    assert row["expected_gain_targets"]["trigger_specialist_review"] > 0.75
    assert row["expected_gain_targets"]["use_condition_obligation_verifier_tool"] < 0.5
    assert row["supervision"]["expected_gain_learning_version"] == "counterfactual-v1"
    assert row["supervision"]["expected_gain_breakdown"]["use_alternative_route_verifier_tool"]["used"] is True


def test_router_export_row_marks_missed_counterfactual_opportunity_for_unused_specialist() -> None:
    example = TraceExample(
        example_id="router-missed-1",
        dataset=DATASET_BIG_BENCH_MISTAKE,
        question="Consider cases for x > 0 and x <= 0.",
        steps=["Assume x > 0, so the expression must be positive."],
        gold_answer="",
        model_answer="",
        gold_first_mistake_index=0,
    )
    step = {
        "step_index": 0,
        "stage1_parse": {
            "mathematical_concepts": "Reason about cases.",
            "key_analyses": "Track whether each branch satisfies the required condition.",
            "calculations": "",
        },
        "stage2_step_type": "condition_case",
        "stage2_step_type_meta": {
            "risk_flags": ["condition_risk", "branch_risk"],
            "specialists": [
                "alternative_route_verifier_tool",
                "condition_obligation_verifier_tool",
            ],
        },
        "stage2_tool_trace": [
            {
                "tool_name": "alternative_route_verifier_tool",
                "result": {
                    "status": "ok",
                    "valid_alternative": False,
                    "hard_contradiction": False,
                },
            }
        ],
        "stage2_original_parse": {
            "mathematical_concepts_label": "correct-and-aligned",
            "key_analyses_label": "correct-and-aligned",
            "calculations_label": "correct-and-aligned",
            "principle_labels": {
                "mathematical_concepts": "correct-and-aligned",
                "key_analyses": "correct-and-aligned",
                "calculations": "correct-and-aligned",
            },
        },
        "stage2_parse": {
            "mathematical_concepts_label": "correct-and-aligned",
            "key_analyses_label": "correct-and-aligned",
            "calculations_label": "correct-and-aligned",
            "principle_labels": {
                "mathematical_concepts": "correct-and-aligned",
                "key_analyses": "correct-and-aligned",
                "calculations": "correct-and-aligned",
            },
        },
        "stage2_route_meta": {
            "selected_specialists": ["alternative_route_verifier_tool"],
            "trigger_specialist_review": False,
        },
        "parse_status": {},
    }

    row = build_router_export_row(example, step, label_strategy="expected-gain")

    assert row is not None
    assert row["expected_gain_targets"]["use_condition_obligation_verifier_tool"] > 0.55
    assert (
        row["supervision"]["expected_gain_breakdown"]["use_condition_obligation_verifier_tool"]["counterfactual_gain"]
        > 0.0
    )
    assert "heuristic_route_disagrees" in row["supervision"]["expected_gain_breakdown"][
        "use_condition_obligation_verifier_tool"
    ]["reasons"]


def test_route_decision_respects_per_label_thresholds() -> None:
    decision = build_route_decision_from_scores(
        {
            "use_alternative_route_verifier_tool": 0.68,
            "use_equivalence_substitution_verifier_tool": 0.61,
            "use_condition_obligation_verifier_tool": 0.18,
            "trigger_specialist_review": 0.74,
        },
        confidence_threshold=0.55,
        per_label_thresholds={
            "use_alternative_route_verifier_tool": 0.7,
            "use_equivalence_substitution_verifier_tool": 0.6,
            "trigger_specialist_review": 0.72,
        },
        source="test",
        model_name=None,
    )

    assert decision.selected_specialists == ("equivalence_substitution_verifier_tool",)
    assert decision.trigger_specialist_review is True


def test_router_eval_supports_threshold_search_and_ablation() -> None:
    rows = [
        {
            "dataset": DATASET_BIG_BENCH_MISTAKE,
            "question": "q1",
            "previous_steps": [],
            "current_step": "s1",
            "heuristic_step_type": "substitution",
            "heuristic_risk_flags": [],
            "heuristic_specialists": ["alternative_route_verifier_tool"],
            "expected_gain_targets": {
                "use_alternative_route_verifier_tool": 0.92,
                "use_equivalence_substitution_verifier_tool": 0.18,
                "use_condition_obligation_verifier_tool": 0.08,
                "trigger_specialist_review": 0.88,
            },
        },
        {
            "dataset": DATASET_BIG_BENCH_MISTAKE,
            "question": "q2",
            "previous_steps": [],
            "current_step": "s2",
            "heuristic_step_type": "algebraic_transformation",
            "heuristic_risk_flags": [],
            "heuristic_specialists": ["equivalence_substitution_verifier_tool"],
            "expected_gain_targets": {
                "use_alternative_route_verifier_tool": 0.12,
                "use_equivalence_substitution_verifier_tool": 0.91,
                "use_condition_obligation_verifier_tool": 0.14,
                "trigger_specialist_review": 0.41,
            },
        },
    ]
    score_rows = [
        {
            "use_alternative_route_verifier_tool": 0.81,
            "use_equivalence_substitution_verifier_tool": 0.19,
            "use_condition_obligation_verifier_tool": 0.09,
            "trigger_specialist_review": 0.76,
        },
        {
            "use_alternative_route_verifier_tool": 0.16,
            "use_equivalence_substitution_verifier_tool": 0.83,
            "use_condition_obligation_verifier_tool": 0.22,
            "trigger_specialist_review": 0.36,
        },
    ]

    thresholds, report = calibrate_per_label_thresholds(
        rows,
        score_rows,
        target_field="expected_gain_targets",
        objective="utility",
    )
    predictions = [
        {
            "labels": {
                label: int(score_row[label] >= thresholds[label])
                for label in thresholds
            },
            "scores": score_row,
            "confidence": 0.8,
            "fallback_used": False,
        }
        for score_row in score_rows
    ]
    metrics = evaluate_router_predictions(
        rows,
        predictions,
        target_field="expected_gain_targets",
    )
    ablation = build_route_ablation_summary(
        rows,
        predictions,
        target_field="expected_gain_targets",
    )

    assert set(thresholds) == set(score_rows[0])
    assert report["use_alternative_route_verifier_tool"]["objective"] == "utility"
    assert metrics["micro_f1"] >= 0.75
    assert metrics["avg_policy_utility"] >= 0.75
    assert ablation["use_alternative_route_verifier_tool"]["delta_avg_policy_utility"] <= 0.0
