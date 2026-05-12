from __future__ import annotations

from mathchecker.core.models import Stage2Parse
from mathchecker.pipeline.predictor import PedCoTPredictor
from mathchecker.pipeline.tools import (
    alternative_route_verifier_tool,
    condition_obligation_verifier_tool,
    equivalence_substitution_verifier_tool,
)


def test_alternative_route_verifier_accepts_valid_rewrite_path() -> None:
    result = alternative_route_verifier_tool(
        {
            "question": "Compute 12 + 18.",
            "previous_steps": ["We need to add 12 and 18."],
            "current_step": "Let us rewrite 18 as 20 - 2, which is an equivalent path.",
        }
    )

    assert result["hard_contradiction"] is False
    assert result["valid_alternative"] is True


def test_equivalence_substitution_verifier_detects_false_numeric_relation() -> None:
    result = equivalence_substitution_verifier_tool(
        {
            "question": "Check the next arithmetic step.",
            "previous_steps": ["We need to simplify the expression."],
            "current_step": "Rewrite the expression as 2 + 2 = 5.",
        }
    )

    assert result["hard_contradiction"] is True


def test_condition_obligation_verifier_flags_empty_answer_conflict() -> None:
    result = condition_obligation_verifier_tool(
        {
            "question": "What is 7 + 5?",
            "previous_steps": ["We should compute the final value."],
            "current_step": "So the answer is no answer.",
        }
    )

    assert result["hard_contradiction"] is True


def test_stage2_specialist_adjustment_downgrades_reference_only_contradiction() -> None:
    original = Stage2Parse(
        mathematical_concepts_label="contradiction-found",
        key_analyses_label="correct-and-aligned",
        calculations_label="reasonable-but-incomplete",
        success=True,
    )
    tool_trace = [
        {
            "tool_name": "alternative_route_verifier_tool",
            "result": {
                "status": "alternative_route_verified",
                "verification_type": "alternative_route",
                "valid_alternative": True,
                "hard_contradiction": False,
                "preferred_dimension": "key_analyses",
                "evidence": ["Alternative valid route detected."],
            },
        }
    ]

    adjusted, status = PedCoTPredictor._apply_stage2_specialist_adjustment(original, tool_trace)

    assert adjusted.mathematical_concepts_label == "reasonable-but-incomplete"
    assert status["stage2_specialist_adjustment_applied"] is True


def test_stage2_specialist_adjustment_upgrades_hard_tool_evidence() -> None:
    original = Stage2Parse(
        mathematical_concepts_label="correct-and-aligned",
        key_analyses_label="correct-and-aligned",
        calculations_label="correct-and-aligned",
        success=True,
    )
    tool_trace = [
        {
            "tool_name": "equivalence_substitution_verifier_tool",
            "result": {
                "status": "hard_contradiction",
                "verification_type": "equivalence_substitution",
                "hard_contradiction": True,
                "preferred_dimension": "calculations",
                "evidence": ["False numeric relation."],
            },
        }
    ]

    adjusted, status = PedCoTPredictor._apply_stage2_specialist_adjustment(original, tool_trace)

    assert adjusted.calculations_label == "contradiction-found"
    assert status["stage2_specialist_adjustment_applied"] is True
