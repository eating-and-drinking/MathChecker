from __future__ import annotations

from pathlib import Path

from pedcot.core.constants import DATASET_BIG_BENCH_MISTAKE
from pedcot.core.models import TraceExample
from pedcot.data.stores import StageCacheStore
from pedcot.pipeline.predictor import PedCoTPredictor
from pedcot.pipeline.router import RouteDecision, RouterContext
from pedcot.pipeline.router_dataset import build_router_export_row, local_gold_step_label
from pedcot.pipeline.step_classifier import (
    apply_step_type_specialist_route,
    classify_step_type,
    parse_step_type_classifier_response,
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


class InvalidStepTypeClient(FakeReviewClient):
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
        if metadata is not None and metadata.get("stage") == "stage2_step_type":
            return "not-json", {}
        return super().generate(
            model=model,
            prompt=prompt,
            metadata=metadata,
            tools=tools,
            tool_handlers=tool_handlers,
            required_tool_names=required_tool_names,
        )


class FakeLearnedRouter:
    def __init__(self, decision: RouteDecision) -> None:
        self.decision = decision
        self.calls: list[RouterContext] = []

    def route(self, context: RouterContext) -> RouteDecision:
        self.calls.append(context)
        return self.decision


def test_step_classifier_routes_substitution_steps_to_equivalence_specialists() -> None:
    classification = classify_step_type(
        dataset=DATASET_BIG_BENCH_MISTAKE,
        question="If x = 3, compute x + 2.",
        previous_steps=[],
        current_step="Substitute x = 3 into x + 2 and simplify.",
    )

    routed = apply_step_type_specialist_route(
        {"bb_substitution_tool"},
        classification=classification,
        available_tool_names={
            "bb_substitution_tool",
            "alternative_route_verifier_tool",
            "equivalence_substitution_verifier_tool",
            "condition_obligation_verifier_tool",
        },
    )

    assert classification.step_type == "substitution"
    assert "alternative_route_verifier_tool" in routed
    assert "equivalence_substitution_verifier_tool" in routed
    assert "bb_substitution_tool" in routed


def test_predict_trace_persists_specialist_review_layer(tmp_path: Path) -> None:
    cache_store = StageCacheStore(tmp_path / "stage-cache.jsonl")
    predictor = PedCoTPredictor(
        client=FakeReviewClient(),
        cache_store=cache_store,
        stage1_tools_mode="none",
        stage2_tools_mode="triad",
    )
    example = TraceExample(
        example_id="ex-1",
        dataset=DATASET_BIG_BENCH_MISTAKE,
        question="If x = 3, compute x + 2.",
        steps=["Substitute x = 3 into x + 2 and simplify."],
        gold_answer="5",
        model_answer="x + 2 = 3 + 2",
        gold_first_mistake_index=None,
    )

    prediction = predictor.predict_trace(example, model="fake-model")

    assert prediction.completed is True
    assert prediction.pred_first_mistake_index is None
    step = prediction.steps[0]
    assert step["stage2_step_type"] == "substitution"
    assert step["stage2_step_type_meta"]["specialists"] == [
        "alternative_route_verifier_tool",
        "equivalence_substitution_verifier_tool",
    ]
    assert step["stage2_step_type_meta"]["source"] == "hybrid"
    assert step["stage2_step_type_meta"]["llm_used"] is True
    assert step["stage2_step_type_meta"]["llm_success"] is True
    assert step["stage2_review_prompt"] is not None
    assert step["stage2_specialist_review_prompt"] is not None
    assert step["stage2_review_parse"]["key_analyses_label"] == "contradiction-found"
    assert step["stage2_specialist_review_parse"]["key_analyses_label"] == "reasonable-but-incomplete"
    assert step["stage2_parse"]["key_analyses_label"] == "reasonable-but-incomplete"
    assert step["parse_status"]["stage2_specialist_review_enabled"] is True
    assert step["parse_status"]["stage2_specialist_review_applied"] is True
    assert step["parse_status"]["stage2_specialist_adjustment_applied"] is False


def test_step_type_parser_accepts_json_code_fence() -> None:
    parsed = parse_step_type_classifier_response(
        """```json
{
  "step_type": "condition_case",
  "reasoning": "The step introduces an if-branch.",
  "risk_flags": ["condition_risk", "branch_risk"],
  "confidence": 0.8
}
```"""
    )

    assert parsed.success is True
    assert parsed.classification is not None
    assert parsed.classification.step_type == "condition_case"
    assert parsed.classification.source == "llm"


def test_predict_trace_falls_back_to_heuristic_step_type_when_llm_parse_fails(tmp_path: Path) -> None:
    cache_store = StageCacheStore(tmp_path / "stage-cache.jsonl")
    predictor = PedCoTPredictor(
        client=InvalidStepTypeClient(),
        cache_store=cache_store,
        stage1_tools_mode="none",
        stage2_tools_mode="triad",
        stage2_step_type_mode="hybrid",
    )
    example = TraceExample(
        example_id="ex-2",
        dataset=DATASET_BIG_BENCH_MISTAKE,
        question="If x = 3, compute x + 2.",
        steps=["Substitute x = 3 into x + 2 and simplify."],
        gold_answer="5",
        model_answer="x + 2 = 3 + 2",
        gold_first_mistake_index=None,
    )

    prediction = predictor.predict_trace(example, model="fake-model")

    step = prediction.steps[0]
    assert step["stage2_step_type"] == "substitution"
    assert step["stage2_step_type_meta"]["source"] == "heuristic_fallback"
    assert step["stage2_step_type_meta"]["llm_used"] is True
    assert step["stage2_step_type_meta"]["llm_success"] is False
    assert step["stage2_step_type_meta"]["fallback_to_heuristic"] is True


def test_predict_trace_can_use_learned_router_to_override_specialists(tmp_path: Path) -> None:
    cache_store = StageCacheStore(tmp_path / "stage-cache.jsonl")
    learned_router = FakeLearnedRouter(
        RouteDecision(
            selected_specialists=("alternative_route_verifier_tool",),
            trigger_specialist_review=True,
            confidence=0.93,
            source="learned_router",
            model_name="local-qwen3-router",
            candidate_scores={
                "use_alternative_route_verifier_tool": 0.93,
                "use_equivalence_substitution_verifier_tool": 0.12,
                "use_condition_obligation_verifier_tool": 0.08,
                "trigger_specialist_review": 0.89,
            },
        )
    )
    predictor = PedCoTPredictor(
        client=FakeReviewClient(),
        cache_store=cache_store,
        stage1_tools_mode="none",
        stage2_tools_mode="triad",
        stage2_router_mode="learned-hybrid",
        learned_router=learned_router,
    )
    example = TraceExample(
        example_id="ex-3",
        dataset=DATASET_BIG_BENCH_MISTAKE,
        question="If x = 3, compute x + 2.",
        steps=["Substitute x = 3 into x + 2 and simplify."],
        gold_answer="5",
        model_answer="x + 2 = 3 + 2",
        gold_first_mistake_index=None,
    )

    prediction = predictor.predict_trace(example, model="fake-model")

    step = prediction.steps[0]
    called_names = set(step["parse_status"]["stage2_tool_called_names"])
    assert len(learned_router.calls) == 1
    assert step["stage2_route_meta"]["source"] == "learned_router"
    assert step["stage2_route_meta"]["selected_specialists"] == ["alternative_route_verifier_tool"]
    assert "bb_arithmetic_chain_tool" in called_names
    assert "alternative_route_verifier_tool" in called_names
    assert "equivalence_substitution_verifier_tool" not in called_names


def test_predict_trace_falls_back_to_step_type_route_when_learned_router_confidence_is_low(tmp_path: Path) -> None:
    cache_store = StageCacheStore(tmp_path / "stage-cache.jsonl")
    learned_router = FakeLearnedRouter(
        RouteDecision(
            selected_specialists=("condition_obligation_verifier_tool",),
            trigger_specialist_review=False,
            confidence=0.21,
            source="learned_router",
            model_name="local-qwen3-router",
            candidate_scores={
                "use_alternative_route_verifier_tool": 0.41,
                "use_equivalence_substitution_verifier_tool": 0.38,
                "use_condition_obligation_verifier_tool": 0.61,
                "trigger_specialist_review": 0.22,
            },
        )
    )
    predictor = PedCoTPredictor(
        client=FakeReviewClient(),
        cache_store=cache_store,
        stage1_tools_mode="none",
        stage2_tools_mode="triad",
        stage2_router_mode="learned-hybrid",
        stage2_router_confidence_threshold=0.55,
        learned_router=learned_router,
    )
    example = TraceExample(
        example_id="ex-4",
        dataset=DATASET_BIG_BENCH_MISTAKE,
        question="If x = 3, compute x + 2.",
        steps=["Substitute x = 3 into x + 2 and simplify."],
        gold_answer="5",
        model_answer="x + 2 = 3 + 2",
        gold_first_mistake_index=None,
    )

    prediction = predictor.predict_trace(example, model="fake-model")

    step = prediction.steps[0]
    called_names = set(step["parse_status"]["stage2_tool_called_names"])
    assert step["stage2_route_meta"]["source"] == "step_type_classifier_fallback"
    assert step["stage2_route_meta"]["fallback_used"] is True
    assert step["stage2_route_meta"]["fallback_reason"] == "low_confidence"
    assert "alternative_route_verifier_tool" in called_names
    assert "equivalence_substitution_verifier_tool" in called_names


def test_router_export_row_uses_weak_supervision_by_default(tmp_path: Path) -> None:
    cache_store = StageCacheStore(tmp_path / "stage-cache.jsonl")
    predictor = PedCoTPredictor(
        client=FakeReviewClient(),
        cache_store=cache_store,
        stage1_tools_mode="none",
        stage2_tools_mode="triad",
    )
    example = TraceExample(
        example_id="ex-5",
        dataset=DATASET_BIG_BENCH_MISTAKE,
        question="If x = 3, compute x + 2.",
        steps=["Substitute x = 3 into x + 2 and simplify."],
        gold_answer="5",
        model_answer="x + 2 = 3 + 2",
        gold_first_mistake_index=None,
    )

    prediction = predictor.predict_trace(example, model="fake-model")
    row = build_router_export_row(example, prediction.steps[0], label_strategy="weak-supervision")

    assert row is not None
    assert row["label_strategy"] == "weak-supervision"
    assert row["weak_labels"]["use_alternative_route_verifier_tool"] == 1
    assert row["weak_labels"]["use_equivalence_substitution_verifier_tool"] == 1
    assert row["weak_labels"]["trigger_specialist_review"] == 1
    assert row["sample_weight"] >= 1.5
    assert "gold_alignment_improved" in row["supervision"]["reasons"]


def test_router_export_row_falls_back_to_imitation_when_weak_signals_are_empty() -> None:
    example = TraceExample(
        example_id="ex-6",
        dataset=DATASET_BIG_BENCH_MISTAKE,
        question="Compute 2 + 2.",
        steps=["Compute 2 + 2."],
        gold_answer="4",
        model_answer="4",
        gold_first_mistake_index=None,
    )
    step = {
        "step_index": 0,
        "stage2_step_type": "arithmetic",
        "stage2_step_type_meta": {
            "risk_flags": ["arithmetic_risk"],
            "specialists": ["alternative_route_verifier_tool"],
        },
        "stage2_tool_trace": [],
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
            "trigger_specialist_review": True,
        },
    }

    row = build_router_export_row(example, step, label_strategy="weak-supervision")

    assert row is not None
    assert row["weak_labels"] == row["imitation_labels"]
    assert "fallback_to_imitation_labels" in row["supervision"]["reasons"]


def test_local_gold_step_label_skips_steps_after_first_mistake() -> None:
    example = TraceExample(
        example_id="ex-7",
        dataset=DATASET_BIG_BENCH_MISTAKE,
        question="Test",
        steps=["s1", "s2", "s3"],
        gold_answer="",
        model_answer="",
        gold_first_mistake_index=1,
    )

    assert local_gold_step_label(example, 0) == 1
    assert local_gold_step_label(example, 1) == 0
    assert local_gold_step_label(example, 2) is None
