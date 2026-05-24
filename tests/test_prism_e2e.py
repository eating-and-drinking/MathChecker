"""End-to-end test for PrismPredictor with a mock LLM client.

Exercises the full chain:
  TraceExample -> PedCoTPredictor (stage1 + stage2 LLM via FakeClient)
              -> PrismPredictor evidence extraction
              -> stage1 + stage2 + (no specialist) Bayes updates
              -> Phi-invariant + conformal stop
              -> TracePrediction output

This catches regressions in the LLM-glue layer that the synthetic-evidence
tests in test_prism_infer.py do NOT.
"""
from __future__ import annotations

from pathlib import Path

from mathchecker.core.models import TraceExample
from mathchecker.data.stores import StageCacheStore
from mathchecker.pipeline.predictor import PedCoTPredictor
from mathchecker.prism.predictor import PrismPredictor


# ---- Fake LLM client ----

# Pre-canned stage1 + stage2 responses for a 4-step trace, indexed by step_index.
# Question: sum of odd integers between 1 and 20. Step 2 has an arithmetic error
# (5 * 20 = 110 instead of 100). Stage1 explicitly predicts 100, so the stage1
# consistency extractor will fire on step 2.

_STAGE1_PER_STEP = {
    0: """1. Mathematical Concepts to Apply:
Identify all odd integers in the given range.
2. Key Analyses for the Next Step:
List odd numbers from 1 through 19 inclusive.
3. Mathematical Expressions to Compute:
Odd integers: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19.
""",
    1: """1. Mathematical Concepts to Apply:
Pairing for arithmetic series sums.
2. Key Analyses for the Next Step:
Pair smallest with largest to get equal pair sums.
3. Mathematical Expressions to Compute:
There are 5 pairs each summing to 20.
""",
    2: """1. Mathematical Concepts to Apply:
Multiply pair count by pair sum.
2. Key Analyses for the Next Step:
Compute 5 * 20.
3. Mathematical Expressions to Compute:
5 * 20 = 100.
""",
    3: """1. Mathematical Concepts to Apply:
Final answer reporting.
2. Key Analyses for the Next Step:
State the sum.
3. Mathematical Expressions to Compute:
Total sum = 100.
""",
}


def _stage2_clean(step_index: int) -> str:
    return f"""1. Mathematical Concepts to Apply:
The step uses the correct concept for step {step_index}.
Label: correct-and-aligned
2. Key Analyses for the Next Step:
The analysis matches the soft reference.
Label: correct-and-aligned
3. Mathematical Expressions to Compute:
Arithmetic is consistent.
Label: correct-and-aligned
"""


def _stage2_contradiction(step_index: int) -> str:
    return f"""1. Mathematical Concepts to Apply:
The step uses the right concept for step {step_index}.
Label: correct-and-aligned
2. Key Analyses for the Next Step:
The analysis matches stage1.
Label: correct-and-aligned
3. Mathematical Expressions to Compute:
The arithmetic does not match: 5 * 20 should be 100, not 110.
Label: contradiction-found
"""


_STAGE2_PER_STEP = {
    0: _stage2_clean(0),
    1: _stage2_clean(1),
    2: _stage2_contradiction(2),
    3: _stage2_clean(3),
}


class FakeLLMClient:
    """Implements the TextGenerationClient Protocol with canned responses."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        metadata=None,
        tools=None,
        tool_handlers=None,
        required_tool_names=None,
    ) -> tuple[str, dict]:
        meta = metadata or {}
        stage = meta.get("stage")
        step_index = int(meta.get("step_index", 0))
        self.calls.append({"stage": stage, "step_index": step_index})

        if stage == "stage1":
            text = _STAGE1_PER_STEP.get(step_index, _STAGE1_PER_STEP[0])
            return text, {"tool_trace": [], "tool_errors": [], "tool_called_names": [], "tool_call_failures": 0}

        if stage == "stage2":
            text = _STAGE2_PER_STEP.get(step_index, _STAGE2_PER_STEP[0])
            return text, {"tool_trace": [], "tool_errors": [], "tool_called_names": [], "tool_call_failures": 0}

        if stage == "stage2_step_type":
            return (
                '{"step_type": "arithmetic", "reasoning": "computation", "risk_flags": []}',
                {"tool_trace": [], "tool_errors": []},
            )

        if stage in {"stage2_review", "stage2_specialist_review"}:
            return _STAGE2_PER_STEP.get(step_index, _STAGE2_PER_STEP[0]), {"tool_trace": [], "tool_errors": []}

        return "", {"tool_trace": [], "tool_errors": []}


# ---- Tests ----

def _make_example() -> TraceExample:
    return TraceExample(
        example_id="e2e_demo",
        dataset="big-bench-mistake",
        question="Compute the sum of all odd integers between 1 and 20.",
        steps=[
            "Identify the odd integers between 1 and 20: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19.",
            "Pair them up: (1+19)=20, (3+17)=20, (5+15)=20, (7+13)=20, (9+11)=20. That's 5 pairs.",
            "Multiply: 5 * 20 = 110.",
            "Therefore the sum is 110.",
        ],
        gold_answer="100",
        model_answer="110",
        gold_first_mistake_index=2,
    )


def _make_prism_predictor(client, tmp_path: Path) -> PrismPredictor:
    cache = StageCacheStore(tmp_path / "stage_cache.jsonl")
    base = PedCoTPredictor(
        client=client,
        cache_store=cache,
        stage1_tools_mode="none",
        stage2_tools_mode="none",
        stage2_step_type_mode="heuristic",  # avoid an extra LLM call for step classification
        stage2_router_mode="step-type",
    )
    return PrismPredictor(
        base_predictor=base,
        delta=0.1,
        budget=3,
        lam=0.05,
        p_no_error_prior=0.4,
    )


def test_prism_predictor_end_to_end_finds_first_mistake(tmp_path: Path) -> None:
    client = FakeLLMClient()
    predictor = _make_prism_predictor(client, tmp_path)
    example = _make_example()

    prediction = predictor.predict_trace(example=example, model="fake-llm")

    assert prediction.completed, f"trace did not complete: {prediction.error}"
    assert prediction.pred_first_mistake_index == 2, (
        f"expected first mistake at step 2, got {prediction.pred_first_mistake_index}"
    )
    assert prediction.pred_trace_label == 0  # NEGATIVE_TRACE_LABEL (has error)
    assert prediction.gold_first_mistake_index == 2

    # PrismPredictor appends a meta row with the conformal schedule + posterior.
    meta_row = next((s for s in prediction.steps if s.get("prism_meta")), None)
    assert meta_row is not None
    assert meta_row["committed"] is True
    assert meta_row["conformal_schedule"]["delta"] == 0.1
    # Posterior at end should heavily concentrate on tau=2.
    posterior_probs = meta_row["posterior"]["probs"]
    assert posterior_probs[2] > 0.5, f"step-2 mass too low: {posterior_probs}"


def test_prism_predictor_makes_at_most_one_llm_call_per_stage_per_step(tmp_path: Path) -> None:
    """When stage2 finds the error at step 2, PRISM should EARLY-STOP and not
    invoke stage1/stage2 on step 3."""
    client = FakeLLMClient()
    predictor = _make_prism_predictor(client, tmp_path)
    example = _make_example()

    predictor.predict_trace(example=example, model="fake-llm")

    # Count stage1+stage2 calls per step.
    stage1_steps = {c["step_index"] for c in client.calls if c["stage"] == "stage1"}
    stage2_steps = {c["step_index"] for c in client.calls if c["stage"] == "stage2"}

    # Should have invoked stages 0, 1, 2 — but NOT 3 (early-stop after commit).
    assert stage1_steps == {0, 1, 2}, f"stage1 step coverage: {stage1_steps}"
    assert stage2_steps == {0, 1, 2}, f"stage2 step coverage: {stage2_steps}"


def test_prism_predictor_handles_clean_trace(tmp_path: Path) -> None:
    """All-clean trace -> should run to end, predict 'no error' (tau=infty)."""
    client = FakeLLMClient()
    predictor = _make_prism_predictor(client, tmp_path)
    example = TraceExample(
        example_id="clean_trace",
        dataset="big-bench-mistake",
        question="What is 7 + 5?",
        steps=[
            "Add 7 and 5 step by step.",
            "7 + 5 = 12.",
        ],
        gold_answer="12",
        model_answer="12",
        gold_first_mistake_index=None,
    )

    prediction = predictor.predict_trace(example=example, model="fake-llm")
    assert prediction.completed
    # All steps clean -> no error.
    assert prediction.pred_first_mistake_index is None
    assert prediction.pred_trace_label == 1  # POSITIVE_TRACE_LABEL (no error)
