from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from ..core.constants import DATASET_BIG_BENCH_MISTAKE

SPECIALIST_TOOL_NAMES = {
    "alternative_route_verifier_tool",
    "equivalence_substitution_verifier_tool",
    "condition_obligation_verifier_tool",
}

STEP_TYPE_CHOICES = (
    "decomposition",
    "final_conclusion",
    "condition_case",
    "substitution",
    "algebraic_transformation",
    "arithmetic",
    "reasoning_transition",
)

_STEP_TYPE_SPECIALIST_ROUTE = {
    "decomposition": (
        "alternative_route_verifier_tool",
        "equivalence_substitution_verifier_tool",
    ),
    "final_conclusion": (
        "alternative_route_verifier_tool",
        "condition_obligation_verifier_tool",
    ),
    "condition_case": (
        "alternative_route_verifier_tool",
        "condition_obligation_verifier_tool",
    ),
    "substitution": (
        "alternative_route_verifier_tool",
        "equivalence_substitution_verifier_tool",
    ),
    "algebraic_transformation": (
        "alternative_route_verifier_tool",
        "equivalence_substitution_verifier_tool",
    ),
    "arithmetic": ("alternative_route_verifier_tool",),
    "reasoning_transition": ("alternative_route_verifier_tool",),
}


@dataclass(slots=True, frozen=True)
class StepTypeClassification:
    step_type: str
    specialist_tool_names: tuple[str, ...]
    reasoning: str
    risk_flags: tuple[str, ...]
    source: str = "heuristic"
    confidence: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class StepTypeClassificationParse:
    classification: StepTypeClassification | None = None
    success: bool = False
    error: str | None = None


def build_step_type_classification(
    *,
    step_type: str,
    reasoning: str,
    risk_flags: tuple[str, ...] | list[str] = (),
    source: str = "heuristic",
    confidence: float | None = None,
) -> StepTypeClassification:
    normalized_type = normalize_step_type(step_type) or "reasoning_transition"
    return StepTypeClassification(
        step_type=normalized_type,
        specialist_tool_names=_STEP_TYPE_SPECIALIST_ROUTE[normalized_type],
        reasoning=reasoning,
        risk_flags=tuple(str(flag) for flag in risk_flags if isinstance(flag, str) and flag),
        source=source,
        confidence=confidence,
    )


def classify_step_type(
    *,
    dataset: str,
    question: str,
    previous_steps: list[str],
    current_step: str,
) -> StepTypeClassification:
    del question
    del previous_steps

    lowered = current_step.lower().strip()
    risk_flags: list[str] = []

    if _is_decomposition_step(dataset, lowered):
        return build_step_type_classification(
            step_type="decomposition",
            reasoning="Step rewrites the expression into components or an alternative decomposition.",
            risk_flags=["equivalence_risk", "decomposition_risk"],
        )

    if _is_final_conclusion_step(lowered):
        return build_step_type_classification(
            step_type="final_conclusion",
            reasoning="Step appears to conclude the solution or provide a final answer.",
            risk_flags=["conclusion_risk", "obligation_risk"],
        )

    if _is_condition_case_step(lowered):
        return build_step_type_classification(
            step_type="condition_case",
            reasoning="Step introduces a branch, assumption, or case split that must respect conditions.",
            risk_flags=["condition_risk", "branch_risk"],
        )

    if _is_substitution_step(lowered):
        return build_step_type_classification(
            step_type="substitution",
            reasoning="Step substitutes values, variables, or intermediate expressions into a new form.",
            risk_flags=["substitution_risk", "equivalence_risk"],
        )

    if _is_transformation_step(lowered):
        return build_step_type_classification(
            step_type="algebraic_transformation",
            reasoning="Step performs a rewrite or algebraic transformation that may be valid but differ from the reference path.",
            risk_flags=["equivalence_risk", "alternative_route_risk"],
        )

    if _is_arithmetic_step(lowered):
        return build_step_type_classification(
            step_type="arithmetic",
            reasoning="Step mainly performs numeric or arithmetic simplification.",
            risk_flags=["arithmetic_risk"],
        )

    return build_step_type_classification(
        step_type="reasoning_transition",
        reasoning="Step advances the solution in a general reasoning transition without a stronger structural signal.",
        risk_flags=risk_flags,
    )


def apply_step_type_specialist_route(
    required_tool_names: set[str],
    *,
    classification: StepTypeClassification,
    available_tool_names: set[str],
) -> set[str]:
    routed = {name for name in required_tool_names if name not in SPECIALIST_TOOL_NAMES}
    specialist_names = {
        name for name in classification.specialist_tool_names if name in available_tool_names
    }
    if not specialist_names and "alternative_route_verifier_tool" in available_tool_names:
        specialist_names.add("alternative_route_verifier_tool")
    routed.update(specialist_names)
    return routed


def parse_step_type_classifier_response(text: str) -> StepTypeClassificationParse:
    payload_text = _extract_json_object(text)
    if payload_text is None:
        return StepTypeClassificationParse(
            success=False,
            error="Could not find a JSON object in step-type classifier response.",
        )

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        return StepTypeClassificationParse(success=False, error=f"Invalid JSON: {exc.msg}.")

    if not isinstance(payload, dict):
        return StepTypeClassificationParse(success=False, error="Step-type classifier response JSON must be an object.")

    step_type = normalize_step_type(payload.get("step_type"))
    if step_type is None:
        return StepTypeClassificationParse(
            success=False,
            error="Missing or unsupported step_type in step-type classifier response.",
        )

    reasoning_raw = payload.get("reasoning")
    reasoning = reasoning_raw.strip() if isinstance(reasoning_raw, str) else ""
    if not reasoning:
        reasoning = "LLM classifier selected the step type without providing additional reasoning."

    risk_flags = _normalize_risk_flags(payload.get("risk_flags"))
    confidence = _normalize_confidence(payload.get("confidence"))
    classification = build_step_type_classification(
        step_type=step_type,
        reasoning=reasoning,
        risk_flags=risk_flags,
        source="llm",
        confidence=confidence,
    )
    return StepTypeClassificationParse(
        classification=classification,
        success=True,
    )


def normalize_step_type(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "decompose": "decomposition",
        "decomposition_step": "decomposition",
        "final": "final_conclusion",
        "conclusion": "final_conclusion",
        "final_answer": "final_conclusion",
        "condition": "condition_case",
        "case": "condition_case",
        "branch": "condition_case",
        "substitute": "substitution",
        "transformation": "algebraic_transformation",
        "rewrite": "algebraic_transformation",
        "algebraic": "algebraic_transformation",
        "calculation": "arithmetic",
        "numeric": "arithmetic",
        "reasoning": "reasoning_transition",
        "transition": "reasoning_transition",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in STEP_TYPE_CHOICES:
        return None
    return normalized


def _is_decomposition_step(dataset: str, lowered_step: str) -> bool:
    return (
        dataset == DATASET_BIG_BENCH_MISTAKE
        and "can be written as" in lowered_step
    ) or any(
        marker in lowered_step
        for marker in ["decompose", "decomposition", "split the expression", "break the expression"]
    )


def _is_final_conclusion_step(lowered_step: str) -> bool:
    return any(
        marker in lowered_step
        for marker in [
            "so the answer is",
            "therefore",
            "thus",
            "hence",
            "final answer",
            "final equation",
            "we get the answer",
        ]
    )


def _is_condition_case_step(lowered_step: str) -> bool:
    return any(
        marker in lowered_step
        for marker in ["if ", "when ", "case ", "cases", "assume", "suppose", "otherwise"]
    )


def _is_substitution_step(lowered_step: str) -> bool:
    return any(
        marker in lowered_step
        for marker in ["substitute", "substituting", "plug in", "plugging in", "replace", "replacing"]
    )


def _is_transformation_step(lowered_step: str) -> bool:
    return any(
        marker in lowered_step
        for marker in [
            "equivalently",
            "equivalent",
            "rewrite",
            "rewritten",
            "can be written as",
            "factor",
            "expand",
            "let ",
            "set ",
            "denote",
        ]
    )


def _is_arithmetic_step(lowered_step: str) -> bool:
    if any(marker in lowered_step for marker in ["let's calculate", "lets calculate", "compute", "calculate"]):
        return True
    return bool(re.search(r"\d+\s*[\+\-\*/=]\s*\d+", lowered_step))


def _extract_json_object(text: str) -> str | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            cleaned = "\n".join(lines[1:-1]).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return cleaned[start : end + 1]


def _normalize_risk_flags(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    return ()


def _normalize_confidence(value: object) -> float | None:
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        try:
            numeric = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric
