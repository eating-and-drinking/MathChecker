from __future__ import annotations

from typing import Any

from ..core.models import TraceExample
from .router import (
    SPECIALIST_TOOL_NAMES,
    RouterContext,
    build_router_training_labels,
    format_router_input_text,
)

_ALT_LABEL = "use_alternative_route_verifier_tool"
_EQUIV_LABEL = "use_equivalence_substitution_verifier_tool"
_COND_LABEL = "use_condition_obligation_verifier_tool"
_REVIEW_LABEL = "trigger_specialist_review"
_LABEL_TO_SPECIALIST = {
    _ALT_LABEL: "alternative_route_verifier_tool",
    _EQUIV_LABEL: "equivalence_substitution_verifier_tool",
    _COND_LABEL: "condition_obligation_verifier_tool",
}
_STEP_TYPE_LABEL_RELEVANCE = {
    "decomposition": {_ALT_LABEL},
    "substitution": {_ALT_LABEL, _EQUIV_LABEL},
    "algebraic_transformation": {_ALT_LABEL, _EQUIV_LABEL},
    "condition_case": {_ALT_LABEL, _COND_LABEL, _REVIEW_LABEL},
    "final_conclusion": {_ALT_LABEL, _COND_LABEL, _REVIEW_LABEL},
    "arithmetic": {_EQUIV_LABEL},
    "reasoning_transition": {_ALT_LABEL},
}
_RISK_FLAG_LABEL_RELEVANCE = {
    "substitution_risk": {_ALT_LABEL, _EQUIV_LABEL},
    "equivalence_risk": {_ALT_LABEL, _EQUIV_LABEL},
    "arithmetic_risk": {_EQUIV_LABEL},
    "condition_risk": {_COND_LABEL, _REVIEW_LABEL},
    "branch_risk": {_COND_LABEL, _REVIEW_LABEL},
    "conclusion_risk": {_COND_LABEL, _REVIEW_LABEL},
}


def prediction_step_specialists(step: dict) -> list[str]:
    route_meta = step.get("stage2_route_meta")
    if isinstance(route_meta, dict):
        selected = route_meta.get("selected_specialists")
        if isinstance(selected, list):
            return [str(item) for item in selected]

    status = step.get("parse_status")
    if isinstance(status, dict):
        selected = status.get("stage2_route_selected_specialists")
        if isinstance(selected, list):
            return [str(item) for item in selected]

    trace = step.get("stage2_tool_trace")
    specialists: list[str] = []
    if isinstance(trace, list):
        for item in trace:
            if not isinstance(item, dict) or item.get("discarded"):
                continue
            name = item.get("tool_name")
            if isinstance(name, str) and name in SPECIALIST_TOOL_NAMES and name not in specialists:
                specialists.append(name)
    return specialists


def prediction_step_review_trigger(step: dict) -> bool:
    route_meta = step.get("stage2_route_meta")
    if isinstance(route_meta, dict) and "trigger_specialist_review" in route_meta:
        return bool(route_meta.get("trigger_specialist_review"))

    status = step.get("parse_status")
    if isinstance(status, dict) and "stage2_route_trigger_specialist_review" in status:
        return bool(status.get("stage2_route_trigger_specialist_review"))
    if isinstance(status, dict) and "stage2_specialist_review_enabled" in status:
        return bool(status.get("stage2_specialist_review_enabled"))
    return False


def stage2_binary_label(parse_payload: dict[str, Any] | None) -> int | None:
    if not isinstance(parse_payload, dict):
        return None
    principle_labels = parse_payload.get("principle_labels")
    if isinstance(principle_labels, dict):
        labels = [value for value in principle_labels.values() if isinstance(value, str)]
    else:
        labels = [
            value
            for value in [
                parse_payload.get("mathematical_concepts_label"),
                parse_payload.get("key_analyses_label"),
                parse_payload.get("calculations_label"),
            ]
            if isinstance(value, str)
        ]
    if not labels:
        return None
    return 0 if "contradiction-found" in labels else 1


def local_gold_step_label(example: TraceExample, step_index: int) -> int | None:
    first_mistake = example.gold_first_mistake_index
    if first_mistake is None:
        return 1
    if step_index < first_mistake:
        return 1
    if step_index == first_mistake:
        return 0
    return None


def build_router_context(example: TraceExample, step: dict) -> RouterContext | None:
    step_index = step.get("step_index")
    if not isinstance(step_index, int):
        return None
    if step_index < 0 or step_index >= len(example.steps):
        return None

    step_type = step.get("stage2_step_type")
    step_type_meta = step.get("stage2_step_type_meta", {})
    if not isinstance(step_type_meta, dict):
        step_type_meta = {}
    risk_flags = step_type_meta.get("risk_flags", [])
    heuristic_specialists = step_type_meta.get("specialists", [])
    if not isinstance(risk_flags, list):
        risk_flags = []
    if not isinstance(heuristic_specialists, list):
        heuristic_specialists = []
    stage1_parse = step.get("stage1_parse", {})
    if not isinstance(stage1_parse, dict):
        stage1_parse = {}

    return RouterContext(
        dataset=example.dataset,
        question=example.question,
        previous_steps=tuple(example.steps[:step_index]),
        current_step=example.steps[step_index],
        heuristic_step_type=str(step_type or "reasoning_transition"),
        heuristic_risk_flags=tuple(str(item) for item in risk_flags),
        heuristic_specialists=tuple(str(item) for item in heuristic_specialists),
        stage1_mathematical_concepts=_optional_text(stage1_parse.get("mathematical_concepts")),
        stage1_key_analyses=_optional_text(stage1_parse.get("key_analyses")),
        stage1_calculations=_optional_text(stage1_parse.get("calculations")),
    )


def build_imitation_labels(step: dict) -> dict[str, int]:
    return build_router_training_labels(
        selected_specialists=prediction_step_specialists(step),
        trigger_specialist_review=prediction_step_review_trigger(step),
    )


def build_weak_supervision_labels(
    example: TraceExample,
    step: dict,
) -> tuple[dict[str, int], dict[str, float], dict[str, float], dict[str, Any]]:
    scores = {
        _ALT_LABEL: 0.0,
        _EQUIV_LABEL: 0.0,
        _COND_LABEL: 0.0,
        _REVIEW_LABEL: 0.0,
    }
    reasons: list[str] = []
    imitation_labels = build_imitation_labels(step)
    positive_tools: set[str] = set()
    used_specialists = prediction_step_specialists(step)
    used_specialist_set = set(used_specialists)
    review_trigger = prediction_step_review_trigger(step)
    route_meta = step.get("stage2_route_meta")
    route_source = route_meta.get("source") if isinstance(route_meta, dict) else None
    if route_source == "learned_router":
        reasons.append("router_source=learned_router")
    elif route_source == "step_type_classifier_fallback":
        reasons.append("router_source=step_type_classifier_fallback")

    for evidence in _iter_specialist_evidence(step):
        tool_name = evidence.get("tool_name")
        if not isinstance(tool_name, str):
            continue

        hard_contradiction = bool(evidence.get("hard_contradiction", False))
        valid_alternative = bool(evidence.get("valid_alternative", False))
        valid_equivalent = bool(evidence.get("valid_equivalent_transformation", False))
        valid_progression = bool(evidence.get("valid_progression", False))
        obligation_satisfied = bool(evidence.get("obligation_satisfied", True))
        binding_conflict = str(evidence.get("binding_conflict", "none"))
        contradiction_level = str(evidence.get("contradiction_level", "none"))

        if tool_name == "alternative_route_verifier_tool":
            if valid_alternative:
                scores[_ALT_LABEL] += 1.0
                positive_tools.add(tool_name)
                reasons.append("alternative_route_valid")
            if hard_contradiction:
                scores[_ALT_LABEL] += 0.5
                positive_tools.add(tool_name)
                reasons.append("alternative_route_detected_conflict")

        if tool_name == "equivalence_substitution_verifier_tool":
            if valid_equivalent:
                scores[_EQUIV_LABEL] += 1.0
                positive_tools.add(tool_name)
                reasons.append("equivalence_verified")
            if hard_contradiction:
                scores[_EQUIV_LABEL] += 1.0
                positive_tools.add(tool_name)
                reasons.append("equivalence_detected_conflict")

        if tool_name == "condition_obligation_verifier_tool":
            if hard_contradiction or binding_conflict != "none" or contradiction_level != "none" or not obligation_satisfied:
                scores[_COND_LABEL] += 1.0
                positive_tools.add(tool_name)
                reasons.append("condition_detected_conflict")
            elif valid_progression and _step_type_is_conditional_or_final(step):
                scores[_COND_LABEL] += 0.5
                positive_tools.add(tool_name)
                reasons.append("condition_valid_progression_on_conditional_step")

    original_label = stage2_binary_label(step.get("stage2_original_parse"))
    final_label = stage2_binary_label(step.get("stage2_parse"))
    specialist_review_label = stage2_binary_label(step.get("stage2_specialist_review_parse"))
    gold_label = local_gold_step_label(example, int(step.get("step_index", -1)))
    specialist_review_applied = bool(step.get("stage2_specialist_review_applied", False))
    adjustment_applied = bool(
        isinstance(step.get("parse_status"), dict)
        and step["parse_status"].get("stage2_specialist_adjustment_applied", False)
    )

    if specialist_review_applied or step.get("stage2_specialist_review_parse") is not None:
        scores[_REVIEW_LABEL] += 1.0
        reasons.append("specialist_review_executed")
    if adjustment_applied:
        scores[_REVIEW_LABEL] += 1.0
        reasons.append("specialist_adjustment_applied")

    if specialist_review_label is not None and original_label is not None and specialist_review_label != original_label:
        scores[_REVIEW_LABEL] += 0.5
        reasons.append("specialist_review_changed_step_label")

    if original_label == 0 and final_label == 1:
        scores[_REVIEW_LABEL] += 0.5
        reasons.append("final_route_relaxed_contradiction")
        if "alternative_route_verifier_tool" in used_specialist_set:
            scores[_ALT_LABEL] += 0.5
        if "equivalence_substitution_verifier_tool" in used_specialist_set:
            scores[_EQUIV_LABEL] += 0.5
    elif original_label == 1 and final_label == 0:
        scores[_REVIEW_LABEL] += 0.5
        reasons.append("final_route_added_contradiction")
        if "equivalence_substitution_verifier_tool" in used_specialist_set:
            scores[_EQUIV_LABEL] += 0.5
        if "condition_obligation_verifier_tool" in used_specialist_set:
            scores[_COND_LABEL] += 0.5

    if gold_label is not None and final_label is not None:
        if original_label is not None and original_label != gold_label and final_label == gold_label:
            scores[_REVIEW_LABEL] += 1.0
            reasons.append("gold_alignment_improved")
            for tool_name in used_specialist_set:
                if tool_name == "alternative_route_verifier_tool":
                    scores[_ALT_LABEL] += 0.75
                elif tool_name == "equivalence_substitution_verifier_tool":
                    scores[_EQUIV_LABEL] += 0.75
                elif tool_name == "condition_obligation_verifier_tool":
                    scores[_COND_LABEL] += 0.75
        elif original_label is not None and original_label == gold_label and final_label != gold_label:
            scores[_REVIEW_LABEL] -= 1.0
            reasons.append("gold_alignment_degraded")
            for tool_name in used_specialist_set:
                if tool_name == "alternative_route_verifier_tool":
                    scores[_ALT_LABEL] -= 0.75
                elif tool_name == "equivalence_substitution_verifier_tool":
                    scores[_EQUIV_LABEL] -= 0.75
                elif tool_name == "condition_obligation_verifier_tool":
                    scores[_COND_LABEL] -= 0.75

    if positive_tools:
        scores[_REVIEW_LABEL] += 0.25

    weak_labels = {label: int(score > 0.0) for label, score in scores.items()}
    if all(value == 0 for value in weak_labels.values()):
        weak_labels = imitation_labels
        reasons.append("fallback_to_imitation_labels")

    heuristic_specialist_set = _heuristic_specialist_set(step)
    heuristic_risk_flags = _heuristic_risk_flag_set(step)
    policy_targets = _build_policy_targets(
        scores=scores,
        weak_labels=weak_labels,
        imitation_labels=imitation_labels,
        heuristic_specialists=heuristic_specialist_set,
        positive_tools=positive_tools,
        specialist_review_positive=(
            specialist_review_applied
            or adjustment_applied
            or "specialist_review_changed_step_label" in reasons
            or "gold_alignment_improved" in reasons
        ),
    )
    policy_confidence = _policy_confidence(policy_targets)
    expected_gain_targets, expected_gain_breakdown = _build_expected_gain_targets(
        step=step,
        scores=scores,
        policy_targets=policy_targets,
        used_specialists=used_specialist_set,
        heuristic_specialists=heuristic_specialist_set,
        heuristic_risk_flags=heuristic_risk_flags,
        positive_tools=positive_tools,
        review_trigger=review_trigger,
        specialist_review_positive=(
            specialist_review_applied
            or adjustment_applied
            or "specialist_review_changed_step_label" in reasons
            or "gold_alignment_improved" in reasons
        ),
        original_label=original_label,
        final_label=final_label,
        gold_label=gold_label,
        route_improved=original_label is not None and final_label is not None and original_label != final_label,
        gold_improved="gold_alignment_improved" in reasons,
        gold_degraded="gold_alignment_degraded" in reasons,
    )
    expected_gain_confidence = _policy_confidence(expected_gain_targets)
    benefit_signal_strength = round(
        sum(abs(score) for score in scores.values()) / max(len(scores), 1),
        4,
    )

    sample_weight = 1.0
    if "gold_alignment_improved" in reasons:
        sample_weight = 2.0
    elif specialist_review_applied or adjustment_applied or positive_tools:
        sample_weight = 1.5
    if policy_confidence >= 0.45:
        sample_weight = min(2.5, sample_weight + 0.25)
    if expected_gain_confidence >= 0.55:
        sample_weight = min(3.0, sample_weight + 0.25)

    supervision = {
        "sample_weight": round(sample_weight, 4),
        "label_scores": {label: round(score, 4) for label, score in scores.items()},
        "policy_targets": policy_targets,
        "policy_confidence": policy_confidence,
        "expected_gain_targets": expected_gain_targets,
        "expected_gain_confidence": expected_gain_confidence,
        "expected_gain_breakdown": expected_gain_breakdown,
        "expected_gain_learning_version": "counterfactual-v1",
        "benefit_signal_strength": benefit_signal_strength,
        "benefit_learning_version": "benefit-aware-v1",
        "reasons": reasons,
        "used_specialists": used_specialists,
        "gold_step_label": gold_label,
        "original_step_label": original_label,
        "final_step_label": final_label,
        "specialist_review_step_label": specialist_review_label,
    }
    return weak_labels, policy_targets, expected_gain_targets, supervision


def build_router_export_row(
    example: TraceExample,
    step: dict,
    *,
    label_strategy: str,
) -> dict[str, Any] | None:
    context = build_router_context(example, step)
    if context is None:
        return None

    imitation_labels = build_imitation_labels(step)
    weak_labels, policy_targets, expected_gain_targets, supervision = build_weak_supervision_labels(example, step)
    if label_strategy == "expected-gain":
        labels = expected_gain_targets
    elif label_strategy == "benefit-aware":
        labels = policy_targets
    elif label_strategy == "weak-supervision":
        labels = weak_labels
    else:
        labels = imitation_labels

    return {
        "dataset": example.dataset,
        "example_id": example.example_id,
        "step_index": step["step_index"],
        "question": example.question,
        "previous_steps": list(context.previous_steps),
        "current_step": context.current_step,
        "stage1_mathematical_concepts": context.stage1_mathematical_concepts,
        "stage1_key_analyses": context.stage1_key_analyses,
        "stage1_calculations": context.stage1_calculations,
        "heuristic_step_type": context.heuristic_step_type,
        "heuristic_risk_flags": list(context.heuristic_risk_flags),
        "heuristic_specialists": list(context.heuristic_specialists),
        "text": format_router_input_text(context),
        "labels": labels,
        "imitation_labels": imitation_labels,
        "weak_labels": weak_labels,
        "policy_targets": policy_targets,
        "expected_gain_targets": expected_gain_targets,
        "label_strategy": label_strategy,
        "sample_weight": supervision["sample_weight"],
        "supervision": supervision,
    }


def _iter_specialist_evidence(step: dict) -> list[dict[str, Any]]:
    trace = step.get("stage2_tool_trace")
    if not isinstance(trace, list):
        return []
    evidence: list[dict[str, Any]] = []
    for item in trace:
        if not isinstance(item, dict) or item.get("discarded"):
            continue
        tool_name = item.get("tool_name")
        result = item.get("result")
        if not isinstance(tool_name, str) or tool_name not in SPECIALIST_TOOL_NAMES or not isinstance(result, dict):
            continue
        evidence.append({"tool_name": tool_name, **result})
    return evidence


def _step_type_is_conditional_or_final(step: dict) -> bool:
    step_type = step.get("stage2_step_type")
    if not isinstance(step_type, str):
        return False
    return step_type in {"condition_case", "final_conclusion"}


def _heuristic_specialist_set(step: dict) -> set[str]:
    step_type_meta = step.get("stage2_step_type_meta", {})
    if not isinstance(step_type_meta, dict):
        return set()
    specialists = step_type_meta.get("specialists", [])
    if not isinstance(specialists, list):
        return set()
    return {str(item) for item in specialists}


def _heuristic_risk_flag_set(step: dict) -> set[str]:
    step_type_meta = step.get("stage2_step_type_meta", {})
    if not isinstance(step_type_meta, dict):
        return set()
    risk_flags = step_type_meta.get("risk_flags", [])
    if not isinstance(risk_flags, list):
        return set()
    return {str(item) for item in risk_flags}


def _build_policy_targets(
    *,
    scores: dict[str, float],
    weak_labels: dict[str, int],
    imitation_labels: dict[str, int],
    heuristic_specialists: set[str],
    positive_tools: set[str],
    specialist_review_positive: bool,
) -> dict[str, float]:
    positive_map = {
        _ALT_LABEL: "alternative_route_verifier_tool" in positive_tools,
        _EQUIV_LABEL: "equivalence_substitution_verifier_tool" in positive_tools,
        _COND_LABEL: "condition_obligation_verifier_tool" in positive_tools,
        _REVIEW_LABEL: specialist_review_positive,
    }
    heuristic_map = {
        _ALT_LABEL: "alternative_route_verifier_tool" in heuristic_specialists,
        _EQUIV_LABEL: "equivalence_substitution_verifier_tool" in heuristic_specialists,
        _COND_LABEL: "condition_obligation_verifier_tool" in heuristic_specialists,
        _REVIEW_LABEL: bool(heuristic_specialists),
    }
    targets: dict[str, float] = {}
    for label, score in scores.items():
        target = 0.4 + (0.15 * score)
        target += 0.1 if weak_labels.get(label, 0) else -0.1
        if imitation_labels.get(label, 0):
            target += 0.08
        if heuristic_map.get(label, False):
            target += 0.05
        if imitation_labels.get(label, 0) and score <= 0.0 and not positive_map.get(label, False):
            target -= 0.15
        targets[label] = round(min(0.98, max(0.02, target)), 4)
    return targets


def _policy_confidence(policy_targets: dict[str, float]) -> float:
    if not policy_targets:
        return 0.0
    return round(
        sum(abs(value - 0.5) * 2.0 for value in policy_targets.values()) / len(policy_targets),
        4,
    )


def _build_expected_gain_targets(
    *,
    step: dict,
    scores: dict[str, float],
    policy_targets: dict[str, float],
    used_specialists: set[str],
    heuristic_specialists: set[str],
    heuristic_risk_flags: set[str],
    positive_tools: set[str],
    review_trigger: bool,
    specialist_review_positive: bool,
    original_label: int | None,
    final_label: int | None,
    gold_label: int | None,
    route_improved: bool,
    gold_improved: bool,
    gold_degraded: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    unresolved_error = gold_label is not None and final_label is not None and final_label != gold_label
    persistent_route_failure = unresolved_error and original_label is not None and original_label == final_label
    persistent_contradiction = original_label == 0 and final_label == 0
    targets: dict[str, float] = {}
    breakdown: dict[str, Any] = {}

    for label in (_ALT_LABEL, _EQUIV_LABEL, _COND_LABEL):
        specialist = _LABEL_TO_SPECIALIST[label]
        used = specialist in used_specialists
        positive = specialist in positive_tools
        relevant = _label_is_relevant(
            step=step,
            label=label,
            heuristic_specialists=heuristic_specialists,
            heuristic_risk_flags=heuristic_risk_flags,
        )
        observed_gain = 0.0
        counterfactual_gain = 0.0
        cost = 0.0
        reasons: list[str] = []

        if used:
            cost += 0.12
            reasons.append("action_used")
            if positive:
                observed_gain += 1.0
                reasons.append("positive_tool_signal")
            if gold_improved:
                observed_gain += 0.85
                reasons.append("gold_alignment_improved")
            elif route_improved:
                observed_gain += 0.35
                reasons.append("route_changed")
            if gold_degraded:
                observed_gain -= 1.05
                reasons.append("gold_alignment_degraded")
            elif not positive and not route_improved:
                observed_gain -= 0.4
                reasons.append("used_without_visible_gain")
            if not relevant and not positive:
                observed_gain -= 0.2
                reasons.append("used_while_low_relevance")
        else:
            reasons.append("action_not_used")
            if relevant and unresolved_error:
                counterfactual_gain += 0.85
                reasons.append("missed_relevant_action_on_unresolved_error")
            elif relevant and persistent_contradiction:
                counterfactual_gain += 0.55
                reasons.append("missed_relevant_action_on_persistent_contradiction")
            if specialist in heuristic_specialists:
                counterfactual_gain += 0.2
                reasons.append("heuristic_route_disagrees")
            if not unresolved_error and not relevant:
                counterfactual_gain -= 0.15
                reasons.append("low_need_to_add_action")

        gain = observed_gain + counterfactual_gain - cost
        target = _blend_prior_and_gain(policy_targets[label], gain)
        targets[label] = target
        breakdown[label] = {
            "used": used,
            "positive_signal": positive,
            "relevant": relevant,
            "observed_gain": round(observed_gain, 4),
            "counterfactual_gain": round(counterfactual_gain, 4),
            "cost": round(cost, 4),
            "expected_gain": round(gain, 4),
            "target": target,
            "reasons": reasons,
        }

    review_observed = review_trigger or step.get("stage2_specialist_review_parse") is not None
    review_observed_gain = 0.0
    review_counterfactual_gain = 0.0
    review_cost = 0.0
    review_reasons: list[str] = []
    review_relevant = bool(heuristic_specialists) or bool(positive_tools)

    if review_observed:
        review_cost += 0.1
        review_reasons.append("review_executed")
        if specialist_review_positive:
            review_observed_gain += 1.1
            review_reasons.append("review_showed_positive_effect")
        if gold_improved:
            review_observed_gain += 0.75
            review_reasons.append("review_helped_gold_alignment")
        elif route_improved:
            review_observed_gain += 0.35
            review_reasons.append("review_changed_route")
        if gold_degraded:
            review_observed_gain -= 1.0
            review_reasons.append("review_correlated_with_gold_degradation")
        elif not specialist_review_positive:
            review_observed_gain -= 0.35
            review_reasons.append("review_without_visible_gain")
    else:
        review_reasons.append("review_not_executed")
        if unresolved_error and review_relevant:
            review_counterfactual_gain += 0.85
            review_reasons.append("missed_review_on_unresolved_error")
        if persistent_route_failure:
            review_counterfactual_gain += 0.45
            review_reasons.append("missed_review_on_persistent_route_failure")
        if not unresolved_error and not review_relevant:
            review_counterfactual_gain -= 0.1
            review_reasons.append("low_need_to_add_review")

    review_gain = review_observed_gain + review_counterfactual_gain - review_cost
    review_target = _blend_prior_and_gain(policy_targets[_REVIEW_LABEL], review_gain)
    targets[_REVIEW_LABEL] = review_target
    breakdown[_REVIEW_LABEL] = {
        "used": review_observed,
        "positive_signal": specialist_review_positive,
        "relevant": review_relevant,
        "observed_gain": round(review_observed_gain, 4),
        "counterfactual_gain": round(review_counterfactual_gain, 4),
        "cost": round(review_cost, 4),
        "expected_gain": round(review_gain, 4),
        "target": review_target,
        "reasons": review_reasons,
    }
    return targets, breakdown


def _label_is_relevant(
    *,
    step: dict,
    label: str,
    heuristic_specialists: set[str],
    heuristic_risk_flags: set[str],
) -> bool:
    if label == _REVIEW_LABEL:
        return bool(heuristic_specialists)
    specialist = _LABEL_TO_SPECIALIST[label]
    if specialist in heuristic_specialists:
        return True

    step_type = step.get("stage2_step_type")
    if isinstance(step_type, str) and label in _STEP_TYPE_LABEL_RELEVANCE.get(step_type, set()):
        return True

    for risk_flag in heuristic_risk_flags:
        if label in _RISK_FLAG_LABEL_RELEVANCE.get(risk_flag, set()):
            return True
    return False


def _blend_prior_and_gain(prior: float, gain: float) -> float:
    gain_component = min(0.98, max(0.02, 0.5 + (0.22 * gain)))
    gain_weight = min(0.8, 0.35 + (0.25 * abs(gain)))
    prior_weight = 1.0 - gain_weight
    target = (prior_weight * prior) + (gain_weight * gain_component)
    return round(min(0.98, max(0.02, target)), 4)


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None
