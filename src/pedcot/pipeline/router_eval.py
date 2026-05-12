from __future__ import annotations

from typing import Any

from .router import ROUTER_LABELS, RouterContext, build_route_decision_from_scores


def build_row_router_context(row: dict[str, Any]) -> RouterContext:
    previous_steps = row.get("previous_steps", [])
    risk_flags = row.get("heuristic_risk_flags", [])
    heuristic_specialists = row.get("heuristic_specialists", [])
    if not isinstance(previous_steps, list):
        previous_steps = []
    if not isinstance(risk_flags, list):
        risk_flags = []
    if not isinstance(heuristic_specialists, list):
        heuristic_specialists = []
    return RouterContext(
        dataset=str(row.get("dataset", "")),
        question=str(row.get("question", "")),
        previous_steps=tuple(str(item) for item in previous_steps),
        current_step=str(row.get("current_step", "")),
        heuristic_step_type=str(row.get("heuristic_step_type", "reasoning_transition")),
        heuristic_risk_flags=tuple(str(item) for item in risk_flags),
        heuristic_specialists=tuple(str(item) for item in heuristic_specialists),
        stage1_mathematical_concepts=_optional_text(row.get("stage1_mathematical_concepts")),
        stage1_key_analyses=_optional_text(row.get("stage1_key_analyses")),
        stage1_calculations=_optional_text(row.get("stage1_calculations")),
    )


def heuristic_route_labels(row: dict[str, Any]) -> dict[str, int]:
    heuristic_specialists = row.get("heuristic_specialists", [])
    if not isinstance(heuristic_specialists, list):
        heuristic_specialists = []
    specialist_set = {str(item) for item in heuristic_specialists}
    return {
        "use_alternative_route_verifier_tool": int("alternative_route_verifier_tool" in specialist_set),
        "use_equivalence_substitution_verifier_tool": int("equivalence_substitution_verifier_tool" in specialist_set),
        "use_condition_obligation_verifier_tool": int("condition_obligation_verifier_tool" in specialist_set),
        "trigger_specialist_review": int(bool(specialist_set)),
    }


def imitation_route_labels(row: dict[str, Any]) -> dict[str, int]:
    return {
        label: int(value)
        for label, value in label_values_from_row(row, "imitation_labels").items()
    }


def labels_from_scores(
    scores: dict[str, float],
    *,
    confidence_threshold: float,
    per_label_thresholds: dict[str, float] | None = None,
) -> dict[str, int]:
    decision = build_route_decision_from_scores(
        scores,
        confidence_threshold=confidence_threshold,
        per_label_thresholds=per_label_thresholds,
        source="offline_eval",
        model_name=None,
    )
    selected = set(decision.selected_specialists)
    return {
        "use_alternative_route_verifier_tool": int("alternative_route_verifier_tool" in selected),
        "use_equivalence_substitution_verifier_tool": int("equivalence_substitution_verifier_tool" in selected),
        "use_condition_obligation_verifier_tool": int("condition_obligation_verifier_tool" in selected),
        "trigger_specialist_review": int(decision.trigger_specialist_review),
    }


def label_values_from_row(row: dict[str, Any], field: str) -> dict[str, float]:
    for field_name in _field_fallback_order(field):
        payload = row.get(field_name)
        if not isinstance(payload, dict):
            continue
        if not all(label in payload for label in ROUTER_LABELS):
            continue
        values: dict[str, float] = {}
        valid = True
        for label in ROUTER_LABELS:
            value = payload.get(label)
            if not isinstance(value, (int, float)):
                valid = False
                break
            values[label] = float(value)
        if valid:
            return values
    raise ValueError(f"Row does not contain a valid {field} payload.")


def binary_label_targets(
    row: dict[str, Any],
    *,
    field: str,
    threshold: float = 0.5,
) -> dict[str, int]:
    values = label_values_from_row(row, field)
    return {label: int(value >= threshold) for label, value in values.items()}


def evaluate_router_predictions(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    target_field: str,
    binary_target_threshold: float = 0.5,
    confidence_threshold: float = 0.55,
) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError("rows and predictions must have the same length.")

    tp = fp = fn = 0.0
    exact_match = 0
    utility_sum = 0.0
    selected_target_sum = 0.0
    selected_count = 0
    positive_count = 0
    confidence_values: list[float] = []
    high_confidence = 0
    fallback_count = 0
    score_probs: list[float] = []
    score_targets: list[int] = []
    per_label_counts = {
        label: {"tp": 0.0, "fp": 0.0, "fn": 0.0}
        for label in ROUTER_LABELS
    }

    for row, prediction in zip(rows, predictions, strict=True):
        target_values = label_values_from_row(row, target_field)
        gold = {label: int(value >= binary_target_threshold) for label, value in target_values.items()}
        pred_labels = {
            label: int(bool(prediction.get("labels", {}).get(label, 0)))
            for label in ROUTER_LABELS
        }

        if all(pred_labels[label] == gold[label] for label in ROUTER_LABELS):
            exact_match += 1

        confidence = prediction.get("confidence")
        if isinstance(confidence, (int, float)):
            confidence_value = float(confidence)
            confidence_values.append(confidence_value)
            if confidence_value >= confidence_threshold:
                high_confidence += 1

        if bool(prediction.get("fallback_used", False)):
            fallback_count += 1

        score_payload = prediction.get("scores", {})
        if isinstance(score_payload, dict):
            for label in ROUTER_LABELS:
                score = score_payload.get(label)
                if isinstance(score, (int, float)):
                    score_probs.append(float(score))
                    score_targets.append(gold[label])

        for label in ROUTER_LABELS:
            pred_value = pred_labels[label]
            gold_value = gold[label]
            target_value = target_values[label]
            if pred_value and gold_value:
                tp += 1.0
                per_label_counts[label]["tp"] += 1.0
            elif pred_value and not gold_value:
                fp += 1.0
                per_label_counts[label]["fp"] += 1.0
            elif (not pred_value) and gold_value:
                fn += 1.0
                per_label_counts[label]["fn"] += 1.0

            utility_sum += target_value if pred_value else 1.0 - target_value
            if pred_value:
                selected_target_sum += target_value
                selected_count += 1
                positive_count += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    per_label: dict[str, dict[str, float]] = {}
    macro_f1_values: list[float] = []
    for label, counts in per_label_counts.items():
        label_precision = counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) else 0.0
        label_recall = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) else 0.0
        label_f1 = (
            2 * label_precision * label_recall / (label_precision + label_recall)
            if (label_precision + label_recall)
            else 0.0
        )
        macro_f1_values.append(label_f1)
        per_label[label] = {
            "precision": round(label_precision, 4),
            "recall": round(label_recall, 4),
            "f1": round(label_f1, 4),
        }

    num_rows = len(rows)
    num_decisions = max(num_rows * len(ROUTER_LABELS), 1)
    avg_policy_utility = utility_sum / num_decisions
    calibration_error = _expected_calibration_error(score_probs, score_targets)

    return {
        "num_rows": num_rows,
        "micro_precision": round(precision, 4),
        "micro_recall": round(recall, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(sum(macro_f1_values) / max(len(macro_f1_values), 1), 4),
        "exact_match": round(exact_match / max(num_rows, 1), 4),
        "avg_policy_utility": round(avg_policy_utility, 4),
        "avg_selected_target": round(selected_target_sum / selected_count, 4) if selected_count else None,
        "positive_rate": round(positive_count / num_decisions, 4),
        "avg_confidence": round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else None,
        "high_confidence_coverage": round(high_confidence / max(num_rows, 1), 4) if confidence_values else None,
        "fallback_rate": round(fallback_count / max(num_rows, 1), 4),
        "expected_calibration_error": calibration_error,
        "per_label": per_label,
    }


def build_route_ablation_summary(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    target_field: str,
    binary_target_threshold: float = 0.5,
    confidence_threshold: float = 0.55,
) -> dict[str, Any]:
    baseline = evaluate_router_predictions(
        rows,
        predictions,
        target_field=target_field,
        binary_target_threshold=binary_target_threshold,
        confidence_threshold=confidence_threshold,
    )
    ablations: dict[str, Any] = {}
    for label in ROUTER_LABELS:
        ablated_predictions: list[dict[str, Any]] = []
        for prediction in predictions:
            labels = {
                route_label: int(bool(prediction.get("labels", {}).get(route_label, 0)))
                for route_label in ROUTER_LABELS
            }
            labels[label] = 0
            ablated_predictions.append(
                {
                    **prediction,
                    "labels": labels,
                }
            )
        metrics = evaluate_router_predictions(
            rows,
            ablated_predictions,
            target_field=target_field,
            binary_target_threshold=binary_target_threshold,
            confidence_threshold=confidence_threshold,
        )
        ablations[label] = {
            "micro_f1": metrics["micro_f1"],
            "avg_policy_utility": metrics["avg_policy_utility"],
            "delta_micro_f1": round(metrics["micro_f1"] - baseline["micro_f1"], 4),
            "delta_avg_policy_utility": round(metrics["avg_policy_utility"] - baseline["avg_policy_utility"], 4),
        }
    return ablations


def calibrate_per_label_thresholds(
    rows: list[dict[str, Any]],
    score_rows: list[dict[str, float]],
    *,
    target_field: str,
    objective: str = "utility",
    binary_target_threshold: float = 0.5,
    grid: list[float] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    if len(rows) != len(score_rows):
        raise ValueError("rows and score_rows must have the same length.")
    threshold_grid = grid or [round(value / 100, 2) for value in range(20, 86, 5)]
    thresholds: dict[str, float] = {}
    report: dict[str, Any] = {}

    for label in ROUTER_LABELS:
        best_threshold = 0.55
        best_score = -1.0
        trials: dict[str, float] = {}
        for threshold in threshold_grid:
            score = _single_label_objective(
                rows,
                score_rows,
                label=label,
                threshold=threshold,
                target_field=target_field,
                objective=objective,
                binary_target_threshold=binary_target_threshold,
            )
            trials[f"{threshold:.2f}"] = round(score, 4)
            if score > best_score:
                best_threshold = threshold
                best_score = score
        thresholds[label] = round(best_threshold, 4)
        report[label] = {
            "best_threshold": round(best_threshold, 4),
            "best_score": round(best_score, 4),
            "objective": objective,
            "grid_scores": trials,
        }

    return thresholds, report


def _single_label_objective(
    rows: list[dict[str, Any]],
    score_rows: list[dict[str, float]],
    *,
    label: str,
    threshold: float,
    target_field: str,
    objective: str,
    binary_target_threshold: float,
) -> float:
    if objective == "utility":
        utility_sum = 0.0
        for row, scores in zip(rows, score_rows, strict=True):
            target = label_values_from_row(row, target_field)[label]
            pred = float(scores.get(label, 0.0)) >= threshold
            utility_sum += target if pred else 1.0 - target
        return utility_sum / max(len(rows), 1)

    tp = fp = fn = 0.0
    for row, scores in zip(rows, score_rows, strict=True):
        gold = binary_label_targets(
            row,
            field=target_field,
            threshold=binary_target_threshold,
        )[label]
        pred = int(float(scores.get(label, 0.0)) >= threshold)
        if pred and gold:
            tp += 1.0
        elif pred and not gold:
            fp += 1.0
        elif (not pred) and gold:
            fn += 1.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def _field_fallback_order(field: str) -> list[str]:
    if field == "expected_gain_targets":
        return ["expected_gain_targets", "labels", "policy_targets", "weak_labels", "imitation_labels"]
    if field == "policy_targets":
        return ["policy_targets", "labels", "weak_labels", "imitation_labels"]
    if field == "weak_labels":
        return ["weak_labels", "labels", "imitation_labels"]
    if field == "imitation_labels":
        return ["imitation_labels", "labels"]
    return [field]


def _expected_calibration_error(probabilities: list[float], labels: list[int], bins: int = 10) -> float | None:
    if not probabilities or len(probabilities) != len(labels):
        return None
    bin_totals = [0 for _ in range(bins)]
    bin_confidence = [0.0 for _ in range(bins)]
    bin_accuracy = [0.0 for _ in range(bins)]
    for probability, label in zip(probabilities, labels, strict=True):
        clipped = min(max(probability, 0.0), 1.0)
        index = min(int(clipped * bins), bins - 1)
        bin_totals[index] += 1
        bin_confidence[index] += clipped
        bin_accuracy[index] += float(label)
    total = len(probabilities)
    error = 0.0
    for count, confidence_sum, accuracy_sum in zip(bin_totals, bin_confidence, bin_accuracy, strict=True):
        if count == 0:
            continue
        avg_confidence = confidence_sum / count
        avg_accuracy = accuracy_sum / count
        error += (count / total) * abs(avg_confidence - avg_accuracy)
    return round(error, 4)


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None
