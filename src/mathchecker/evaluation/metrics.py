from __future__ import annotations

from dataclasses import dataclass

from ..core.models import TracePrediction


@dataclass(slots=True)
class BinaryMetrics:
    precision: float
    recall: float
    f1: float


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def binary_metrics(y_true: list[int], y_pred: list[int], positive_label: int) -> BinaryMetrics:
    tp = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth == positive_label and pred == positive_label)
    fp = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth != positive_label and pred == positive_label)
    fn = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth == positive_label and pred != positive_label)

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall) if precision or recall else 0.0
    return BinaryMetrics(precision=precision, recall=recall, f1=f1)


def compute_metrics(predictions: list[TracePrediction]) -> dict[str, float]:
    if not predictions:
        raise ValueError("No predictions found for evaluation.")
    incomplete = [prediction.example_id for prediction in predictions if not prediction.completed or prediction.pred_trace_label is None]
    if incomplete:
        raise ValueError(f"Incomplete predictions cannot be evaluated: {', '.join(incomplete[:5])}")

    gold_trace = [prediction.gold_trace_label for prediction in predictions]
    pred_trace = [prediction.pred_trace_label for prediction in predictions]
    mf_acc = sum(
        1
        for prediction in predictions
        if prediction.pred_first_mistake_index == prediction.gold_first_mistake_index
    ) / len(predictions)
    cls_acc = sum(1 for gold, pred in zip(gold_trace, pred_trace, strict=True) if gold == pred) / len(predictions)

    pos_metrics = binary_metrics(gold_trace, pred_trace, positive_label=1)
    neg_metrics = binary_metrics(gold_trace, pred_trace, positive_label=0)

    positive_support = sum(1 for value in gold_trace if value == 1)
    negative_support = len(gold_trace) - positive_support
    avg_f1 = (
        positive_support * pos_metrics.f1 + negative_support * neg_metrics.f1
    ) / len(gold_trace)

    return {
        "MF_Acc": mf_acc,
        "Avg_F1": avg_f1,
        "Cls_Acc": cls_acc,
        "P+": pos_metrics.precision,
        "R+": pos_metrics.recall,
        "F+": pos_metrics.f1,
        "P-": neg_metrics.precision,
        "R-": neg_metrics.recall,
        "F-": neg_metrics.f1,
    }
