from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mathchecker.data.jsonl import read_jsonl
from mathchecker.pipeline.router import LearnedRouterConfig, LearnedSpecialistRouter, build_route_decision_from_scores
from mathchecker.pipeline.router_eval import (
    binary_label_targets,
    build_route_ablation_summary,
    build_row_router_context,
    calibrate_per_label_thresholds,
    evaluate_router_predictions,
    heuristic_route_labels,
    imitation_route_labels,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate and ablate the learned specialist router.")
    parser.add_argument("--data-jsonl", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("heuristic", "imitation", "oracle", "learned", "learned-hybrid"),
        default="learned-hybrid",
    )
    parser.add_argument("--router-model", type=str, default=None)
    parser.add_argument(
        "--target-field",
        choices=("labels", "imitation_labels", "weak_labels", "policy_targets", "expected_gain_targets"),
        default="expected_gain_targets",
    )
    parser.add_argument("--binary-target-threshold", type=float, default=0.5)
    parser.add_argument("--router-threshold", type=float, default=0.55)
    parser.add_argument("--optimize-thresholds", action="store_true")
    parser.add_argument("--threshold-objective", choices=("utility", "f1"), default="utility")
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--save-thresholds-path", type=Path, default=None)
    parser.add_argument("--write-router-config", action="store_true")
    return parser


def _load_rows(path: Path) -> list[dict]:
    rows = [row for row in read_jsonl(path) if isinstance(row, dict)]
    if not rows:
        raise ValueError(f"No rows found in {path}.")
    return rows


def _build_predictions(
    rows: list[dict],
    *,
    mode: str,
    router_model: str | None,
    target_field: str,
    binary_target_threshold: float,
    router_threshold: float,
    optimize_thresholds: bool,
    threshold_objective: str,
) -> tuple[list[dict], dict[str, float] | None, dict[str, dict] | None]:
    if mode == "heuristic":
        return (
            [
                {
                    "labels": heuristic_route_labels(row),
                    "scores": {},
                    "confidence": None,
                    "fallback_used": False,
                    "source": "heuristic",
                }
                for row in rows
            ],
            None,
            None,
        )

    if mode == "imitation":
        return (
            [
                {
                    "labels": imitation_route_labels(row),
                    "scores": {},
                    "confidence": None,
                    "fallback_used": False,
                    "source": "imitation",
                }
                for row in rows
            ],
            None,
            None,
        )

    if mode == "oracle":
        return (
            [
                {
                    "labels": binary_label_targets(
                        row,
                        field=target_field,
                        threshold=binary_target_threshold,
                    ),
                    "scores": {},
                    "confidence": 1.0,
                    "fallback_used": False,
                    "source": "oracle",
                }
                for row in rows
            ],
            None,
            None,
        )

    if not router_model:
        raise ValueError("--router-model is required for learned router evaluation.")

    router = LearnedSpecialistRouter(
        LearnedRouterConfig(
            model_path=router_model,
            confidence_threshold=router_threshold,
        )
    )
    score_rows: list[dict[str, float]] = []
    for row in rows:
        score_result = router.score(build_row_router_context(row))
        if not score_result.success:
            raise RuntimeError(score_result.error or "Failed to score router rows.")
        score_rows.append(score_result.scores)

    per_label_thresholds = None
    threshold_report = None
    if optimize_thresholds:
        per_label_thresholds, threshold_report = calibrate_per_label_thresholds(
            rows,
            score_rows,
            target_field=target_field,
            objective=threshold_objective,
            binary_target_threshold=binary_target_threshold,
        )

    predictions: list[dict] = []
    for row, scores in zip(rows, score_rows, strict=True):
        learned_decision = build_route_decision_from_scores(
            scores,
            confidence_threshold=router_threshold,
            per_label_thresholds=per_label_thresholds,
            source="offline_eval_learned",
            model_name=router_model,
        )
        final_labels = {
            "use_alternative_route_verifier_tool": int(
                "alternative_route_verifier_tool" in set(learned_decision.selected_specialists)
            ),
            "use_equivalence_substitution_verifier_tool": int(
                "equivalence_substitution_verifier_tool" in set(learned_decision.selected_specialists)
            ),
            "use_condition_obligation_verifier_tool": int(
                "condition_obligation_verifier_tool" in set(learned_decision.selected_specialists)
            ),
            "trigger_specialist_review": int(learned_decision.trigger_specialist_review),
        }
        fallback_used = False
        source = "learned"
        if mode == "learned-hybrid" and (
            learned_decision.confidence is None or learned_decision.confidence < router_threshold
        ):
            final_labels = heuristic_route_labels(row)
            fallback_used = True
            source = "heuristic_fallback"

        predictions.append(
            {
                "labels": final_labels,
                "scores": scores,
                "confidence": learned_decision.confidence,
                "fallback_used": fallback_used,
                "source": source,
            }
        )

    return predictions, per_label_thresholds, threshold_report


def main() -> int:
    args = _build_parser().parse_args()
    rows = _load_rows(args.data_jsonl)
    predictions, per_label_thresholds, threshold_report = _build_predictions(
        rows,
        mode=args.mode,
        router_model=args.router_model,
        target_field=args.target_field,
        binary_target_threshold=args.binary_target_threshold,
        router_threshold=args.router_threshold,
        optimize_thresholds=args.optimize_thresholds,
        threshold_objective=args.threshold_objective,
    )
    metrics = evaluate_router_predictions(
        rows,
        predictions,
        target_field=args.target_field,
        binary_target_threshold=args.binary_target_threshold,
        confidence_threshold=args.router_threshold,
    )
    ablation = build_route_ablation_summary(
        rows,
        predictions,
        target_field=args.target_field,
        binary_target_threshold=args.binary_target_threshold,
        confidence_threshold=args.router_threshold,
    )
    summary = {
        "mode": args.mode,
        "target_field": args.target_field,
        "router_model": args.router_model,
        "router_threshold": args.router_threshold,
        "binary_target_threshold": args.binary_target_threshold,
        "optimized_per_label_thresholds": per_label_thresholds,
        "threshold_search_report": threshold_report,
        "metrics": metrics,
        "ablation": ablation,
    }

    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(rendered + "\n", encoding="utf-8")

    if args.save_thresholds_path is not None and per_label_thresholds is not None:
        threshold_payload = {
            "per_label_thresholds": per_label_thresholds,
            "objective": args.threshold_objective,
            "router_threshold": args.router_threshold,
            "target_field": args.target_field,
            "report": threshold_report,
        }
        args.save_thresholds_path.parent.mkdir(parents=True, exist_ok=True)
        args.save_thresholds_path.write_text(
            json.dumps(threshold_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if args.write_router_config and per_label_thresholds is not None:
        if not args.router_model:
            raise ValueError("--write-router-config requires --router-model.")
        config_path = Path(args.router_model) / "mathchecker_router_config.json"
        existing: dict = {}
        if config_path.exists():
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        existing["per_label_thresholds"] = per_label_thresholds
        existing["threshold_objective"] = args.threshold_objective
        existing["target_field"] = args.target_field
        existing["confidence_threshold"] = args.router_threshold
        config_path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
