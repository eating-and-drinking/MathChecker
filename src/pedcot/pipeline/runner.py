from __future__ import annotations

import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..core.constants import DATASET_ALL, PAPER_DATASETS, SUPPORTED_DATASETS
from ..data.datasets import download_all_datasets, load_dataset
from ..data.jsonl import append_jsonl
from ..evaluation.metrics import compute_metrics
from ..llm.openai_client import OpenAIResponsesClient
from .predictor import PedCoTPredictor
from ..data.stores import PredictionStore, StageCacheStore
from ..core.models import TraceExample, TracePrediction
from ..utils import ensure_parent_dir


def _sanitize_for_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def prediction_file_path(output_dir: Path, dataset: str, model: str) -> Path:
    return output_dir / "predictions" / f"{dataset}__{_sanitize_for_filename(model)}.jsonl"


def stage_cache_file_path(output_dir: Path) -> Path:
    return output_dir / "cache" / "stage_cache.jsonl"


def failed_tool_steps_file_path(output_dir: Path, dataset: str, model: str) -> Path:
    return output_dir / "failures" / f"{dataset}__{_sanitize_for_filename(model)}__tool_failures.jsonl"


def degradation_file_path(output_dir: Path, dataset: str, model: str) -> Path:
    return output_dir / "degradations" / f"{dataset}__{_sanitize_for_filename(model)}__degradations.jsonl"


def cache_degradation_file_path(output_dir: Path) -> Path:
    return output_dir / "cache" / "degradation_events.jsonl"


def metrics_json_path(output_dir: Path, dataset: str, model: str) -> Path:
    return output_dir / "metrics" / f"{dataset}__{_sanitize_for_filename(model)}.json"


def metrics_csv_path(output_dir: Path, dataset: str, model: str) -> Path:
    return output_dir / "metrics" / f"{dataset}__{_sanitize_for_filename(model)}.csv"


def resolve_datasets(dataset: str) -> list[str]:
    if dataset == DATASET_ALL:
        return list(PAPER_DATASETS)
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return [dataset]


def _error_prediction(example: TraceExample, model: str, error: Exception) -> TracePrediction:
    return TracePrediction(
        example_id=example.example_id,
        dataset=example.dataset,
        model=model,
        pred_first_mistake_index=None,
        pred_trace_label=None,
        gold_first_mistake_index=example.gold_first_mistake_index,
        gold_trace_label=example.gold_trace_label,
        steps=[],
        completed=False,
        error=str(error),
    )


def _in_progress_prediction(example: TraceExample, model: str) -> TracePrediction:
    return TracePrediction(
        example_id=example.example_id,
        dataset=example.dataset,
        model=model,
        pred_first_mistake_index=None,
        pred_trace_label=None,
        gold_first_mistake_index=example.gold_first_mistake_index,
        gold_trace_label=example.gold_trace_label,
        steps=[],
        completed=False,
        error="in_progress",
    )


_RETRYABLE_EXCEPTION_MARKERS = (
    "timed out",
    "timeout",
    "connection",
    "temporarily unavailable",
    "rate limit",
    "status code: 429",
    "status code: 500",
    "status code: 502",
    "status code: 503",
    "status code: 504",
    "request processing failed",
    "service unavailable",
)

_NON_RETRYABLE_EXCEPTION_MARKERS = (
    "authenticationerror",
    "api key",
    "unauthorized",
    "forbidden",
    "incorrect model id",
    "permission",
    "invalid_request_error",
)

_RETRYABLE_PREDICTION_ERROR_MARKERS = (
    "could not split stage-1 response into three sections.",
    "could not split stage-2 response into three sections.",
    "missing one or more principle labels in stage-2 response.",
    "stage-1 required tools failed after retries.",
    "stage-2 required tools failed after retries.",
)

def _is_retryable_exception(exc: Exception) -> bool:
    message = str(exc).lower()
    if any(marker in message for marker in _NON_RETRYABLE_EXCEPTION_MARKERS):
        return False
    return any(marker in message for marker in _RETRYABLE_EXCEPTION_MARKERS)


def _is_retryable_prediction_error(error: str | None) -> bool:
    if not error:
        return False
    message = error.lower()
    return any(marker in message for marker in _RETRYABLE_PREDICTION_ERROR_MARKERS)


def _retry_delay_seconds(retry_index: int, *, ceiling: float) -> float:
    return min(ceiling, float(2 ** min(retry_index - 1, 6)))


def _step_status(step: dict[str, Any]) -> dict[str, Any]:
    status = step.get("parse_status")
    return status if isinstance(status, dict) else {}


def _step_requires_stage1_tool_success(step: dict[str, Any]) -> bool:
    status = _step_status(step)
    mode = status.get("stage1_tools_mode")
    mode_requires = isinstance(mode, str) and mode in {"python", "logic", "both"}
    explicit_requires = bool(status.get("stage1_tool_required", False))
    return explicit_requires or mode_requires


def _step_has_stage1_tool_success(step: dict[str, Any]) -> bool:
    status = _step_status(step)
    return bool(status.get("stage1_tool_success", False))


def _step_requires_stage2_tool_success(step: dict[str, Any]) -> bool:
    status = _step_status(step)
    mode = status.get("stage2_tools_mode")
    mode_requires = isinstance(mode, str) and mode in {"triad"}
    explicit_requires = bool(status.get("stage2_tool_required", False))
    return explicit_requires or mode_requires


def _step_has_stage2_tool_success(step: dict[str, Any]) -> bool:
    status = _step_status(step)
    return bool(status.get("stage2_tool_success", False))


def _prediction_has_full_tool_success(prediction: TracePrediction) -> bool:
    if not prediction.completed:
        return False
    for step in prediction.steps:
        if _step_requires_stage1_tool_success(step) and not _step_has_stage1_tool_success(step):
            return False
        if _step_requires_stage2_tool_success(step) and not _step_has_stage2_tool_success(step):
            return False
    return True


def _append_tool_failures(
    *,
    output_dir: Path,
    dataset: str,
    model: str,
    prediction: TracePrediction,
    stage1_tools: str,
    stage2_tools: str,
) -> None:
    if stage1_tools == "none" and stage2_tools == "none":
        return
    path = failed_tool_steps_file_path(output_dir, dataset, model)
    for step in prediction.steps:
        status = _step_status(step)
        if _step_requires_stage1_tool_success(step) and not _step_has_stage1_tool_success(step):
            append_jsonl(
                path,
                {
                    "dataset": prediction.dataset,
                    "model": prediction.model,
                    "example_id": prediction.example_id,
                    "step_index": step.get("step_index"),
                    "stage": "stage1",
                    "completed": prediction.completed,
                    "prediction_error": prediction.error,
                    "stage1_tools_mode": status.get("stage1_tools_mode", stage1_tools),
                    "required_tool_names": status.get("stage1_tool_required_names", []),
                    "called_tool_names": status.get("stage1_tool_called_names", []),
                    "missing_tool_names": status.get("stage1_tool_missing_names", []),
                    "tool_error_count": status.get("stage1_tool_error_count", 0),
                    "tool_errors": step.get("stage1_tool_errors", []),
                    "tool_trace": step.get("stage1_tool_trace", []),
                },
            )
        if _step_requires_stage2_tool_success(step) and not _step_has_stage2_tool_success(step):
            required_stage2 = set(status.get("stage2_tool_required_names", []) or [])
            called_stage2 = set(status.get("stage2_tool_called_names", []) or [])
            append_jsonl(
                path,
                {
                    "dataset": prediction.dataset,
                    "model": prediction.model,
                    "example_id": prediction.example_id,
                    "step_index": step.get("step_index"),
                    "stage": "stage2",
                    "completed": prediction.completed,
                    "prediction_error": prediction.error,
                    "stage2_tools_mode": status.get("stage2_tools_mode", stage2_tools),
                    "required_tool_names": sorted(required_stage2),
                    "called_tool_names": sorted(called_stage2),
                    "missing_tool_names": sorted(required_stage2 - called_stage2),
                    "tool_error_count": status.get("stage2_tool_error_count", 0),
                    "tool_errors": step.get("stage2_tool_errors", []),
                    "tool_trace": step.get("stage2_tool_trace", []),
                },
            )


def _append_degradation_events(
    *,
    output_dir: Path,
    dataset: str,
    model: str,
    prediction: TracePrediction,
) -> None:
    report_path = degradation_file_path(output_dir, dataset, model)
    cache_path = cache_degradation_file_path(output_dir)
    event_time = int(time.time())

    for step in prediction.steps:
        status = _step_status(step)
        step_index = step.get("step_index")

        if bool(status.get("stage1_tool_degraded", False)):
            row = {
                "timestamp": event_time,
                "dataset": prediction.dataset,
                "model": prediction.model,
                "example_id": prediction.example_id,
                "step_index": step_index,
                "stage": "stage1",
                "degraded_to_no_tool": True,
                "degrade_reason": status.get("stage1_tool_degrade_reason", "unknown"),
                "failed_tools": status.get("stage1_tool_degrade_failed_tools", []),
                "fallback_success": bool(status.get("stage1_tool_degrade_fallback_success", False)),
                "prediction_completed": prediction.completed,
                "prediction_error": prediction.error,
            }
            append_jsonl(report_path, row)
            append_jsonl(cache_path, row)

        if bool(status.get("stage2_tool_degraded", False)):
            row = {
                "timestamp": event_time,
                "dataset": prediction.dataset,
                "model": prediction.model,
                "example_id": prediction.example_id,
                "step_index": step_index,
                "stage": "stage2",
                "degraded_to_no_tool": True,
                "degrade_reason": status.get("stage2_tool_degrade_reason", "unknown"),
                "failed_tools": status.get("stage2_tool_degrade_failed_tools", []),
                "fallback_success": bool(status.get("stage2_tool_degrade_fallback_success", False)),
                "prediction_completed": prediction.completed,
                "prediction_error": prediction.error,
            }
            append_jsonl(report_path, row)
            append_jsonl(cache_path, row)



def run_dataset(
    *,
    dataset: str,
    model: str,
    data_dir: Path,
    output_dir: Path,
    max_traces: int | None,
    resume: bool,
    concurrency: int,
    timeout: float,
    stage1_tools: str = "none",
    stage2_tools: str = "none",
    stage2_step_type_classifier: str = "hybrid",
    stage2_router: str = "step-type",
    stage2_router_model: str | None = None,
    stage2_router_threshold: float = 0.55,
    only_example_ids: set[str] | None = None,
) -> list[TracePrediction]:
    examples = load_dataset(dataset, data_dir=data_dir)
    if only_example_ids is not None:
        examples = [example for example in examples if example.example_id in only_example_ids]
    if max_traces is not None:
        examples = examples[:max_traces]

    cache_store = StageCacheStore(stage_cache_file_path(output_dir))
    store = PredictionStore(prediction_file_path(output_dir, dataset, model))

    pending_examples = [
        example
        for example in examples
        if not (resume and store.has_completed(example.dataset, example.example_id, model))
    ]

    if not pending_examples:
        return store.load_predictions(dataset=dataset, model=model)

    client = OpenAIResponsesClient(timeout=timeout)
    validate_configuration = getattr(client, "validate_configuration", None)
    if callable(validate_configuration):
        validate_configuration()
    predictor = PedCoTPredictor(
        client=client,
        cache_store=cache_store,
        stage1_tools_mode=stage1_tools,
        stage2_tools_mode=stage2_tools,
        stage2_step_type_mode=stage2_step_type_classifier,
        stage2_router_mode=stage2_router,
        stage2_router_model_path=stage2_router_model,
        stage2_router_confidence_threshold=stage2_router_threshold,
    )

    def worker(example: TraceExample) -> TracePrediction:
        retry_count = 0
        while True:
            try:
                prediction = predictor.predict_trace(example=example, model=model)
            except Exception as exc:  # noqa: BLE001
                if _is_retryable_exception(exc):
                    retry_count += 1
                    delay_seconds = _retry_delay_seconds(retry_count, ceiling=60.0)
                    tqdm.write(
                        f"[retry {retry_count}] {example.example_id}: transient request error: {exc}; "
                        f"sleeping {delay_seconds:.0f}s before retry"
                    )
                    time.sleep(delay_seconds)
                    continue
                return _error_prediction(example, model, exc)

            if prediction.completed:
                return prediction

            if _is_retryable_prediction_error(prediction.error):
                retry_count += 1
                delay_seconds = _retry_delay_seconds(retry_count, ceiling=30.0)
                tqdm.write(
                    f"[retry {retry_count}] {example.example_id}: retryable parse error: {prediction.error}; "
                    f"sleeping {delay_seconds:.0f}s before retry"
                )
                time.sleep(delay_seconds)
                continue

            return prediction

    if pending_examples:
        with tqdm(total=len(pending_examples), desc=f"{dataset}:{model}", unit="trace") as progress:
            if concurrency <= 1:
                for example in pending_examples:
                    store.append(_in_progress_prediction(example, model))
                    prediction = worker(example)
                    store.append(prediction)
                    _append_tool_failures(
                        output_dir=output_dir,
                        dataset=dataset,
                        model=model,
                        prediction=prediction,
                        stage1_tools=stage1_tools,
                        stage2_tools=stage2_tools,
                    )
                    _append_degradation_events(
                        output_dir=output_dir,
                        dataset=dataset,
                        model=model,
                        prediction=prediction,
                    )
                    progress.update(1)
            else:
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    future_map = {}
                    for example in pending_examples:
                        store.append(_in_progress_prediction(example, model))
                        future_map[executor.submit(worker, example)] = example
                    for future in as_completed(future_map):
                        prediction = future.result()
                        store.append(prediction)
                        _append_tool_failures(
                            output_dir=output_dir,
                            dataset=dataset,
                            model=model,
                            prediction=prediction,
                            stage1_tools=stage1_tools,
                            stage2_tools=stage2_tools,
                        )
                        _append_degradation_events(
                            output_dir=output_dir,
                            dataset=dataset,
                            model=model,
                            prediction=prediction,
                        )
                        progress.update(1)
    return store.load_predictions(dataset=dataset, model=model)


def run_command(
    *,
    dataset: str,
    model: str,
    data_dir: Path,
    output_dir: Path,
    max_traces: int | None,
    resume: bool,
    concurrency: int,
    timeout: float,
    stage1_tools: str = "none",
    stage2_tools: str = "none",
    stage2_step_type_classifier: str = "hybrid",
    stage2_router: str = "step-type",
    stage2_router_model: str | None = None,
    stage2_router_threshold: float = 0.55,
    only_example_ids_by_dataset: dict[str, set[str]] | None = None,
) -> dict[str, list[TracePrediction]]:
    results: dict[str, list[TracePrediction]] = {}
    for dataset_name in resolve_datasets(dataset):
        only_example_ids = None
        if only_example_ids_by_dataset is not None:
            only_example_ids = only_example_ids_by_dataset.get(dataset_name, set())
        results[dataset_name] = run_dataset(
            dataset=dataset_name,
            model=model,
            data_dir=data_dir,
            output_dir=output_dir,
            max_traces=max_traces,
            resume=resume,
            concurrency=concurrency,
            timeout=timeout,
            stage1_tools=stage1_tools,
            stage2_tools=stage2_tools,
            stage2_step_type_classifier=stage2_step_type_classifier,
            stage2_router=stage2_router,
            stage2_router_model=stage2_router_model,
            stage2_router_threshold=stage2_router_threshold,
            only_example_ids=only_example_ids,
        )
    return results


def evaluate_command(
    *,
    dataset: str,
    model: str,
    output_dir: Path,
    require_tool_success: bool = False,
) -> dict[str, dict[str, float]]:
    summaries: dict[str, dict[str, float]] = {}
    for dataset_name in resolve_datasets(dataset):
        store = PredictionStore(prediction_file_path(output_dir, dataset_name, model))
        predictions = store.load_predictions(dataset=dataset_name, model=model)
        total_count = len(predictions)
        if require_tool_success:
            predictions = [prediction for prediction in predictions if _prediction_has_full_tool_success(prediction)]
            if not predictions:
                raise ValueError(
                    f"No predictions satisfy full stage tool success for dataset={dataset_name}, model={model}."
                )
        metrics = compute_metrics(predictions)
        metrics["Num_Total"] = float(total_count)
        metrics["Num_Evaluated"] = float(len(predictions))
        metrics["Tool_Coverage"] = float(len(predictions) / total_count) if total_count else 0.0
        metrics["Require_Tool_Success"] = 1.0 if require_tool_success else 0.0
        summaries[dataset_name] = metrics
        _write_metrics_files(output_dir=output_dir, dataset=dataset_name, model=model, metrics=metrics)
    return summaries


def _write_metrics_files(*, output_dir: Path, dataset: str, model: str, metrics: dict[str, float]) -> None:
    json_path = metrics_json_path(output_dir, dataset, model)
    csv_path = metrics_csv_path(output_dir, dataset, model)
    ensure_parent_dir(json_path)
    ensure_parent_dir(csv_path)

    payload = {"dataset": dataset, "model": model, **metrics}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload.keys()))
        writer.writeheader()
        writer.writerow(payload)


def download_data_command(*, data_dir: Path, force: bool) -> list[Path]:
    return download_all_datasets(data_dir=data_dir, force=force)


def rerun_failed_command(
    *,
    dataset: str,
    model: str,
    data_dir: Path,
    output_dir: Path,
    concurrency: int,
    timeout: float,
    stage1_tools: str = "none",
    stage2_tools: str = "none",
    stage2_step_type_classifier: str = "hybrid",
    stage2_router: str = "step-type",
    stage2_router_model: str | None = None,
    stage2_router_threshold: float = 0.55,
) -> dict[str, list[TracePrediction]]:
    rerun_ids: dict[str, set[str]] = {}
    for dataset_name in resolve_datasets(dataset):
        store = PredictionStore(prediction_file_path(output_dir, dataset_name, model))
        predictions = store.load_predictions(dataset=dataset_name, model=model)
        failed_ids = {
            prediction.example_id for prediction in predictions if not _prediction_has_full_tool_success(prediction)
        }
        if failed_ids:
            rerun_ids[dataset_name] = failed_ids

    if not rerun_ids:
        results: dict[str, list[TracePrediction]] = {}
        for dataset_name in resolve_datasets(dataset):
            store = PredictionStore(prediction_file_path(output_dir, dataset_name, model))
            results[dataset_name] = store.load_predictions(dataset=dataset_name, model=model)
        return results

    return run_command(
        dataset=dataset,
        model=model,
        data_dir=data_dir,
        output_dir=output_dir,
        max_traces=None,
        resume=False,
        concurrency=concurrency,
        timeout=timeout,
        stage1_tools=stage1_tools,
        stage2_tools=stage2_tools,
        stage2_step_type_classifier=stage2_step_type_classifier,
        stage2_router=stage2_router,
        stage2_router_model=stage2_router_model,
        stage2_router_threshold=stage2_router_threshold,
        only_example_ids_by_dataset=rerun_ids,
    )

