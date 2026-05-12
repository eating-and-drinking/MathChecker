from __future__ import annotations

import argparse
from pathlib import Path

from .core.constants import DATASET_ALL, SUPPORTED_DATASETS
from .pipeline.runner import download_data_command, evaluate_command, rerun_failed_command, run_command

STAGE1_TOOLS_CHOICES = ("none", "python", "logic", "both")
STAGE2_TOOLS_CHOICES = ("none", "triad")
STAGE2_STEP_TYPE_CHOICES = ("heuristic", "llm", "hybrid")
STAGE2_ROUTER_CHOICES = ("step-type", "learned", "learned-hybrid")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mathchecker", description="MathChecker reproduction CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download-data", help="Download the sampled datasets from GitHub.")
    download_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    download_parser.add_argument("--force", action="store_true", help="Redownload even if local files already exist.")

    run_parser = subparsers.add_parser("run", help="Run MathChecker inference.")
    run_parser.add_argument("--dataset", choices=[*SUPPORTED_DATASETS, DATASET_ALL], required=True)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    run_parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    run_parser.add_argument("--max-traces", type=int, default=None)
    run_parser.add_argument("--concurrency", type=int, default=1)
    run_parser.add_argument("--timeout", type=float, default=120.0)
    run_parser.add_argument("--stage1-tools", choices=STAGE1_TOOLS_CHOICES, default="none")
    run_parser.add_argument("--stage2-tools", choices=STAGE2_TOOLS_CHOICES, default="none")
    run_parser.add_argument(
        "--stage2-step-type-classifier",
        choices=STAGE2_STEP_TYPE_CHOICES,
        default="hybrid",
    )
    run_parser.add_argument(
        "--stage2-router",
        choices=STAGE2_ROUTER_CHOICES,
        default="step-type",
    )
    run_parser.add_argument("--stage2-router-model", default=None)
    run_parser.add_argument("--stage2-router-threshold", type=float, default=0.55)
    resume_group = run_parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    run_parser.set_defaults(resume=True)

    rerun_failed_parser = subparsers.add_parser(
        "rerun-failed",
        help="Rerun only traces that failed required stage tool-success checks.",
    )
    rerun_failed_parser.add_argument("--dataset", choices=[*SUPPORTED_DATASETS, DATASET_ALL], required=True)
    rerun_failed_parser.add_argument("--model", required=True)
    rerun_failed_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    rerun_failed_parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    rerun_failed_parser.add_argument("--concurrency", type=int, default=1)
    rerun_failed_parser.add_argument("--timeout", type=float, default=120.0)
    rerun_failed_parser.add_argument("--stage1-tools", choices=STAGE1_TOOLS_CHOICES, default="none")
    rerun_failed_parser.add_argument("--stage2-tools", choices=STAGE2_TOOLS_CHOICES, default="none")
    rerun_failed_parser.add_argument(
        "--stage2-step-type-classifier",
        choices=STAGE2_STEP_TYPE_CHOICES,
        default="hybrid",
    )
    rerun_failed_parser.add_argument(
        "--stage2-router",
        choices=STAGE2_ROUTER_CHOICES,
        default="step-type",
    )
    rerun_failed_parser.add_argument("--stage2-router-model", default=None)
    rerun_failed_parser.add_argument("--stage2-router-threshold", type=float, default=0.55)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate saved MathChecker predictions.")
    evaluate_parser.add_argument("--dataset", choices=[*SUPPORTED_DATASETS, DATASET_ALL], required=True)
    evaluate_parser.add_argument("--model", required=True)
    evaluate_parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    evaluate_parser.add_argument("--require-tool-success", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "download-data":
        paths = download_data_command(data_dir=args.data_dir, force=args.force)
        for path in paths:
            print(path)
        return 0

    if args.command == "run":
        results = run_command(
            dataset=args.dataset,
            model=args.model,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            max_traces=args.max_traces,
            resume=args.resume,
            concurrency=args.concurrency,
            timeout=args.timeout,
            stage1_tools=args.stage1_tools,
            stage2_tools=args.stage2_tools,
            stage2_step_type_classifier=args.stage2_step_type_classifier,
            stage2_router=args.stage2_router,
            stage2_router_model=args.stage2_router_model,
            stage2_router_threshold=args.stage2_router_threshold,
        )
        for dataset_name, predictions in results.items():
            completed = sum(1 for item in predictions if item.completed)
            print(f"{dataset_name}: {completed}/{len(predictions)} completed")
        return 0

    if args.command == "rerun-failed":
        results = rerun_failed_command(
            dataset=args.dataset,
            model=args.model,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            concurrency=args.concurrency,
            timeout=args.timeout,
            stage1_tools=args.stage1_tools,
            stage2_tools=args.stage2_tools,
            stage2_step_type_classifier=args.stage2_step_type_classifier,
            stage2_router=args.stage2_router,
            stage2_router_model=args.stage2_router_model,
            stage2_router_threshold=args.stage2_router_threshold,
        )
        for dataset_name, predictions in results.items():
            completed = sum(1 for item in predictions if item.completed)
            print(f"{dataset_name}: {completed}/{len(predictions)} completed after rerun-failed")
        return 0

    if args.command == "evaluate":
        summaries = evaluate_command(
            dataset=args.dataset,
            model=args.model,
            output_dir=args.output_dir,
            require_tool_success=args.require_tool_success,
        )
        for dataset_name, metrics in summaries.items():
            pretty = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
            print(f"{dataset_name}: {pretty}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
