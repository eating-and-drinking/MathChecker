from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pedcot.data.datasets import load_dataset
from pedcot.data.jsonl import read_jsonl
from pedcot.pipeline.router_dataset import build_router_export_row
from pedcot.utils import write_jsonl


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export router training data for the learned specialist router.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--export-path", type=Path, required=True)
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument(
        "--label-strategy",
        choices=("imitation", "weak-supervision", "benefit-aware", "expected-gain"),
        default="expected-gain",
    )
    return parser


def _sanitize_for_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _prediction_file_path(output_dir: Path, dataset: str, model: str) -> Path:
    return output_dir / "predictions" / f"{dataset}__{_sanitize_for_filename(model)}.jsonl"


def main() -> int:
    args = _build_parser().parse_args()

    dataset_examples = load_dataset(args.dataset, data_dir=args.data_dir)
    example_by_id = {example.example_id: example for example in dataset_examples}
    prediction_path = _prediction_file_path(args.output_dir, args.dataset, args.model)
    rows: list[dict] = []

    for prediction in read_jsonl(prediction_path):
        if not args.include_incomplete and not prediction.get("completed", False):
            continue
        example_id = prediction.get("example_id")
        if not isinstance(example_id, str) or example_id not in example_by_id:
            continue
        example = example_by_id[example_id]
        steps = prediction.get("steps", [])
        if not isinstance(steps, list):
            continue

        for step in steps:
            if not isinstance(step, dict):
                continue
            row = build_router_export_row(
                example,
                step,
                label_strategy=args.label_strategy,
            )
            if row is not None:
                rows.append(row)

    write_jsonl(args.export_path, rows)
    print(f"exported {len(rows)} rows to {args.export_path} with label_strategy={args.label_strategy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
