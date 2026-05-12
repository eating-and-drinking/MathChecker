from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pedcot.data.jsonl import read_jsonl
from pedcot.pipeline.router import ROUTER_LABELS, RouterContext, format_router_input_text
from pedcot.utils import ensure_parent_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the first learned specialist router with Qwen3-0.6B.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--num-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--router-threshold", type=float, default=0.55)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--disable-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--save-merged-model", action="store_true")
    parser.add_argument("--disable-sample-weights", action="store_true")
    parser.add_argument(
        "--target-field",
        choices=("labels", "imitation_labels", "weak_labels", "policy_targets", "expected_gain_targets"),
        default="labels",
    )
    return parser


def _row_targets(row: dict[str, Any], target_field: str) -> dict[str, float] | None:
    candidates = [target_field]
    if target_field != "labels":
        candidates.append("labels")
    for field_name in candidates:
        payload = row.get(field_name)
        if not isinstance(payload, dict):
            continue
        if not all(label in payload for label in ROUTER_LABELS):
            continue
        targets: dict[str, float] = {}
        valid = True
        for label in ROUTER_LABELS:
            value = payload.get(label)
            if not isinstance(value, (int, float)):
                valid = False
                break
            targets[label] = float(value)
        if valid:
            return targets
    return None


def _load_training_rows(path: Path, target_field: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        if not isinstance(row, dict):
            continue
        resolved_targets = _row_targets(row, target_field)
        if resolved_targets is None:
            continue
        normalized_row = dict(row)
        normalized_row["_resolved_targets"] = resolved_targets
        rows.append(normalized_row)
    if not rows:
        raise ValueError(f"No valid router training rows found in {path}.")
    return rows


def _split_rows(rows: list[dict[str, Any]], ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if ratio <= 0.0 or len(rows) < 2:
        return rows, []
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    eval_size = max(1, int(len(shuffled) * ratio))
    if eval_size >= len(shuffled):
        eval_size = len(shuffled) - 1
    return shuffled[eval_size:], shuffled[:eval_size]


def _build_text(row: dict[str, Any]) -> str:
    text = row.get("text")
    if isinstance(text, str) and text.strip():
        return text

    previous_steps = row.get("previous_steps", [])
    risk_flags = row.get("heuristic_risk_flags", [])
    heuristic_specialists = row.get("heuristic_specialists", [])
    if not isinstance(previous_steps, list):
        previous_steps = []
    if not isinstance(risk_flags, list):
        risk_flags = []
    if not isinstance(heuristic_specialists, list):
        heuristic_specialists = []
    context = RouterContext(
        dataset=str(row.get("dataset", "")),
        question=str(row.get("question", "")),
        previous_steps=tuple(str(item) for item in previous_steps),
        current_step=str(row.get("current_step", "")),
        heuristic_step_type=str(row.get("heuristic_step_type", "reasoning_transition")),
        heuristic_risk_flags=tuple(str(item) for item in risk_flags),
        heuristic_specialists=tuple(str(item) for item in heuristic_specialists),
        stage1_mathematical_concepts=(
            str(row.get("stage1_mathematical_concepts"))
            if isinstance(row.get("stage1_mathematical_concepts"), str)
            else None
        ),
        stage1_key_analyses=(
            str(row.get("stage1_key_analyses"))
            if isinstance(row.get("stage1_key_analyses"), str)
            else None
        ),
        stage1_calculations=(
            str(row.get("stage1_calculations"))
            if isinstance(row.get("stage1_calculations"), str)
            else None
        ),
    )
    return format_router_input_text(context)


def main() -> int:
    args = _build_parser().parse_args()
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from torch import nn
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Router training requires optional dependencies. Run `uv sync --group router` first."
        ) from exc

    train_rows = _load_training_rows(args.train_jsonl, args.target_field)
    eval_rows = _load_training_rows(args.eval_jsonl, args.target_field) if args.eval_jsonl is not None else []
    if not eval_rows:
        train_rows, eval_rows = _split_rows(train_rows, args.validation_ratio, args.seed)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    label2id = {label: index for index, label in enumerate(ROUTER_LABELS)}
    id2label = {index: label for label, index in label2id.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(ROUTER_LABELS),
        label2id=label2id,
        id2label=id2label,
        problem_type="multi_label_classification",
        trust_remote_code=True,
    )
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if not args.disable_lora:
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                modules_to_save=["score"],
            ),
        )

    class RouterDataset(torch.utils.data.Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self._features: list[dict[str, Any]] = []
            for row in rows:
                encoded = tokenizer(
                    _build_text(row),
                    truncation=True,
                    max_length=args.max_length,
                )
                encoded["labels"] = [float(row["_resolved_targets"][label]) for label in ROUTER_LABELS]
                encoded["sample_weight"] = float(row.get("sample_weight", 1.0))
                self._features.append(encoded)

        def __len__(self) -> int:
            return len(self._features)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return self._features[index]

    class RouterCollator:
        def __init__(self) -> None:
            self._base = DataCollatorWithPadding(tokenizer=tokenizer)

        def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
            labels = [feature.pop("labels") for feature in features]
            sample_weights = [feature.pop("sample_weight", 1.0) for feature in features]
            batch = self._base(features)
            batch["labels"] = torch.tensor(labels, dtype=torch.float32)
            batch["sample_weight"] = torch.tensor(sample_weights, dtype=torch.float32)
            return batch

    class WeightedMultiLabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            sample_weight = inputs.pop("sample_weight", None)
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.BCEWithLogitsLoss(reduction="none")
            loss_matrix = loss_fct(logits, labels)
            per_example_loss = loss_matrix.mean(dim=-1)
            if sample_weight is not None and not args.disable_sample_weights:
                per_example_loss = per_example_loss * sample_weight.to(per_example_loss.device)
            loss = per_example_loss.mean()
            if return_outputs:
                return loss, outputs
            return loss

    def compute_metrics(eval_pred) -> dict[str, float]:
        import numpy as np

        logits, labels = eval_pred
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        predictions = probabilities >= args.router_threshold
        gold = labels >= 0.5

        tp = float(np.logical_and(predictions, gold).sum())
        fp = float(np.logical_and(predictions, np.logical_not(gold)).sum())
        fn = float(np.logical_and(np.logical_not(predictions), gold).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        micro_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        exact_match = float(np.all(predictions == gold, axis=1).mean())
        avg_policy_utility = float(np.where(predictions, labels, 1.0 - labels).mean())
        positive_rate = float(predictions.mean())
        selected_targets = labels[predictions]
        avg_selected_target = float(selected_targets.mean()) if selected_targets.size else 0.0

        per_label_f1: dict[str, float] = {}
        for index, label in enumerate(ROUTER_LABELS):
            label_pred = predictions[:, index]
            label_gold = gold[:, index]
            label_tp = float(np.logical_and(label_pred, label_gold).sum())
            label_fp = float(np.logical_and(label_pred, np.logical_not(label_gold)).sum())
            label_fn = float(np.logical_and(np.logical_not(label_pred), label_gold).sum())
            label_precision = label_tp / (label_tp + label_fp) if (label_tp + label_fp) else 0.0
            label_recall = label_tp / (label_tp + label_fn) if (label_tp + label_fn) else 0.0
            per_label_f1[f"{label}_f1"] = (
                2 * label_precision * label_recall / (label_precision + label_recall)
                if (label_precision + label_recall)
                else 0.0
            )

        return {
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": micro_f1,
            "exact_match": exact_match,
            "avg_policy_utility": avg_policy_utility,
            "positive_rate": positive_rate,
            "avg_selected_target": avg_selected_target,
            **per_label_f1,
        }

    train_dataset = RouterDataset(train_rows)
    eval_dataset = RouterDataset(eval_rows) if eval_rows else None

    ensure_parent_dir(args.output_dir / "placeholder.txt")
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        evaluation_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch" if eval_dataset is not None else "no",
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = WeightedMultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=RouterCollator(),
        compute_metrics=compute_metrics if eval_dataset is not None else None,
    )
    trainer.train()
    final_eval_metrics = trainer.evaluate() if eval_dataset is not None else {}
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    router_config = {
        "router_type": "qwen3_multilabel_sequence_classifier",
        "base_model": args.base_model,
        "label_order": list(ROUTER_LABELS),
        "confidence_threshold": args.router_threshold,
        "max_length": args.max_length,
        "uses_lora": not args.disable_lora,
        "target_field": args.target_field,
    }
    (args.output_dir / "pedcot_router_config.json").write_text(
        json.dumps(router_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if final_eval_metrics:
        (args.output_dir / "router_eval_metrics.json").write_text(
            json.dumps(final_eval_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if args.save_merged_model and not args.disable_lora and hasattr(model, "merge_and_unload"):
        merged_dir = args.output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        (merged_dir / "pedcot_router_config.json").write_text(
            json.dumps(router_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"trained router saved to {args.output_dir}")
    print(
        f"train_rows={len(train_rows)} eval_rows={len(eval_rows)} "
        f"labels={list(ROUTER_LABELS)} target_field={args.target_field}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
