#!/usr/bin/env python3
"""Calibrate PRISM likelihood parameters on a hold-out predictions set.

Replaces the hand-coded sensitivities and label->mistake-rate table in
likelihoods.py with empirically-fitted values from real trace predictions
that include gold tau (gold_first_mistake_index).

What we fit
-----------

1. **Per-label empirical mistake rate**. For each principle-label class
   (correct-and-aligned, reasonable-but-incomplete, nothing-extracted,
   contradiction-found), count how often a step bearing that label was
   actually the first-mistake step (or at-or-after it). Used in place of
   the hard-coded _LABEL_TO_MISTAKE_PROB.

2. **Per-specialist isotonic regression**. For each specialist tool, fit
   an isotonic regression mapping the raw `hard_conflict_strength` it
   reports to the empirical probability that the step really was the
   first mistake. This is the canonical calibration trick that turns
   well-ordered-but-mis-scaled scores into proper probabilities.

3. **Per-channel sensitivity**. For each of {stage1, stage2, specialist},
   pick the scalar sensitivity in [0, 1] that minimizes negative
   log-likelihood of gold tau under the channel's likelihood. One scalar
   per channel keeps it identifiable on small calibration sets.

Output schema
-------------

artifacts/prism_likelihoods.json:
    {
      "version": 1,
      "label_mistake_prob": {label: float},
      "specialist_isotonic": {
        tool_name: {"x": [...], "y": [...]}   # piecewise constant breakpoints
      },
      "sensitivities": {
        "stage1": float, "stage2": float, "specialist": float
      },
      "calibration_size": int
    }

This file is loaded by `mathchecker.prism.likelihoods.load_calibration()` at
import time when present, and falls back to the hand-coded defaults when
missing.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mathchecker.prism.likelihoods import (  # noqa: E402
    make_specialist_likelihood,
    make_stage1_likelihood,
    make_stage2_likelihood,
)
from mathchecker.prism.stage1_consistency import extract_stage1_inconsistency  # noqa: E402


PRINCIPLE_LABELS = (
    "correct-and-aligned",
    "reasonable-but-incomplete",
    "nothing-extracted",
    "contradiction-found",
)

SPECIALIST_TOOLS = (
    "alternative_route_verifier_tool",
    "equivalence_substitution_verifier_tool",
    "condition_obligation_verifier_tool",
)


def _iter_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                yield rec


def _step_is_first_mistake(step_index: int, gold_tau: int | None) -> bool:
    if gold_tau is None:
        return False
    return step_index == gold_tau


def _fit_label_mistake_prob(records: list[dict]) -> dict[str, float]:
    """For each principle-label class, return empirical P(step is first mistake | label)."""
    pos = Counter()
    total = Counter()
    for prediction in records:
        if not prediction.get("completed"):
            continue
        gold_tau = prediction.get("gold_first_mistake_index")
        for step in prediction.get("steps", []) or []:
            if step.get("prism_meta"):
                continue
            step_index = step.get("step_index")
            if not isinstance(step_index, int) or step_index < 0:
                continue
            stage2 = step.get("stage2_parse") or {}
            labels = stage2.get("principle_labels") or {}
            is_first_mistake = _step_is_first_mistake(step_index, gold_tau)
            for value in labels.values():
                if not isinstance(value, str):
                    continue
                key = value.strip().lower()
                if key not in PRINCIPLE_LABELS:
                    continue
                total[key] += 1
                if is_first_mistake:
                    pos[key] += 1
    out: dict[str, float] = {}
    # Laplace-smooth lightly to keep extremes bounded.
    for label in PRINCIPLE_LABELS:
        n = total[label]
        k = pos[label]
        out[label] = (k + 1.0) / (n + 5.0)
    return out


def _fit_specialist_isotonic(records: list[dict]) -> dict[str, dict[str, list[float]]]:
    """Per specialist, fit isotonic regression on (hard_conflict, is_first_mistake)."""
    paired: dict[str, list[tuple[float, int]]] = {name: [] for name in SPECIALIST_TOOLS}
    for prediction in records:
        if not prediction.get("completed"):
            continue
        gold_tau = prediction.get("gold_first_mistake_index")
        for step in prediction.get("steps", []) or []:
            if step.get("prism_meta"):
                continue
            step_index = step.get("step_index")
            if not isinstance(step_index, int) or step_index < 0:
                continue
            is_first_mistake = int(_step_is_first_mistake(step_index, gold_tau))
            for item in step.get("stage2_tool_trace") or []:
                if not isinstance(item, dict) or item.get("discarded"):
                    continue
                name = item.get("tool_name")
                result = item.get("result")
                if not isinstance(name, str) or name not in SPECIALIST_TOOLS:
                    continue
                if not isinstance(result, dict):
                    continue
                # Re-derive hard_conflict_strength using same heuristic as predictor.
                hard = 0.10
                if result.get("hard_contradiction") or result.get("contradiction_level") == "hard_contradiction":
                    hard = 0.92
                elif isinstance(result.get("contradictions"), list) and result["contradictions"]:
                    hard = 0.55
                paired[name].append((hard, is_first_mistake))

    out: dict[str, dict[str, list[float]]] = {}
    for name, pairs in paired.items():
        if not pairs:
            out[name] = {"x": [0.0, 1.0], "y": [0.05, 0.5]}  # uninformative default
            continue
        out[name] = _pool_adjacent_violators(pairs)
    return out


def _pool_adjacent_violators(pairs: list[tuple[float, int]]) -> dict[str, list[float]]:
    """Classic PAV isotonic regression. Returns piecewise-constant {x, y}."""
    pairs = sorted(pairs, key=lambda p: p[0])
    xs = [p[0] for p in pairs]
    ys = [float(p[1]) for p in pairs]
    weights = [1.0] * len(pairs)

    # Pool-adjacent-violators in place.
    i = 0
    while i < len(ys) - 1:
        if ys[i] <= ys[i + 1]:
            i += 1
            continue
        # Pool i and i+1.
        w_sum = weights[i] + weights[i + 1]
        pooled = (ys[i] * weights[i] + ys[i + 1] * weights[i + 1]) / w_sum
        ys[i] = pooled
        weights[i] = w_sum
        del ys[i + 1]
        del weights[i + 1]
        del xs[i + 1]
        if i > 0 and ys[i - 1] > ys[i]:
            i -= 1
    return {"x": xs, "y": ys}


def _fit_sensitivities(
    records: list[dict],
    label_mistake_prob: dict[str, float],
) -> dict[str, float]:
    """Pick per-channel sensitivity by minimizing NLL of gold tau under that channel alone."""
    grid = [round(v, 2) for v in [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95]]
    best = {"stage1": 0.55, "stage2": 0.85, "specialist": 0.75}

    for channel in ("stage1", "stage2", "specialist"):
        best_nll = math.inf
        best_s = best[channel]
        for s in grid:
            nll = _channel_nll(records, channel, s, label_mistake_prob)
            if nll < best_nll:
                best_nll = nll
                best_s = s
        best[channel] = best_s

    return best


def _channel_nll(
    records: list[dict],
    channel: str,
    sensitivity: float,
    label_mistake_prob: dict[str, float],
) -> float:
    """Negative log-likelihood of gold tau under one channel.

    For each step we build the channel's likelihood vector at that sensitivity,
    then evaluate it at the gold tau index. Sums -log p across steps.
    """
    nll = 0.0
    n = 0
    for prediction in records:
        if not prediction.get("completed"):
            continue
        gold_tau = prediction.get("gold_first_mistake_index")
        steps = [s for s in (prediction.get("steps") or []) if not s.get("prism_meta")]
        num_steps = len(steps)
        if num_steps == 0:
            continue
        gold_idx = gold_tau if (gold_tau is not None and 0 <= gold_tau < num_steps) else num_steps
        example_steps_text = prediction.get("example_steps") or []

        for step in steps:
            step_index = step.get("step_index")
            if not isinstance(step_index, int) or step_index < 0 or step_index >= num_steps:
                continue
            lik = _channel_likelihood(
                channel=channel,
                step=step,
                step_index=step_index,
                num_steps=num_steps,
                sensitivity=sensitivity,
                example_step_text=(
                    example_steps_text[step_index]
                    if step_index < len(example_steps_text)
                    else ""
                ),
            )
            if lik is None:
                continue
            p = lik[gold_idx]
            nll -= math.log(max(p, 1e-12))
            n += 1
    return nll / max(n, 1)


def _channel_likelihood(
    *,
    channel: str,
    step: dict,
    step_index: int,
    num_steps: int,
    sensitivity: float,
    example_step_text: str,
) -> list[float] | None:
    if channel == "stage2":
        stage2 = step.get("stage2_parse") or {}
        labels = stage2.get("principle_labels") or {}
        principle_labels = [labels.get(k) for k in ("mathematical_concepts", "key_analyses", "calculations")]
        if not any(principle_labels):
            return None
        lik = make_stage2_likelihood(
            step_index=step_index,
            num_steps=num_steps,
            principle_labels=principle_labels,
            sensitivity=sensitivity,
        )
        return list(lik.values)

    if channel == "stage1":
        stage1 = step.get("stage1_parse") or {}
        sig = extract_stage1_inconsistency(
            stage1_calculations=stage1.get("calculations"),
            current_step=example_step_text,
        )
        if sig.inconsistency_strength == 0.0:
            return None
        lik = make_stage1_likelihood(
            step_index=step_index,
            num_steps=num_steps,
            inconsistency_strength=sig.inconsistency_strength,
            sensitivity=sensitivity,
        )
        return list(lik.values)

    if channel == "specialist":
        # Average over all specialist emissions at this step.
        hards = []
        valids = []
        for item in step.get("stage2_tool_trace") or []:
            if not isinstance(item, dict) or item.get("discarded"):
                continue
            name = item.get("tool_name")
            result = item.get("result")
            if not isinstance(name, str) or name not in SPECIALIST_TOOLS or not isinstance(result, dict):
                continue
            hard = 0.10
            if result.get("hard_contradiction"):
                hard = 0.92
            elif isinstance(result.get("contradictions"), list) and result["contradictions"]:
                hard = 0.55
            valid = 0.85 if result.get("valid_alternative") else 0.05
            hards.append(hard)
            valids.append(valid)
        if not hards:
            return None
        avg_hard = sum(hards) / len(hards)
        avg_valid = sum(valids) / len(valids)
        lik = make_specialist_likelihood(
            step_index=step_index,
            num_steps=num_steps,
            hard_conflict_strength=avg_hard,
            valid_alternative_strength=avg_valid,
            sensitivity=sensitivity,
        )
        return list(lik.values)

    return None


def calibrate(predictions_path: Path, output_path: Path) -> dict:
    records = list(_iter_records(predictions_path))
    if not records:
        raise RuntimeError(f"No predictions found in {predictions_path}")

    label_probs = _fit_label_mistake_prob(records)
    isotonic = _fit_specialist_isotonic(records)
    sensitivities = _fit_sensitivities(records, label_probs)

    payload = {
        "version": 1,
        "label_mistake_prob": label_probs,
        "specialist_isotonic": isotonic,
        "sensitivities": sensitivities,
        "calibration_size": sum(1 for r in records if r.get("completed")),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate PRISM likelihood parameters from a predictions JSONL with gold tau.",
    )
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/prism_likelihoods.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if not args.predictions_path.exists():
        print(f"predictions path does not exist: {args.predictions_path}", file=sys.stderr)
        return 1
    payload = calibrate(args.predictions_path, args.output_path)
    print(f"calibration_size: {payload['calibration_size']}")
    print(f"label_mistake_prob: {payload['label_mistake_prob']}")
    print(f"sensitivities: {payload['sensitivities']}")
    print(f"wrote {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
