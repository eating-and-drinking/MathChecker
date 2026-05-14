#!/usr/bin/env python3
"""Train the PRISM Expected-Information-Gain regressor.

This script replaces the legacy `export_router_dataset.py`,
`train_learned_router.py`, and `evaluate_router.py` triplet. It implements a
single training target: regressing the closed-form EIG of each specialist
against the current posterior.

Pipeline
--------

1. Read offline trace predictions produced with `--pipeline prism` (or
   `--pipeline legacy` -- both work, because PRISM only needs the per-step
   stage2 + tool-trace evidence, not the legacy router output).
2. For each (trace, step_index), reconstruct the PRISM posterior using only
   the evidence visible up to and including that step.
3. For each specialist candidate c, compute EIG*(pi_t, c). This is the
   closed-form regression target -- no counterfactual estimation needed.
4. Save (features, EIG*) tuples as a JSONL training set.

Optional training of a small regressor (Qwen3-0.6B + LoRA) is left to the
caller; this script focuses on the EIG-label generation, which is the
non-trivial part.

Usage
-----

    python3 scripts/train_prism_router.py \
        --predictions-path artifacts/predictions/big-bench-mistake__model.jsonl \
        --export-path artifacts/prism_router/train.jsonl

"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

# Make the package importable when running as a standalone script.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mathchecker.prism.eig import (  # noqa: E402
    DEFAULT_SPECIALIST_CANDIDATES,
    SpecialistCandidate,
    expected_information_gain,
)
from mathchecker.prism.likelihoods import (  # noqa: E402
    make_specialist_likelihood,
    make_stage1_likelihood,
    make_stage2_likelihood,
    principle_labels_to_logits,
)
from mathchecker.prism.posterior import Posterior, length_prior  # noqa: E402
from mathchecker.prism.stage1_consistency import extract_stage1_inconsistency  # noqa: E402


def _iter_predictions(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def _evidence_from_step(
    step: dict,
    *,
    current_step_text: str = "",
) -> tuple[list[str | None], dict[str, tuple[float, float]], float]:
    """Return (principle_labels, specialist_emissions, stage1_inconsistency)."""
    stage2 = step.get("stage2_parse") or {}
    principle_labels_dict = stage2.get("principle_labels") or {}
    labels = [
        principle_labels_dict.get("mathematical_concepts"),
        principle_labels_dict.get("key_analyses"),
        principle_labels_dict.get("calculations"),
    ]
    emissions: dict[str, tuple[float, float]] = {}
    for item in step.get("stage2_tool_trace") or []:
        if not isinstance(item, dict) or item.get("discarded"):
            continue
        name = item.get("tool_name")
        result = item.get("result")
        if not isinstance(name, str) or not isinstance(result, dict):
            continue
        if name not in {c.name for c in DEFAULT_SPECIALIST_CANDIDATES}:
            continue
        hard = 0.10
        if result.get("hard_contradiction") or result.get("contradiction_level") == "hard_contradiction":
            hard = 0.92
        elif isinstance(result.get("contradictions"), list) and result["contradictions"]:
            hard = 0.55
        valid_alt = 0.05
        if result.get("valid_alternative") or result.get("valid_equivalent_transformation"):
            valid_alt = 0.85
        prev = emissions.get(name)
        if prev is None or hard > prev[0]:
            emissions[name] = (hard, valid_alt)

    # Stage1 channel (independent of stage2 / specialists).
    stage1_parse = step.get("stage1_parse") or {}
    stage1_calc = stage1_parse.get("calculations") if isinstance(stage1_parse, dict) else None
    stage1_signal = extract_stage1_inconsistency(
        stage1_calculations=stage1_calc,
        current_step=current_step_text,
    )
    return labels, emissions, float(stage1_signal.inconsistency_strength)


def _features_for_step(
    *,
    example_id: str,
    dataset: str,
    step_index: int,
    posterior: Posterior,
) -> dict:
    return {
        "example_id": example_id,
        "dataset": dataset,
        "step_index": step_index,
        "posterior_probs": list(posterior.probs),
        "posterior_max_mass": posterior.max_mass(),
        "posterior_entropy_nats": posterior.entropy(),
    }


def export_eig_targets(
    *,
    predictions_path: Path,
    export_path: Path,
    candidates: tuple[SpecialistCandidate, ...] = DEFAULT_SPECIALIST_CANDIDATES,
    p_no_error_prior: float = 0.4,
    stage2_sensitivity: float = 0.85,
) -> int:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with export_path.open("w", encoding="utf-8") as out:
        for prediction in _iter_predictions(predictions_path):
            if not prediction.get("completed"):
                continue
            steps = prediction.get("steps")
            if not isinstance(steps, list) or not steps:
                continue
            num_steps = sum(1 for step in steps if step.get("step_index", -1) >= 0)
            if num_steps == 0:
                continue
            posterior = Posterior(
                num_steps=num_steps,
                probs=length_prior(num_steps, p_no_error=p_no_error_prior),
            )
            for step in steps:
                if step.get("prism_meta"):
                    continue
                step_index = step.get("step_index")
                if not isinstance(step_index, int) or step_index < 0:
                    continue

                # --- replay evidence updates BEFORE the routing decision ---
                # The order matches the inference loop: stage1 -> stage2 -> route.
                current_step_text = (
                    prediction.get("example_steps", [None] * num_steps)[step_index]
                    if isinstance(prediction.get("example_steps"), list)
                    else ""
                )
                labels, emissions, stage1_q = _evidence_from_step(
                    step, current_step_text=current_step_text or ""
                )

                if stage1_q > 0.0:
                    stage1_lik = make_stage1_likelihood(
                        step_index=step_index,
                        num_steps=num_steps,
                        inconsistency_strength=stage1_q,
                    )
                    posterior.bayes_update(stage1_lik.values)

                stage2_lik = make_stage2_likelihood(
                    step_index=step_index,
                    num_steps=num_steps,
                    principle_labels=labels,
                    sensitivity=stage2_sensitivity,
                )
                posterior.bayes_update(stage2_lik.values)

                # --- compute EIG target for each candidate AT THIS POSTERIOR ---
                target = {}
                for cand in candidates:
                    target[cand.name] = expected_information_gain(
                        posterior=posterior,
                        step_index=step_index,
                        candidate=cand,
                    )

                features = _features_for_step(
                    example_id=prediction.get("example_id", ""),
                    dataset=prediction.get("dataset", ""),
                    step_index=step_index,
                    posterior=posterior,
                )
                features["principle_labels"] = labels
                features["stage1_inconsistency"] = stage1_q
                features["specialist_eig_target"] = target

                out.write(json.dumps(features, ensure_ascii=False) + "\n")
                count += 1

                # --- apply specialist updates so posterior reflects the
                #     observed-evidence state for the next step ---
                for tool_name, (hard, valid_alt) in emissions.items():
                    cand = next(
                        (c for c in candidates if c.name == tool_name),
                        None,
                    )
                    if cand is None:
                        continue
                    spec_lik = make_specialist_likelihood(
                        step_index=step_index,
                        num_steps=num_steps,
                        hard_conflict_strength=hard,
                        valid_alternative_strength=valid_alt,
                        sensitivity=cand.sensitivity,
                        source=cand.name,
                    )
                    posterior.bayes_update(spec_lik.values)

    return count


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate closed-form EIG regression targets for the PRISM router.",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        required=True,
        help="Path to a MathChecker predictions JSONL (e.g. artifacts/predictions/*.jsonl).",
    )
    parser.add_argument(
        "--export-path",
        type=Path,
        required=True,
        help="Where to write the EIG-target training set.",
    )
    parser.add_argument(
        "--p-no-error",
        type=float,
        default=0.4,
        help="Prior mass on tau = infty (no error).",
    )
    parser.add_argument(
        "--stage2-sensitivity",
        type=float,
        default=0.85,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if not args.predictions_path.exists():
        parser.error(f"predictions path does not exist: {args.predictions_path}")
    count = export_eig_targets(
        predictions_path=args.predictions_path,
        export_path=args.export_path,
        p_no_error_prior=args.p_no_error,
        stage2_sensitivity=args.stage2_sensitivity,
    )
    print(f"Wrote {count} EIG-target samples to {args.export_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
