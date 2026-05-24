# MathChecker

MathChecker is a reproducible CLI and library for step-level error detection in mathematical reasoning traces, built on the **PRISM** framework (Posterior-driven Routing with Information-theoretic Stopping for Mathematical reasoning verification).

## Project Layout

```text
src/mathchecker/
  core/         Domain constants and typed models
  data/         Dataset loading, JSONL helpers, and persistence stores
  evaluation/   Offline metrics
  llm/          Model client integrations
  pipeline/     Prompts, parsing, specialist tools, legacy predictor
  prism/        PRISM core: posterior, likelihoods, EIG, conformal, infer, predictor
  cli.py        Command-line entrypoint
  utils.py      Shared utility helpers
scripts/        Calibration + EIG-target training scripts
tests/          Unit + end-to-end tests
```

## Quick Start

```bash
uv sync
uv run mathchecker --help

# Run inference with PRISM (default pipeline)
uv run mathchecker run --dataset big-bench-mistake --model gpt-4o-mini --stage2-tools triad

# Calibrate likelihoods on hold-out predictions
python scripts/calibrate_likelihoods.py --predictions-path artifacts/predictions/*.jsonl

# Export EIG regression labels for router training
python scripts/train_prism_router.py --predictions-path <...> --export-path <...>
```

## Development

```bash
uv run pytest                # full suite (60+ tests)
```

## Documentation

- **Theoretical appendix**: [PRISM_THEORY.md](PRISM_THEORY.md) — rigorous proofs for the five core theorems (submodular routing, split-conformal coverage, universality, sample complexity, mis-specification robustness).
- **Pipeline overview**: [prism_pipeline.html](prism_pipeline.html) — visual training + inference walkthrough with a concrete example.
- Runtime prompt templates live in `src/mathchecker/pipeline/templates/`.
