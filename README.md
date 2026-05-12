# MathChecker

MathChecker is a reproducible CLI and library for step-level error detection in mathematical reasoning traces.

## Project Layout

```text
src/mathchecker/
  core/         Domain constants and typed models
  data/         Dataset loading, JSONL helpers, and persistence stores
  evaluation/   Offline metrics
  llm/          Model client integrations
  pipeline/     Prompts, parsing, tools, predictor, and orchestration
  cli.py        Command-line entrypoint
  utils.py      Shared utility helpers
tests/          Smoke tests
```

## Quick Start

```bash
uv sync
uv run mathchecker --help
```

## Development

```bash
uv run pytest
```

## Notes

- Prompt and method changes are tracked in [IMPROVEMENT_NOTES.md](IMPROVEMENT_NOTES.md).
- Runtime prompt templates live in `src/mathchecker/pipeline/templates/`.
