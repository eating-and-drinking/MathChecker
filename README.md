# PedCoT

PedCoT is a reproducible CLI and library for step-level error detection in mathematical reasoning traces.

## Project Layout

```text
src/pedcot/
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
uv run pedcot --help
```

## Development

```bash
uv run pytest
```

## Notes

- Prompt and method changes are tracked in [IMPROVEMENT_NOTES.md](IMPROVEMENT_NOTES.md).
- Runtime prompt templates live in `src/pedcot/pipeline/templates/`.
