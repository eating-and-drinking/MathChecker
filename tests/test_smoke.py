from __future__ import annotations

import pytest

from mathchecker.cli import main
from mathchecker.pipeline.parsers import parse_stage2_response
from mathchecker.pipeline.prompts import PedCoTPromptBuilder


def test_cli_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0


def test_prompt_builder_loads_templates() -> None:
    builder = PedCoTPromptBuilder()

    prompt = builder.build_stage1_prompt("What is 2 + 2?", [])

    assert "Question: What is 2 + 2?" in prompt
    assert "1. Mathematical Concepts to Apply:" in prompt


def test_stage2_parser_smoke() -> None:
    response = """1. Mathematical Concepts to Apply:
Uses arithmetic.
Label: correct-and-aligned
2. Key Analyses for the Next Step:
Checks the next operation.
Label: reasonable-but-incomplete
3. Mathematical Expressions to Compute:
2 + 2 = 4
Label: correct-and-aligned
"""

    parsed = parse_stage2_response(response)

    assert parsed.success is True
    assert parsed.mathematical_concepts_label == "correct-and-aligned"
    assert parsed.key_analyses_label == "reasonable-but-incomplete"
    assert parsed.calculations_label == "correct-and-aligned"
