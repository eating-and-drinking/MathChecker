from __future__ import annotations

import re

from ..core.constants import PRINCIPLE_LABELS
from ..core.models import Stage1Parse, Stage2Parse

_SECTION_REGEX = re.compile(r"(?m)^\s*([123])\.\s+")
_LABEL_REGEX = re.compile(
    "|".join(re.escape(label) for label in PRINCIPLE_LABELS),
    flags=re.IGNORECASE,
)


def _split_numbered_sections(text: str) -> list[str]:
    matches = list(_SECTION_REGEX.finditer(text))
    sections: list[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append(text[start:end].strip())
    return sections


def _split_by_headings(text: str, headings: list[str]) -> list[str]:
    spans: list[tuple[int, int]] = []
    for heading in headings:
        match = re.search(re.escape(heading), text, flags=re.IGNORECASE)
        if not match:
            return []
        spans.append(match.span())
    starts = [span[0] for span in spans]
    if starts != sorted(starts):
        return []
    sections: list[str] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        sections.append(text[start:end].strip())
    return sections


def _split_by_heading_patterns(text: str, heading_patterns: list[str]) -> list[str]:
    spans: list[tuple[int, int]] = []
    for pattern in heading_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            return []
        spans.append(match.span())
    starts = [span[0] for span in spans]
    if starts != sorted(starts):
        return []
    sections: list[str] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        sections.append(text[start:end].strip())
    return sections


def parse_stage1_response(text: str) -> Stage1Parse:
    sections = _split_by_heading_patterns(
        text,
        [
            r"Mathematical Concepts",
            r"Key Analyses",
            r"(Mathematical Expressions|Uncomputed Expressions)",
        ],
    )
    if len(sections) < 3:
        sections = _split_numbered_sections(text)
    if len(sections) < 3:
        sections = _split_by_headings(text, ["Mathematical Concepts", "Key Analyses", "Mathematical Expressions"])
    if len(sections) < 3:
        return Stage1Parse(success=False, error="Could not split stage-1 response into three sections.")
    return Stage1Parse(
        mathematical_concepts=sections[0],
        key_analyses=sections[1],
        calculations=sections[2],
        success=True,
    )


def _find_last_label(text: str) -> str | None:
    matches = list(_LABEL_REGEX.finditer(text))
    if not matches:
        return None
    return matches[-1].group(0).lower()


def parse_stage2_response(text: str) -> Stage2Parse:
    sections = _split_by_heading_patterns(
        text,
        [
            r"Mathematical Concepts",
            r"Key Analyses",
            r"(Mathematical Expressions|Calculation Results)",
        ],
    )
    if len(sections) < 3:
        sections = _split_numbered_sections(text)
    if len(sections) < 3:
        sections = _split_by_headings(text, ["Mathematical Concepts", "Key Analyses", "Mathematical Expressions"])
    if len(sections) < 3:
        return Stage2Parse(success=False, error="Could not split stage-2 response into three sections.")

    concept_label = _find_last_label(sections[0])
    analyses_label = _find_last_label(sections[1])
    calculations_label = _find_last_label(sections[2])
    if not all([concept_label, analyses_label, calculations_label]):
        return Stage2Parse(
            mathematical_concepts_label=concept_label,
            key_analyses_label=analyses_label,
            calculations_label=calculations_label,
            success=False,
            error="Missing one or more principle labels in stage-2 response.",
        )
    return Stage2Parse(
        mathematical_concepts_label=concept_label,
        key_analyses_label=analyses_label,
        calculations_label=calculations_label,
        success=True,
    )
