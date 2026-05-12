from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TraceExample:
    example_id: str
    dataset: str
    question: str
    steps: list[str]
    gold_answer: str
    model_answer: str
    gold_first_mistake_index: int | None

    @property
    def gold_trace_label(self) -> int:
        return 1 if self.gold_first_mistake_index is None else 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Stage1Parse:
    mathematical_concepts: str | None = None
    key_analyses: str | None = None
    calculations: str | None = None
    success: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Stage2Parse:
    mathematical_concepts_label: str | None = None
    key_analyses_label: str | None = None
    calculations_label: str | None = None
    success: bool = False
    error: str | None = None

    @property
    def principle_labels(self) -> dict[str, str | None]:
        return {
            "mathematical_concepts": self.mathematical_concepts_label,
            "key_analyses": self.key_analyses_label,
            "calculations": self.calculations_label,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["principle_labels"] = self.principle_labels
        return payload


@dataclass(slots=True)
class StepPrediction:
    step_index: int
    pred_step_label: int | None
    stage1_prompt: str
    stage2_prompt: str | None
    stage1_raw_response: str | None
    stage2_raw_response: str | None
    stage1_parse: dict[str, Any]
    stage2_parse: dict[str, Any] | None
    stage2_step_type: str | None = None
    stage2_step_type_meta: dict[str, Any] = field(default_factory=dict)
    stage2_route_meta: dict[str, Any] = field(default_factory=dict)
    principle_labels: dict[str, str | None] = field(default_factory=dict)
    stage1_tool_trace: list[dict[str, Any]] = field(default_factory=list)
    stage1_tool_errors: list[str] = field(default_factory=list)
    stage2_tool_trace: list[dict[str, Any]] = field(default_factory=list)
    stage2_tool_errors: list[str] = field(default_factory=list)
    stage1_attempts: int = 0
    stage2_attempts: int = 0
    stage2_review_prompt: str | None = None
    stage2_review_raw_response: str | None = None
    stage2_review_parse: dict[str, Any] | None = None
    stage2_review_attempts: int = 0
    stage2_original_parse: dict[str, Any] | None = None
    stage2_review_applied: bool = False
    stage2_specialist_review_prompt: str | None = None
    stage2_specialist_review_raw_response: str | None = None
    stage2_specialist_review_parse: dict[str, Any] | None = None
    stage2_specialist_review_attempts: int = 0
    stage2_specialist_review_applied: bool = False
    parse_status: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TracePrediction:
    example_id: str
    dataset: str
    model: str
    pred_first_mistake_index: int | None
    pred_trace_label: int | None
    gold_first_mistake_index: int | None
    gold_trace_label: int
    steps: list[dict[str, Any]]
    completed: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StageCacheRecord:
    dataset: str
    example_id: str
    step_index: int
    stage: str
    model: str
    prompt_hash: str
    prompt_text: str
    response_text: str
    attempt: int
    meta: dict[str, Any]

    def cache_key(self) -> tuple[str, str, int, str, str, str]:
        return (
            self.dataset,
            self.example_id,
            self.step_index,
            self.stage,
            self.model,
            self.prompt_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
