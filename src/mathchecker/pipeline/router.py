"""Legacy specialist-router types.

Most of this module's original content -- LearnedSpecialistRouter, the
per-label confidence thresholds, the Qwen3-0.6B LoRA loader, the
imitation/weak/policy/expected-gain label strategies -- has been retired by
the PRISM refactor (see src/mathchecker/prism/).

What's left here:

  - RouterContext, RouteDecision : dataclasses still used by the legacy
    PedCoTPredictor's step-type heuristic routing path. Kept so
    `--pipeline legacy` continues to work for ablation studies.
  - SpecialistRouter Protocol    : structural type used by the legacy path.
  - SPECIALIST_TOOL_NAMES        : canonical tool name tuple, also re-exported
    via prism.eig.DEFAULT_SPECIALIST_CANDIDATES.

PRISM routes specialists via greedy submodular Expected Information Gain
(see prism/eig.py); the learned multi-label classifier is gone.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

SPECIALIST_TOOL_NAMES = (
    "alternative_route_verifier_tool",
    "equivalence_substitution_verifier_tool",
    "condition_obligation_verifier_tool",
)


@dataclass(slots=True, frozen=True)
class RouterContext:
    dataset: str
    question: str
    previous_steps: tuple[str, ...]
    current_step: str
    heuristic_step_type: str
    heuristic_risk_flags: tuple[str, ...]
    heuristic_specialists: tuple[str, ...]
    stage1_mathematical_concepts: str | None = None
    stage1_key_analyses: str | None = None
    stage1_calculations: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class RouteDecision:
    selected_specialists: tuple[str, ...]
    trigger_specialist_review: bool
    confidence: float | None
    source: str
    model_name: str | None = None
    candidate_scores: dict[str, float] = field(default_factory=dict)
    success: bool = True
    error: str | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SpecialistRouter(Protocol):
    def route(self, context: RouterContext) -> RouteDecision:
        ...


# Backwards-compatible stub classes. These are retained so the legacy
# predictor's optional `learned_router` constructor path doesn't break at
# import time. They are NOT functional; the PRISM CLI does not expose them.

@dataclass(slots=True, frozen=True)
class LearnedRouterConfig:
    """Retired. PRISM uses EIG regression instead. See prism/eig.py."""

    model_path: str
    confidence_threshold: float = 0.55
    max_length: int = 1024
    trust_remote_code: bool = True
    device: str | None = None
    per_label_thresholds: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LearnedSpecialistRouter:
    """Retired stub.

    Always returns a "router unavailable" decision; callers should fall back
    to the heuristic step-type route. Kept as an import target for the legacy
    PedCoTPredictor path so `--pipeline legacy` continues to construct
    cleanly.
    """

    def __init__(self, config: LearnedRouterConfig) -> None:
        self.config = config

    def score(self, context: RouterContext) -> "RouterScoreResult":  # noqa: F821
        return RouterScoreResult(
            scores={},
            success=False,
            model_name=self.config.model_path,
            error="learned router has been retired; PRISM uses EIG routing instead",
        )

    def route(self, context: RouterContext) -> RouteDecision:
        return RouteDecision(
            selected_specialists=(),
            trigger_specialist_review=False,
            confidence=None,
            source="learned_router_retired",
            model_name=self.config.model_path,
            success=False,
            error="learned router has been retired; PRISM uses EIG routing instead",
        )


@dataclass(slots=True, frozen=True)
class RouterScoreResult:
    scores: dict[str, float]
    success: bool
    model_name: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
