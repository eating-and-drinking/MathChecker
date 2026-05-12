from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

SPECIALIST_TOOL_NAMES = (
    "alternative_route_verifier_tool",
    "equivalence_substitution_verifier_tool",
    "condition_obligation_verifier_tool",
)

ROUTER_LABELS = (
    "use_alternative_route_verifier_tool",
    "use_equivalence_substitution_verifier_tool",
    "use_condition_obligation_verifier_tool",
    "trigger_specialist_review",
)

_LABEL_TO_SPECIALIST = {
    "use_alternative_route_verifier_tool": "alternative_route_verifier_tool",
    "use_equivalence_substitution_verifier_tool": "equivalence_substitution_verifier_tool",
    "use_condition_obligation_verifier_tool": "condition_obligation_verifier_tool",
}


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


@dataclass(slots=True, frozen=True)
class RouterScoreResult:
    scores: dict[str, float]
    success: bool
    model_name: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SpecialistRouter(Protocol):
    def route(self, context: RouterContext) -> RouteDecision:
        ...


@dataclass(slots=True, frozen=True)
class LearnedRouterConfig:
    model_path: str
    confidence_threshold: float = 0.55
    max_length: int = 1024
    trust_remote_code: bool = True
    device: str | None = None
    per_label_thresholds: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LearnedSpecialistRouter:
    def __init__(self, config: LearnedRouterConfig) -> None:
        self.config = config
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device: str | None = None
        self._label_order: tuple[str, ...] = ROUTER_LABELS
        self._load_error: str | None = None
        self._max_length = config.max_length
        self._per_label_thresholds = {
            str(label): float(value)
            for label, value in (config.per_label_thresholds or {}).items()
            if isinstance(value, (int, float))
        }

    def score(self, context: RouterContext) -> RouterScoreResult:
        model, tokenizer, torch_module, device = self._load_backend()
        if model is None or tokenizer is None or torch_module is None or device is None:
            return RouterScoreResult(
                scores={},
                success=False,
                model_name=self.config.model_path,
                error=self._load_error or "learned router backend unavailable",
            )

        text = format_router_input_text(context)
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
        )
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }
        with torch_module.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits
            probabilities = torch_module.sigmoid(logits).detach().cpu().tolist()[0]
        scores = {
            label: float(probabilities[index])
            for index, label in enumerate(self._label_order)
            if index < len(probabilities)
        }
        return RouterScoreResult(
            scores=scores,
            success=True,
            model_name=self.config.model_path,
        )

    def route(self, context: RouterContext) -> RouteDecision:
        score_result = self.score(context)
        if not score_result.success:
            return RouteDecision(
                selected_specialists=(),
                trigger_specialist_review=False,
                confidence=None,
                source="learned_router_unavailable",
                model_name=score_result.model_name,
                success=False,
                error=score_result.error,
            )
        return build_route_decision_from_scores(
            score_result.scores,
            confidence_threshold=self.config.confidence_threshold,
            per_label_thresholds=self._per_label_thresholds,
            source="learned_router",
            model_name=score_result.model_name,
        )

    def _load_backend(self) -> tuple[Any, Any, Any, str | None]:
        if self._model is not None and self._tokenizer is not None and self._torch is not None and self._device is not None:
            return self._model, self._tokenizer, self._torch, self._device
        if self._load_error is not None:
            return None, None, None, None

        self._load_local_router_config()

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            self._load_error = (
                "Learned router requires optional training/runtime dependencies. "
                "Install them with `uv sync --group router`."
            )
            return None, None, None, None

        model_path = self.config.model_path
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=self.config.trust_remote_code,
            )
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"Failed to load learned router tokenizer: {exc}"
            return None, None, None, None
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = None
        path_obj = Path(model_path)
        if path_obj.exists() and (path_obj / "adapter_config.json").exists():
            try:
                from peft import AutoPeftModelForSequenceClassification
            except ImportError:
                self._load_error = (
                    "PEFT adapter checkpoint detected for learned router, but `peft` is not installed. "
                    "Install optional dependencies with `uv sync --group router`."
                )
                return None, None, None, None
            try:
                model = AutoPeftModelForSequenceClassification.from_pretrained(
                    model_path,
                    trust_remote_code=self.config.trust_remote_code,
                )
            except Exception as exc:  # noqa: BLE001
                self._load_error = f"Failed to load learned router PEFT model: {exc}"
                return None, None, None, None
        else:
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    trust_remote_code=self.config.trust_remote_code,
                )
            except Exception as exc:  # noqa: BLE001
                self._load_error = f"Failed to load learned router model: {exc}"
                return None, None, None, None

        id2label = getattr(model.config, "id2label", None)
        if isinstance(id2label, dict) and id2label:
            ordered: list[str] = []
            for index in range(len(id2label)):
                value = id2label.get(index, id2label.get(str(index)))
                if isinstance(value, str):
                    ordered.append(value)
            if ordered:
                self._label_order = tuple(ordered)

        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch
        self._device = device
        return model, tokenizer, torch, device

    def _load_local_router_config(self) -> None:
        path_obj = Path(self.config.model_path)
        config_path = path_obj / "pedcot_router_config.json"
        if not path_obj.exists() or not config_path.exists():
            return
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return

        label_order = raw.get("label_order")
        if isinstance(label_order, list) and label_order:
            ordered = [str(item) for item in label_order if isinstance(item, str)]
            if ordered:
                self._label_order = tuple(ordered)

        if not self._per_label_thresholds:
            per_label_thresholds = raw.get("per_label_thresholds")
            if isinstance(per_label_thresholds, dict):
                self._per_label_thresholds = {
                    str(label): float(value)
                    for label, value in per_label_thresholds.items()
                    if isinstance(value, (int, float))
                }
        max_length = raw.get("max_length")
        if isinstance(max_length, int) and max_length > 0:
            self._max_length = max_length


def format_router_input_text(context: RouterContext) -> str:
    previous_lines = "\n".join(
        f"(step {index + 1}) {step}"
        for index, step in enumerate(context.previous_steps)
    ).strip()
    if not previous_lines:
        previous_lines = "(none)"
    stage1_guidance_lines = [
        f"Stage-1 mathematical concepts: {context.stage1_mathematical_concepts or '(none)'}",
        f"Stage-1 key analyses: {context.stage1_key_analyses or '(none)'}",
        f"Stage-1 calculations: {context.stage1_calculations or '(none)'}",
    ]
    risk_flags = ", ".join(context.heuristic_risk_flags) if context.heuristic_risk_flags else "none"
    heuristic_specialists = ", ".join(context.heuristic_specialists) if context.heuristic_specialists else "none"
    return (
        f"Dataset: {context.dataset}\n"
        f"Question: {context.question}\n"
        f"Previous steps:\n{previous_lines}\n"
        f"Current step:\n{context.current_step}\n"
        f"{chr(10).join(stage1_guidance_lines)}\n"
        f"Heuristic step type: {context.heuristic_step_type}\n"
        f"Heuristic risk flags: {risk_flags}\n"
        f"Heuristic specialist route: {heuristic_specialists}\n"
    )


def build_router_training_labels(
    *,
    selected_specialists: list[str] | tuple[str, ...] | set[str],
    trigger_specialist_review: bool,
) -> dict[str, int]:
    specialist_set = {str(item) for item in selected_specialists}
    return {
        "use_alternative_route_verifier_tool": int("alternative_route_verifier_tool" in specialist_set),
        "use_equivalence_substitution_verifier_tool": int(
            "equivalence_substitution_verifier_tool" in specialist_set
        ),
        "use_condition_obligation_verifier_tool": int("condition_obligation_verifier_tool" in specialist_set),
        "trigger_specialist_review": int(trigger_specialist_review),
    }


def build_route_decision_from_scores(
    scores: dict[str, float],
    *,
    confidence_threshold: float,
    per_label_thresholds: dict[str, float] | None = None,
    source: str,
    model_name: str | None,
) -> RouteDecision:
    thresholds = per_label_thresholds or {}
    selected_specialists = tuple(
        specialist
        for label, specialist in _LABEL_TO_SPECIALIST.items()
        if scores.get(label, 0.0) >= float(thresholds.get(label, confidence_threshold))
    )
    review_threshold = float(thresholds.get("trigger_specialist_review", confidence_threshold))
    trigger_specialist_review = scores.get("trigger_specialist_review", 0.0) >= review_threshold
    confidence = compute_route_confidence(
        scores,
        selected_specialists,
        trigger_specialist_review,
        confidence_threshold=confidence_threshold,
        per_label_thresholds=per_label_thresholds,
    )
    return RouteDecision(
        selected_specialists=selected_specialists,
        trigger_specialist_review=trigger_specialist_review,
        confidence=confidence,
        source=source,
        model_name=model_name,
        candidate_scores=scores,
        success=True,
    )


def compute_route_confidence(
    scores: dict[str, float],
    selected_specialists: tuple[str, ...],
    trigger_specialist_review: bool,
    *,
    confidence_threshold: float = 0.55,
    per_label_thresholds: dict[str, float] | None = None,
) -> float | None:
    if not scores:
        return None
    thresholds = per_label_thresholds or {}
    selected_set = set(selected_specialists)
    decision_terms: list[float] = []
    for label in ROUTER_LABELS:
        score = scores.get(label)
        if score is None:
            continue
        threshold = float(thresholds.get(label, confidence_threshold))
        if label == "trigger_specialist_review":
            decision_terms.append(score if trigger_specialist_review else 1.0 - score)
            continue
        specialist = _LABEL_TO_SPECIALIST[label]
        margin = abs(score - threshold)
        decision_value = score if specialist in selected_set else 1.0 - score
        decision_terms.append((decision_value + margin) / 2.0)
    if not decision_terms:
        return None
    return sum(decision_terms) / len(decision_terms)
