from __future__ import annotations

from typing import Any, Protocol

from ..core.constants import (
    DATASET_BIG_BENCH_MISTAKE,
    NEGATIVE_TRACE_LABEL,
    POSITIVE_TRACE_LABEL,
    STAGE_1,
    STAGE_2,
    STAGE_2_STEP_TYPE,
    STAGE_2_REVIEW,
    STAGE_2_SPECIALIST_REVIEW,
)
from .parsers import parse_stage1_response, parse_stage2_response
from .prompts import PedCoTPromptBuilder
from ..data.stores import StageCacheStore
from .router import (
    LearnedRouterConfig,
    LearnedSpecialistRouter,
    RouteDecision,
    RouterContext,
    SPECIALIST_TOOL_NAMES,
    SpecialistRouter,
)
from .tools import Stage1Tooling, build_stage1_tooling, build_stage2_tooling
from .step_classifier import (
    StepTypeClassification,
    apply_step_type_specialist_route,
    build_step_type_classification,
    classify_step_type,
    parse_step_type_classifier_response,
)
from ..core.models import Stage1Parse, Stage2Parse, StageCacheRecord, StepPrediction, TraceExample, TracePrediction
from ..utils import sha256_text

_NETWORK_ERROR_MARKERS = (
    "timeout",
    "timed out",
    "connection",
    "rate limit",
    "status code: 429",
    "status code: 500",
    "status code: 502",
    "status code: 503",
    "status code: 504",
    "service unavailable",
    "temporarily unavailable",
)

TOOL_CALL_POLICY_VERSION = "dedupe-probe-v2"


class TextGenerationClient(Protocol):
    def generate(
        self,
        *,
        model: str,
        prompt: str,
        metadata: dict | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_handlers: dict[str, Any] | None = None,
        required_tool_names: list[str] | None = None,
    ) -> tuple[str, dict]:
        ...


class PedCoTPredictor:
    def __init__(
        self,
        client: TextGenerationClient,
        cache_store: StageCacheStore,
        max_stage_attempts: int = 2,
        stage1_tools_mode: str = "none",
        stage2_tools_mode: str = "none",
        stage2_step_type_mode: str = "hybrid",
        stage2_router_mode: str = "step-type",
        stage2_router_model_path: str | None = None,
        stage2_router_confidence_threshold: float = 0.55,
        learned_router: SpecialistRouter | None = None,
    ) -> None:
        self.client = client
        self.cache_store = cache_store
        self.max_stage_attempts = max_stage_attempts
        self.prompt_builder = PedCoTPromptBuilder()
        self.stage1_tools_mode = stage1_tools_mode
        self.stage2_tools_mode = stage2_tools_mode
        self.stage2_step_type_mode = stage2_step_type_mode
        self.stage2_router_mode = stage2_router_mode
        self.stage2_router_model_path = stage2_router_model_path
        self.stage2_router_confidence_threshold = stage2_router_confidence_threshold
        self.learned_router = learned_router
        if self.learned_router is None and stage2_router_model_path:
            self.learned_router = LearnedSpecialistRouter(
                LearnedRouterConfig(
                    model_path=stage2_router_model_path,
                    confidence_threshold=stage2_router_confidence_threshold,
                )
            )
        self._stage1_tooling_cache: dict[str, Stage1Tooling] = {}
        self._stage2_tooling_cache: dict[str, Stage1Tooling] = {}

    def _stage1_tooling_for_dataset(self, dataset: str) -> Stage1Tooling:
        key = dataset.strip().lower()
        cached = self._stage1_tooling_cache.get(key)
        if cached is not None:
            return cached
        tooling = build_stage1_tooling(self.stage1_tools_mode, dataset=dataset)
        self._stage1_tooling_cache[key] = tooling
        return tooling

    def _stage2_tooling_for_dataset(self, dataset: str) -> Stage1Tooling:
        key = dataset.strip().lower()
        cached = self._stage2_tooling_cache.get(key)
        if cached is not None:
            return cached
        tooling = build_stage2_tooling(self.stage2_tools_mode, dataset=dataset)
        self._stage2_tooling_cache[key] = tooling
        return tooling

    def _cached_or_generate(
        self,
        *,
        example: TraceExample,
        step_index: int,
        stage: str,
        model: str,
        prompt_text: str,
        metadata: dict,
        tools: list[dict[str, Any]] | None = None,
        tool_handlers: dict[str, Any] | None = None,
        required_tool_names: set[str] | None = None,
        cache_context: str | None = None,
    ) -> tuple[str, int, dict, bool]:
        hash_material = prompt_text if cache_context is None else f"{prompt_text}\n\n[cache_context]{cache_context}"
        prompt_hash = sha256_text(hash_material)
        cache_key = (example.dataset, example.example_id, step_index, stage, model, prompt_hash)
        cached = self.cache_store.get(cache_key)
        next_attempt = 1
        if cached is not None:
            next_attempt = cached.attempt + 1
            if stage in {STAGE_1, STAGE_2} and required_tool_names:
                if self._required_tools_success(
                    response_meta=cached.meta,
                    required_tool_names=required_tool_names,
                ):
                    return cached.response_text, cached.attempt, cached.meta, True
            else:
                return cached.response_text, cached.attempt, cached.meta, True

        generate_kwargs: dict[str, Any] = {
            "model": model,
            "prompt": prompt_text,
            "metadata": metadata,
        }
        if tools:
            generate_kwargs["tools"] = tools
            generate_kwargs["tool_handlers"] = tool_handlers or {}
            generate_kwargs["required_tool_names"] = sorted(required_tool_names or set())
        response_text, response_meta = self.client.generate(**generate_kwargs)
        record = StageCacheRecord(
            dataset=example.dataset,
            example_id=example.example_id,
            step_index=step_index,
            stage=stage,
            model=model,
            prompt_hash=prompt_hash,
            prompt_text=prompt_text,
            response_text=response_text,
            attempt=next_attempt,
            meta=response_meta,
        )
        self.cache_store.append(record)
        return response_text, record.attempt, response_meta, False

    def _generate_retry(
        self,
        *,
        example: TraceExample,
        step_index: int,
        stage: str,
        model: str,
        prompt_text: str,
        prior_attempt: int,
        metadata: dict,
        tools: list[dict[str, Any]] | None = None,
        tool_handlers: dict[str, Any] | None = None,
        required_tool_names: set[str] | None = None,
        cache_context: str | None = None,
    ) -> tuple[str, int, dict]:
        generate_kwargs: dict[str, Any] = {
            "model": model,
            "prompt": prompt_text,
            "metadata": metadata,
        }
        if tools:
            generate_kwargs["tools"] = tools
            generate_kwargs["tool_handlers"] = tool_handlers or {}
            generate_kwargs["required_tool_names"] = sorted(required_tool_names or set())
        response_text, response_meta = self.client.generate(**generate_kwargs)
        hash_material = prompt_text if cache_context is None else f"{prompt_text}\n\n[cache_context]{cache_context}"
        prompt_hash = sha256_text(hash_material)
        record = StageCacheRecord(
            dataset=example.dataset,
            example_id=example.example_id,
            step_index=step_index,
            stage=stage,
            model=model,
            prompt_hash=prompt_hash,
            prompt_text=prompt_text,
            response_text=response_text,
            attempt=prior_attempt + 1,
            meta=response_meta,
        )
        self.cache_store.append(record)
        return response_text, record.attempt, response_meta

    def _stage1_tool_status(
        self,
        *,
        response_meta: dict[str, Any],
        required_tool_names: set[str],
    ) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
        if not required_tool_names:
            return [], [], {
                "stage1_tools_mode": self.stage1_tools_mode,
                "stage1_tool_required": False,
                "stage1_tool_success": True,
                "stage1_tool_required_names": [],
                "stage1_tool_called_names": [],
                "stage1_tool_call_count": 0,
                "stage1_tool_missing_names": [],
                "stage1_tool_error_count": 0,
            }

        trace_raw = response_meta.get("tool_trace", [])
        errors_raw = response_meta.get("tool_errors", [])
        trace = trace_raw if isinstance(trace_raw, list) else []
        errors = [str(item) for item in errors_raw] if isinstance(errors_raw, list) else []

        called_names: set[str] = set()
        python_statuses: list[str] = []
        logic_statuses: list[str] = []
        python_overlap_errors: list[str] = []
        python_tool_errors: list[str] = []
        stage1_math_tool_names = {
            "python_calc_tool",
            "prm_constraint_tool",
            "symbolic_relation_tool",
            "domain_guard_tool",
            "unit_ratio_tool",
            "gsm_expr_reference_tool",
        }
        strict_calc_tools = {"python_calc_tool", "prm_constraint_tool"}
        required_call_errors: list[str] = []
        effective_trace_count = 0
        for call in trace:
            if not isinstance(call, dict):
                continue
            if call.get("discarded"):
                continue
            effective_trace_count += 1
            name = call.get("tool_name")
            if isinstance(name, str) and name:
                called_names.add(name)
            result = call.get("result")
            if not isinstance(result, dict):
                continue
            status_value = result.get("status")
            verification_type_value = result.get("verification_type")
            status_text = status_value if isinstance(status_value, str) else ""
            verification_type = verification_type_value if isinstance(verification_type_value, str) else ""
            if isinstance(name, str) and name in required_tool_names and status_text in {"tool_error", "error"}:
                required_call_errors.append(f"{name} returned {status_text} status.")
            if name in stage1_math_tool_names:
                if status_text:
                    python_statuses.append(status_text)
                if name in strict_calc_tools and verification_type and verification_type not in {
                    "numeric",
                    "symbolic",
                    "mixed",
                    "numeric_symbolic",
                    "constraint_symbolic",
                }:
                    python_overlap_errors.append(
                        f"{name} returned unsupported verification_type={verification_type}."
                    )
                if name in strict_calc_tools and status_text == "tool_error":
                    python_tool_errors.append(f"{name} returned tool_error status.")
            if name == "logic_check_tool" and status_text:
                logic_statuses.append(status_text)

        required_names = set(required_tool_names)
        missing_names = sorted(required_names - called_names)
        all_required_called = len(missing_names) == 0
        total_errors = [*errors, *python_overlap_errors, *python_tool_errors, *required_call_errors]
        success = all_required_called and len(total_errors) == 0

        status = {
            "stage1_tools_mode": self.stage1_tools_mode,
            "stage1_tool_required": True,
            "stage1_tool_success": success,
            "stage1_tool_required_names": sorted(required_names),
            "stage1_tool_called_names": sorted(called_names),
            "stage1_tool_call_count": effective_trace_count,
            "stage1_tool_missing_names": missing_names,
            "stage1_tool_error_count": len(total_errors),
            "stage1_python_statuses": python_statuses,
            "stage1_logic_statuses": logic_statuses,
            "stage1_python_logic_overlap": len(python_overlap_errors) > 0,
            "stage1_python_logic_overlap_errors": python_overlap_errors,
            "stage1_python_tool_error_statuses": python_tool_errors,
            "stage1_required_tool_error_statuses": required_call_errors,
            "stage1_math_tool_names": sorted(stage1_math_tool_names),
        }
        return trace, total_errors, status

    @staticmethod
    def _required_tools_success(
        *,
        response_meta: dict[str, Any],
        required_tool_names: set[str],
    ) -> bool:
        trace_raw = response_meta.get("tool_trace", [])
        errors_raw = response_meta.get("tool_errors", [])
        trace = trace_raw if isinstance(trace_raw, list) else []
        errors = errors_raw if isinstance(errors_raw, list) else []
        called_names = {
            str(item.get("tool_name"))
            for item in trace
            if isinstance(item, dict) and not item.get("discarded") and isinstance(item.get("tool_name"), str)
        }
        if not required_tool_names.issubset(called_names) or len(errors) != 0:
            return False

        for item in trace:
            if not isinstance(item, dict):
                continue
            if item.get("discarded"):
                continue
            tool_name = item.get("tool_name")
            if tool_name not in required_tool_names:
                continue
            result = item.get("result")
            if not isinstance(result, dict):
                continue
            status = result.get("status")
            if status in {"tool_error", "error"}:
                return False
        return True

    @staticmethod
    def _is_network_like_error(message: str) -> bool:
        lowered = message.lower()
        return any(marker in lowered for marker in _NETWORK_ERROR_MARKERS)

    def _should_degrade_tool_failure(
        self,
        *,
        stage: str,
        tool_errors: list[str],
        parse_error: str | None,
    ) -> tuple[bool, str]:
        # Degrade only on non-network failures. Transient network failures should keep retry semantics.
        for error in tool_errors:
            if self._is_network_like_error(error):
                return False, "transient_network"
        if parse_error and self._is_network_like_error(parse_error):
            return False, "transient_network"
        return True, f"{stage}_capability_limit"

    @staticmethod
    def _extract_failed_tools(
        *,
        required_tool_names: set[str],
        tool_trace: list[dict[str, Any]],
        tool_errors: list[str],
    ) -> list[str]:
        failed: set[str] = set()
        called: set[str] = set()
        for item in tool_trace:
            if not isinstance(item, dict):
                continue
            if item.get("discarded"):
                continue
            tool_name_raw = item.get("tool_name")
            if not isinstance(tool_name_raw, str) or not tool_name_raw:
                continue
            called.add(tool_name_raw)
            result = item.get("result")
            if not isinstance(result, dict):
                continue
            status = result.get("status")
            if status in {"tool_error", "error"}:
                failed.add(tool_name_raw)
        for name in required_tool_names:
            if name not in called:
                failed.add(name)
        for error in tool_errors:
            lowered = error.lower()
            for name in required_tool_names:
                if name.lower() in lowered:
                    failed.add(name)
        if not failed and required_tool_names:
            return sorted(required_tool_names)
        return sorted(failed)

    def _run_stage1_no_tool_fallback(
        self,
        *,
        example: TraceExample,
        step_index: int,
        model: str,
        prompt_text: str,
        degrade_reason: str,
        failed_tools: list[str],
    ) -> tuple[str, Stage1Parse, int]:
        response_text, attempt, _, from_cache = self._cached_or_generate(
            example=example,
            step_index=step_index,
            stage=STAGE_1,
            model=model,
            prompt_text=prompt_text,
            metadata={
                "dataset": example.dataset,
                "example_id": example.example_id,
                "step_index": step_index,
                "stage": STAGE_1,
                "stage1_tools_mode": self.stage1_tools_mode,
                "tool_call_policy": TOOL_CALL_POLICY_VERSION,
                "stage1_degraded_to_no_tool": True,
                "stage1_degrade_reason": degrade_reason,
                "stage1_failed_tools": failed_tools,
            },
            tools=None,
            tool_handlers=None,
            required_tool_names=None,
            cache_context=(
                f"stage1_tools_mode={self.stage1_tools_mode};"
                f"tool_call_policy={TOOL_CALL_POLICY_VERSION};"
                "stage1_required_tools=none;"
                f"stage1_degraded_to_no_tool=true;"
                f"stage1_degrade_reason={degrade_reason};"
                f"stage1_failed_tools={','.join(failed_tools)}"
            ),
        )
        parsed = parse_stage1_response(response_text)
        fresh_attempts = 0 if from_cache else 1
        while (not parsed.success) and fresh_attempts < self.max_stage_attempts:
            response_text, attempt, _ = self._generate_retry(
                example=example,
                step_index=step_index,
                stage=STAGE_1,
                model=model,
                prompt_text=prompt_text,
                prior_attempt=attempt,
                metadata={
                    "dataset": example.dataset,
                    "example_id": example.example_id,
                    "step_index": step_index,
                    "stage": STAGE_1,
                    "stage1_tools_mode": self.stage1_tools_mode,
                    "tool_call_policy": TOOL_CALL_POLICY_VERSION,
                    "stage1_degraded_to_no_tool": True,
                    "stage1_degrade_reason": degrade_reason,
                    "stage1_failed_tools": failed_tools,
                },
                tools=None,
                tool_handlers=None,
                required_tool_names=None,
                cache_context=(
                    f"stage1_tools_mode={self.stage1_tools_mode};"
                    f"tool_call_policy={TOOL_CALL_POLICY_VERSION};"
                    "stage1_required_tools=none;"
                    f"stage1_degraded_to_no_tool=true;"
                    f"stage1_degrade_reason={degrade_reason};"
                    f"stage1_failed_tools={','.join(failed_tools)}"
                ),
            )
            fresh_attempts += 1
            parsed = parse_stage1_response(response_text)
        return response_text, parsed, attempt

    def _run_stage2_no_tool_fallback(
        self,
        *,
        example: TraceExample,
        step_index: int,
        model: str,
        prompt_text: str,
        degrade_reason: str,
        failed_tools: list[str],
    ) -> tuple[str, Stage2Parse, int]:
        response_text, attempt, _, from_cache = self._cached_or_generate(
            example=example,
            step_index=step_index,
            stage=STAGE_2,
            model=model,
            prompt_text=prompt_text,
            metadata={
                "dataset": example.dataset,
                "example_id": example.example_id,
                "step_index": step_index,
                "stage": STAGE_2,
                "stage2_tools_mode": self.stage2_tools_mode,
                "tool_call_policy": TOOL_CALL_POLICY_VERSION,
                "stage2_degraded_to_no_tool": True,
                "stage2_degrade_reason": degrade_reason,
                "stage2_failed_tools": failed_tools,
            },
            tools=None,
            tool_handlers=None,
            required_tool_names=None,
            cache_context=(
                f"stage2_tools_mode={self.stage2_tools_mode};"
                f"tool_call_policy={TOOL_CALL_POLICY_VERSION};"
                "stage2_required_tools=none;"
                "stage2_degraded_to_no_tool=true;"
                f"stage2_degrade_reason={degrade_reason};"
                f"stage2_failed_tools={','.join(failed_tools)}"
            ),
        )
        parsed = parse_stage2_response(response_text)
        fresh_attempts = 0 if from_cache else 1
        while (not parsed.success) and fresh_attempts < self.max_stage_attempts:
            response_text, attempt, _ = self._generate_retry(
                example=example,
                step_index=step_index,
                stage=STAGE_2,
                model=model,
                prompt_text=prompt_text,
                prior_attempt=attempt,
                metadata={
                    "dataset": example.dataset,
                    "example_id": example.example_id,
                    "step_index": step_index,
                    "stage": STAGE_2,
                    "stage2_tools_mode": self.stage2_tools_mode,
                    "tool_call_policy": TOOL_CALL_POLICY_VERSION,
                    "stage2_degraded_to_no_tool": True,
                    "stage2_degrade_reason": degrade_reason,
                    "stage2_failed_tools": failed_tools,
                },
                tools=None,
                tool_handlers=None,
                required_tool_names=None,
                cache_context=(
                    f"stage2_tools_mode={self.stage2_tools_mode};"
                    f"tool_call_policy={TOOL_CALL_POLICY_VERSION};"
                    "stage2_required_tools=none;"
                    "stage2_degraded_to_no_tool=true;"
                    f"stage2_degrade_reason={degrade_reason};"
                    f"stage2_failed_tools={','.join(failed_tools)}"
                ),
            )
            fresh_attempts += 1
            parsed = parse_stage2_response(response_text)
        return response_text, parsed, attempt

    def _run_stage1(
        self,
        example: TraceExample,
        step_index: int,
        model: str,
    ) -> tuple[str, Stage1Parse, int, str, list[dict[str, Any]], list[str], dict[str, Any]]:
        stage1_tooling = self._stage1_tooling_for_dataset(example.dataset)
        required_stage1_names = set(stage1_tooling.required_tool_names)
        if stage1_tooling.route_required_tool_names is not None:
            routed_names = stage1_tooling.route_required_tool_names(
                example.question,
                example.steps[:step_index],
            )
            if routed_names:
                required_stage1_names = {
                    name for name in routed_names if name in stage1_tooling.handlers
                }
        prompt_text = self.prompt_builder.build_stage1_prompt(
            question=example.question,
            previous_steps=example.steps[:step_index],
        )
        response_text, attempt, response_meta, from_cache = self._cached_or_generate(
            example=example,
            step_index=step_index,
            stage=STAGE_1,
            model=model,
            prompt_text=prompt_text,
            metadata={
                "dataset": example.dataset,
                "example_id": example.example_id,
                "step_index": step_index,
                "stage": STAGE_1,
                "stage1_tools_mode": self.stage1_tools_mode,
                "tool_call_policy": TOOL_CALL_POLICY_VERSION,
                "stage1_required_tool_names": sorted(required_stage1_names),
            },
            tools=stage1_tooling.schemas or None,
            tool_handlers=stage1_tooling.handlers or None,
            required_tool_names=required_stage1_names,
            cache_context=(
                f"stage1_tools_mode={self.stage1_tools_mode};"
                f"tool_call_policy={TOOL_CALL_POLICY_VERSION};"
                f"stage1_required_tools={','.join(sorted(required_stage1_names))}"
            ),
        )
        parsed = parse_stage1_response(response_text)
        fresh_attempts = 0 if from_cache else 1
        trace: list[dict[str, Any]]
        errors: list[str]
        tool_status: dict[str, Any]
        trace, errors, tool_status = self._stage1_tool_status(
            response_meta=response_meta,
            required_tool_names=required_stage1_names,
        )
        should_retry = (not parsed.success) or (not tool_status.get("stage1_tool_success", True))
        while should_retry and fresh_attempts < self.max_stage_attempts:
            response_text, attempt, response_meta = self._generate_retry(
                example=example,
                step_index=step_index,
                stage=STAGE_1,
                model=model,
                prompt_text=prompt_text,
                prior_attempt=attempt,
                metadata={
                    "dataset": example.dataset,
                    "example_id": example.example_id,
                    "step_index": step_index,
                    "stage": STAGE_1,
                    "stage1_tools_mode": self.stage1_tools_mode,
                    "tool_call_policy": TOOL_CALL_POLICY_VERSION,
                    "stage1_required_tool_names": sorted(required_stage1_names),
                },
                tools=stage1_tooling.schemas or None,
                tool_handlers=stage1_tooling.handlers or None,
                required_tool_names=required_stage1_names,
                cache_context=(
                    f"stage1_tools_mode={self.stage1_tools_mode};"
                    f"tool_call_policy={TOOL_CALL_POLICY_VERSION};"
                    f"stage1_required_tools={','.join(sorted(required_stage1_names))}"
                ),
            )
            fresh_attempts += 1
            parsed = parse_stage1_response(response_text)
            trace, errors, tool_status = self._stage1_tool_status(
                response_meta=response_meta,
                required_tool_names=required_stage1_names,
            )
            should_retry = (not parsed.success) or (not tool_status.get("stage1_tool_success", True))
        tool_success = bool(tool_status.get("stage1_tool_success", True))
        if not tool_success:
            degrade_allowed, degrade_reason = self._should_degrade_tool_failure(
                stage="stage1",
                tool_errors=errors,
                parse_error=parsed.error,
            )
            if degrade_allowed:
                failed_tools = self._extract_failed_tools(
                    required_tool_names=required_stage1_names,
                    tool_trace=trace,
                    tool_errors=errors,
                )
                fallback_response, fallback_parsed, fallback_attempt = self._run_stage1_no_tool_fallback(
                    example=example,
                    step_index=step_index,
                    model=model,
                    prompt_text=prompt_text,
                    degrade_reason=degrade_reason,
                    failed_tools=failed_tools,
                )
                if fallback_parsed.success:
                    response_text = fallback_response
                    parsed = fallback_parsed
                    attempt = fallback_attempt
                    tool_status = {
                        **tool_status,
                        "stage1_tool_degraded": True,
                        "stage1_tool_degrade_reason": degrade_reason,
                        "stage1_tool_degrade_failed_tools": failed_tools,
                        "stage1_tool_degrade_fallback_success": True,
                    }
                else:
                    tool_status = {
                        **tool_status,
                        "stage1_tool_degraded": True,
                        "stage1_tool_degrade_reason": degrade_reason,
                        "stage1_tool_degrade_failed_tools": failed_tools,
                        "stage1_tool_degrade_fallback_success": False,
                        "stage1_tool_degrade_fallback_parse_error": fallback_parsed.error,
                    }
            else:
                tool_status = {
                    **tool_status,
                    "stage1_tool_degraded": False,
                    "stage1_tool_degrade_reason": degrade_reason,
                }
        else:
            tool_status = {
                **tool_status,
                "stage1_tool_degraded": False,
                "stage1_tool_degrade_reason": "none",
            }
        return response_text, parsed, attempt, prompt_text, trace, errors, tool_status

    def _stage2_step_type_llm_enabled(self) -> bool:
        return self.stage2_tools_mode == "triad" and self.stage2_step_type_mode in {"llm", "hybrid"}

    def _should_run_stage2_step_type_llm(
        self,
        heuristic: StepTypeClassification,
    ) -> tuple[bool, str]:
        if not self._stage2_step_type_llm_enabled():
            return False, "llm_disabled"
        if self.stage2_step_type_mode == "llm":
            return True, "llm_mode_forced"
        if heuristic.step_type == "arithmetic":
            return False, "hybrid_uses_heuristic_for_arithmetic"
        return True, "hybrid_uses_llm_for_structural_or_ambiguous_steps"

    @staticmethod
    def _merge_step_type_classifications(
        heuristic: StepTypeClassification,
        llm: StepTypeClassification,
    ) -> StepTypeClassification:
        merged_flags = sorted({*heuristic.risk_flags, *llm.risk_flags})
        reasoning = llm.reasoning
        if heuristic.step_type != llm.step_type:
            reasoning = (
                f"{llm.reasoning} "
                f"Heuristic fallback hint: {heuristic.step_type}."
            ).strip()
        return build_step_type_classification(
            step_type=llm.step_type,
            reasoning=reasoning,
            risk_flags=merged_flags,
            source="hybrid",
            confidence=llm.confidence,
        )

    def _classify_stage2_step_type(
        self,
        *,
        example: TraceExample,
        step_index: int,
        model: str,
    ) -> tuple[StepTypeClassification, dict[str, Any]]:
        heuristic = classify_step_type(
            dataset=example.dataset,
            question=example.question,
            previous_steps=example.steps[:step_index],
            current_step=example.steps[step_index],
        )
        should_run_llm, llm_reason = self._should_run_stage2_step_type_llm(heuristic)
        status: dict[str, Any] = {
            "stage2_step_type_mode": self.stage2_step_type_mode,
            "stage2_step_type": heuristic.step_type,
            "stage2_step_type_reasoning": heuristic.reasoning,
            "stage2_step_type_risk_flags": list(heuristic.risk_flags),
            "stage2_step_type_specialists": list(heuristic.specialist_tool_names),
            "stage2_step_type_source": heuristic.source,
            "stage2_step_type_confidence": heuristic.confidence,
            "stage2_step_type_heuristic_step_type": heuristic.step_type,
            "stage2_step_type_heuristic_reasoning": heuristic.reasoning,
            "stage2_step_type_heuristic_risk_flags": list(heuristic.risk_flags),
            "stage2_step_type_llm_enabled": self._stage2_step_type_llm_enabled(),
            "stage2_step_type_llm_used": False,
            "stage2_step_type_llm_attempts": 0,
            "stage2_step_type_llm_success": False,
            "stage2_step_type_llm_reason": llm_reason,
            "stage2_step_type_fallback_to_heuristic": not should_run_llm,
            "stage2_step_type_fallback_reason": llm_reason if not should_run_llm else "",
        }
        if not should_run_llm:
            return heuristic, status

        prompt_text = self.prompt_builder.build_stage2_step_type_prompt(
            question=example.question,
            previous_steps=example.steps[:step_index],
            step_index=step_index,
            current_step=example.steps[step_index],
        )
        response_text, attempt, _, from_cache = self._cached_or_generate(
            example=example,
            step_index=step_index,
            stage=STAGE_2_STEP_TYPE,
            model=model,
            prompt_text=prompt_text,
            metadata={
                "dataset": example.dataset,
                "example_id": example.example_id,
                "step_index": step_index,
                "stage": STAGE_2_STEP_TYPE,
                "stage2_step_type_mode": self.stage2_step_type_mode,
            },
            tools=None,
            tool_handlers=None,
            required_tool_names=None,
            cache_context=f"stage2_step_type_mode={self.stage2_step_type_mode}",
        )
        parsed = parse_step_type_classifier_response(response_text)
        fresh_attempts = 0 if from_cache else 1
        while not parsed.success and fresh_attempts < self.max_stage_attempts:
            response_text, attempt, _ = self._generate_retry(
                example=example,
                step_index=step_index,
                stage=STAGE_2_STEP_TYPE,
                model=model,
                prompt_text=prompt_text,
                prior_attempt=attempt,
                metadata={
                    "dataset": example.dataset,
                    "example_id": example.example_id,
                    "step_index": step_index,
                    "stage": STAGE_2_STEP_TYPE,
                    "stage2_step_type_mode": self.stage2_step_type_mode,
                },
                tools=None,
                tool_handlers=None,
                required_tool_names=None,
                cache_context=f"stage2_step_type_mode={self.stage2_step_type_mode}",
            )
            fresh_attempts += 1
            parsed = parse_step_type_classifier_response(response_text)

        status["stage2_step_type_llm_used"] = True
        status["stage2_step_type_llm_attempts"] = attempt
        if not parsed.success or parsed.classification is None:
            status["stage2_step_type_llm_success"] = False
            status["stage2_step_type_llm_error"] = parsed.error
            status["stage2_step_type_fallback_to_heuristic"] = True
            status["stage2_step_type_fallback_reason"] = "llm_parse_failure"
            status["stage2_step_type_source"] = "heuristic_fallback"
            return heuristic, status

        llm_classification = parsed.classification
        final_classification = (
            llm_classification
            if self.stage2_step_type_mode == "llm"
            else self._merge_step_type_classifications(heuristic, llm_classification)
        )
        status.update(
            {
                "stage2_step_type": final_classification.step_type,
                "stage2_step_type_reasoning": final_classification.reasoning,
                "stage2_step_type_risk_flags": list(final_classification.risk_flags),
                "stage2_step_type_specialists": list(final_classification.specialist_tool_names),
                "stage2_step_type_source": final_classification.source,
                "stage2_step_type_confidence": final_classification.confidence,
                "stage2_step_type_llm_success": True,
                "stage2_step_type_llm_step_type": llm_classification.step_type,
                "stage2_step_type_llm_reasoning": llm_classification.reasoning,
                "stage2_step_type_llm_risk_flags": list(llm_classification.risk_flags),
                "stage2_step_type_fallback_to_heuristic": False,
                "stage2_step_type_fallback_reason": "",
            }
        )
        return final_classification, status

    def _stage2_learned_router_enabled(self) -> bool:
        return self.stage2_tools_mode == "triad" and self.stage2_router_mode in {"learned", "learned-hybrid"}

    @staticmethod
    def _heuristic_route_decision(
        *,
        classification: StepTypeClassification,
        routed_names: set[str],
    ) -> RouteDecision:
        selected_specialists = tuple(
            name for name in classification.specialist_tool_names if name in routed_names
        )
        return RouteDecision(
            selected_specialists=selected_specialists,
            trigger_specialist_review=bool(selected_specialists),
            confidence=classification.confidence,
            source="step_type_classifier",
            model_name=None,
            candidate_scores={},
            success=True,
        )

    def _route_stage2_specialists(
        self,
        *,
        example: TraceExample,
        step_index: int,
        stage1_parse: Stage1Parse,
        classification: StepTypeClassification,
        required_tool_names: set[str],
        available_tool_names: set[str],
    ) -> tuple[set[str], dict[str, Any]]:
        heuristic_required = apply_step_type_specialist_route(
            required_tool_names,
            classification=classification,
            available_tool_names=available_tool_names,
        )
        heuristic_decision = self._heuristic_route_decision(
            classification=classification,
            routed_names=heuristic_required,
        )
        status: dict[str, Any] = {
            "stage2_route_mode": self.stage2_router_mode,
            "stage2_route_source": heuristic_decision.source,
            "stage2_route_confidence": heuristic_decision.confidence,
            "stage2_route_selected_specialists": list(heuristic_decision.selected_specialists),
            "stage2_route_trigger_specialist_review": heuristic_decision.trigger_specialist_review,
            "stage2_route_model_name": heuristic_decision.model_name,
            "stage2_route_candidate_scores": heuristic_decision.candidate_scores,
            "stage2_route_success": True,
            "stage2_route_error": None,
            "stage2_route_fallback_used": False,
            "stage2_route_fallback_reason": "",
        }
        if not self._stage2_learned_router_enabled():
            return heuristic_required, status

        if self.learned_router is None:
            status.update(
                {
                    "stage2_route_source": "step_type_classifier_fallback",
                    "stage2_route_fallback_used": True,
                    "stage2_route_fallback_reason": "learned_router_not_configured",
                }
            )
            return heuristic_required, status

        context = RouterContext(
            dataset=example.dataset,
            question=example.question,
            previous_steps=tuple(example.steps[:step_index]),
            current_step=example.steps[step_index],
            heuristic_step_type=classification.step_type,
            heuristic_risk_flags=classification.risk_flags,
            heuristic_specialists=heuristic_decision.selected_specialists,
            stage1_mathematical_concepts=stage1_parse.mathematical_concepts,
            stage1_key_analyses=stage1_parse.key_analyses,
            stage1_calculations=stage1_parse.calculations,
        )
        learned_decision = self.learned_router.route(context)
        if not learned_decision.success:
            status.update(
                {
                    "stage2_route_source": "step_type_classifier_fallback",
                    "stage2_route_success": False,
                    "stage2_route_error": learned_decision.error,
                    "stage2_route_model_name": learned_decision.model_name,
                    "stage2_route_fallback_used": True,
                    "stage2_route_fallback_reason": "learned_router_error",
                }
            )
            return heuristic_required, status

        non_specialists = {
            name for name in required_tool_names if name not in SPECIALIST_TOOL_NAMES
        }
        learned_specialists = {
            name for name in learned_decision.selected_specialists if name in available_tool_names
        }
        learned_required = non_specialists | learned_specialists
        if self.stage2_router_mode == "learned-hybrid":
            confidence = learned_decision.confidence
            if confidence is None or confidence < self.stage2_router_confidence_threshold:
                status.update(
                    {
                        "stage2_route_source": "step_type_classifier_fallback",
                        "stage2_route_confidence": confidence,
                        "stage2_route_model_name": learned_decision.model_name,
                        "stage2_route_candidate_scores": learned_decision.candidate_scores,
                        "stage2_route_success": True,
                        "stage2_route_fallback_used": True,
                        "stage2_route_fallback_reason": "low_confidence",
                    }
                )
                return heuristic_required, status

        status.update(
            {
                "stage2_route_source": learned_decision.source,
                "stage2_route_confidence": learned_decision.confidence,
                "stage2_route_selected_specialists": list(learned_decision.selected_specialists),
                "stage2_route_trigger_specialist_review": learned_decision.trigger_specialist_review,
                "stage2_route_model_name": learned_decision.model_name,
                "stage2_route_candidate_scores": learned_decision.candidate_scores,
                "stage2_route_success": True,
                "stage2_route_error": None,
                "stage2_route_fallback_used": False,
                "stage2_route_fallback_reason": "",
            }
        )
        return learned_required, status

    def _run_stage2(
        self,
        example: TraceExample,
        step_index: int,
        model: str,
        stage1_parse: Stage1Parse,
    ) -> tuple[str, Stage2Parse, int, str, list[dict[str, Any]], list[str], dict[str, Any]]:
        assert stage1_parse.mathematical_concepts is not None
        assert stage1_parse.key_analyses is not None
        assert stage1_parse.calculations is not None
        stage2_tooling = self._stage2_tooling_for_dataset(example.dataset)
        step_type, step_type_status = self._classify_stage2_step_type(
            example=example,
            step_index=step_index,
            model=model,
        )
        required_stage2_names = set(stage2_tooling.required_tool_names)
        if stage2_tooling.route_required_tool_names is not None:
            routed_names = stage2_tooling.route_required_tool_names(
                example.question,
                [*example.steps[:step_index], example.steps[step_index]],
            )
            if routed_names:
                required_stage2_names = {
                    name for name in routed_names if name in stage2_tooling.handlers
                }
        required_stage2_names, route_status = self._route_stage2_specialists(
            example=example,
            step_index=step_index,
            stage1_parse=stage1_parse,
            classification=step_type,
            required_tool_names=required_stage2_names,
            available_tool_names=set(stage2_tooling.handlers),
        )

        prompt_text = self.prompt_builder.build_stage2_prompt(
            question=example.question,
            previous_steps=example.steps[:step_index],
            step_index=step_index,
            current_step=example.steps[step_index],
            stage1_concepts=stage1_parse.mathematical_concepts,
            stage1_analyses=stage1_parse.key_analyses,
            stage1_calculations=stage1_parse.calculations,
        )
        response_text, attempt, response_meta, from_cache = self._cached_or_generate(
            example=example,
            step_index=step_index,
            stage=STAGE_2,
            model=model,
            prompt_text=prompt_text,
            metadata={
                "dataset": example.dataset,
                "example_id": example.example_id,
                "step_index": step_index,
                "stage": STAGE_2,
                "stage2_tools_mode": self.stage2_tools_mode,
                "tool_call_policy": TOOL_CALL_POLICY_VERSION,
                "stage2_step_type_mode": self.stage2_step_type_mode,
                "stage2_router_mode": self.stage2_router_mode,
                "stage2_step_type": step_type.step_type,
                "stage2_step_type_risk_flags": list(step_type.risk_flags),
                "stage2_route_selected_specialists": route_status.get("stage2_route_selected_specialists", []),
                "stage2_required_tool_names": sorted(required_stage2_names),
            },
            tools=stage2_tooling.schemas or None,
            tool_handlers=stage2_tooling.handlers or None,
            required_tool_names=required_stage2_names,
            cache_context=(
                f"stage2_tools_mode={self.stage2_tools_mode};"
                f"tool_call_policy={TOOL_CALL_POLICY_VERSION};"
                f"stage2_step_type_mode={self.stage2_step_type_mode};"
                f"stage2_router_mode={self.stage2_router_mode};"
                f"stage2_step_type={step_type.step_type};"
                f"stage2_required_tools={','.join(sorted(required_stage2_names))}"
            ),
        )
        parsed = parse_stage2_response(response_text)
        stage2_trace_raw = response_meta.get("tool_trace", [])
        stage2_errors_raw = response_meta.get("tool_errors", [])
        stage2_trace = stage2_trace_raw if isinstance(stage2_trace_raw, list) else []
        stage2_errors = [str(item) for item in stage2_errors_raw] if isinstance(stage2_errors_raw, list) else []
        stage2_tool_success = self._required_tools_success(
            response_meta=response_meta,
            required_tool_names=required_stage2_names,
        )
        stage2_tool_status = {
            "stage2_tools_mode": self.stage2_tools_mode,
            "stage2_tool_required": bool(required_stage2_names),
            "stage2_tool_success": stage2_tool_success,
            "stage2_tool_required_names": sorted(required_stage2_names),
            "stage2_tool_called_names": sorted(
                {
                    str(item.get("tool_name"))
                    for item in stage2_trace
                    if isinstance(item, dict) and not item.get("discarded") and isinstance(item.get("tool_name"), str)
                }
            ),
            "stage2_tool_error_count": len(stage2_errors),
            **step_type_status,
            **route_status,
        }
        fresh_attempts = 0 if from_cache else 1
        while (not parsed.success or not stage2_tool_success) and fresh_attempts < self.max_stage_attempts:
            response_text, attempt, response_meta = self._generate_retry(
                example=example,
                step_index=step_index,
                stage=STAGE_2,
                model=model,
                prompt_text=prompt_text,
                prior_attempt=attempt,
                metadata={
                    "dataset": example.dataset,
                    "example_id": example.example_id,
                    "step_index": step_index,
                    "stage": STAGE_2,
                    "stage2_tools_mode": self.stage2_tools_mode,
                    "tool_call_policy": TOOL_CALL_POLICY_VERSION,
                    "stage2_step_type_mode": self.stage2_step_type_mode,
                    "stage2_router_mode": self.stage2_router_mode,
                    "stage2_step_type": step_type.step_type,
                    "stage2_step_type_risk_flags": list(step_type.risk_flags),
                    "stage2_route_selected_specialists": route_status.get("stage2_route_selected_specialists", []),
                    "stage2_required_tool_names": sorted(required_stage2_names),
                },
                tools=stage2_tooling.schemas or None,
                tool_handlers=stage2_tooling.handlers or None,
                required_tool_names=required_stage2_names,
                cache_context=(
                    f"stage2_tools_mode={self.stage2_tools_mode};"
                    f"tool_call_policy={TOOL_CALL_POLICY_VERSION};"
                    f"stage2_step_type_mode={self.stage2_step_type_mode};"
                    f"stage2_router_mode={self.stage2_router_mode};"
                    f"stage2_step_type={step_type.step_type};"
                    f"stage2_required_tools={','.join(sorted(required_stage2_names))}"
                ),
            )
            fresh_attempts += 1
            parsed = parse_stage2_response(response_text)
            stage2_trace_raw = response_meta.get("tool_trace", [])
            stage2_errors_raw = response_meta.get("tool_errors", [])
            stage2_trace = stage2_trace_raw if isinstance(stage2_trace_raw, list) else []
            stage2_errors = [str(item) for item in stage2_errors_raw] if isinstance(stage2_errors_raw, list) else []
            stage2_tool_success = self._required_tools_success(
                response_meta=response_meta,
                required_tool_names=required_stage2_names,
            )
            stage2_tool_status = {
                "stage2_tools_mode": self.stage2_tools_mode,
                "stage2_tool_required": bool(required_stage2_names),
                "stage2_tool_success": stage2_tool_success,
                "stage2_tool_required_names": sorted(required_stage2_names),
                "stage2_tool_called_names": sorted(
                    {
                        str(item.get("tool_name"))
                        for item in stage2_trace
                        if isinstance(item, dict) and not item.get("discarded") and isinstance(item.get("tool_name"), str)
                    }
                ),
                "stage2_tool_error_count": len(stage2_errors),
                **step_type_status,
                **route_status,
            }
        if not stage2_tool_success:
            degrade_allowed, degrade_reason = self._should_degrade_tool_failure(
                stage="stage2",
                tool_errors=stage2_errors,
                parse_error=parsed.error,
            )
            if degrade_allowed:
                failed_tools = self._extract_failed_tools(
                    required_tool_names=required_stage2_names,
                    tool_trace=stage2_trace,
                    tool_errors=stage2_errors,
                )
                fallback_response, fallback_parsed, fallback_attempt = self._run_stage2_no_tool_fallback(
                    example=example,
                    step_index=step_index,
                    model=model,
                    prompt_text=prompt_text,
                    degrade_reason=degrade_reason,
                    failed_tools=failed_tools,
                )
                if fallback_parsed.success:
                    response_text = fallback_response
                    parsed = fallback_parsed
                    attempt = fallback_attempt
                    stage2_tool_status = {
                        **stage2_tool_status,
                        "stage2_tool_degraded": True,
                        "stage2_tool_degrade_reason": degrade_reason,
                        "stage2_tool_degrade_failed_tools": failed_tools,
                        "stage2_tool_degrade_fallback_success": True,
                    }
                else:
                    stage2_tool_status = {
                        **stage2_tool_status,
                        "stage2_tool_degraded": True,
                        "stage2_tool_degrade_reason": degrade_reason,
                        "stage2_tool_degrade_failed_tools": failed_tools,
                        "stage2_tool_degrade_fallback_success": False,
                        "stage2_tool_degrade_fallback_parse_error": fallback_parsed.error,
                    }
            else:
                stage2_tool_status = {
                    **stage2_tool_status,
                    "stage2_tool_degraded": False,
                    "stage2_tool_degrade_reason": degrade_reason,
                }
        else:
            stage2_tool_status = {
                **stage2_tool_status,
                "stage2_tool_degraded": False,
                "stage2_tool_degrade_reason": "none",
            }
        return response_text, parsed, attempt, prompt_text, stage2_trace, stage2_errors, stage2_tool_status

    def _stage2_review_enabled(self, example: TraceExample) -> bool:
        return example.dataset == DATASET_BIG_BENCH_MISTAKE and self.stage2_tools_mode == "triad"

    @staticmethod
    def _stage2_tool_evidence(tool_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        for item in tool_trace:
            if not isinstance(item, dict) or item.get("discarded"):
                continue
            name = item.get("tool_name")
            result = item.get("result")
            if not isinstance(name, str) or not isinstance(result, dict):
                continue
            evidence.append(
                {
                    "tool_name": name,
                    "status": result.get("status"),
                    "verification_type": result.get("verification_type"),
                    "hard_contradiction": result.get("hard_contradiction", False),
                    "contradictions": result.get("contradictions", []),
                    "checks": result.get("chain_checks")
                    or result.get("substitution_checks")
                    or result.get("checks")
                    or [],
                    "not_verifiable": result.get("not_verifiable", {}),
                    "checked_signal_count": result.get("checked_signal_count", 0),
                    }
                )
        return evidence

    @staticmethod
    def _stage2_specialist_tool_evidence(tool_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
        relevant_tool_names = {
            "alternative_route_verifier_tool",
            "equivalence_substitution_verifier_tool",
            "condition_obligation_verifier_tool",
            "contradiction_probe_tool",
            "bb_arithmetic_chain_tool",
            "bb_variable_state_tool",
            "bb_substitution_tool",
            "bb_decomposition_equivalence_tool",
            "gsm_expr_check_tool",
        }
        evidence: list[dict[str, Any]] = []
        for item in tool_trace:
            if not isinstance(item, dict) or item.get("discarded"):
                continue
            tool_name = item.get("tool_name")
            result = item.get("result")
            if not isinstance(tool_name, str) or tool_name not in relevant_tool_names or not isinstance(result, dict):
                continue
            evidence.append(
                {
                    "tool_name": tool_name,
                    "status": result.get("status"),
                    "verification_type": result.get("verification_type"),
                    "hard_contradiction": result.get("hard_contradiction", False),
                    "valid_alternative": result.get("valid_alternative", False),
                    "valid_equivalent_transformation": result.get("valid_equivalent_transformation", False),
                    "preferred_dimension": result.get("preferred_dimension"),
                    "contradiction_level": result.get("contradiction_level"),
                    "binding_conflict": result.get("binding_conflict"),
                    "contradictions": result.get("contradictions", []),
                    "evidence": result.get("evidence", []),
                    "not_verifiable": result.get("not_verifiable", {}),
                }
            )
        return evidence

    @staticmethod
    def _with_stage2_label(stage2_parse: Stage2Parse, *, dimension: str, label: str) -> Stage2Parse:
        concepts_label = stage2_parse.mathematical_concepts_label
        analyses_label = stage2_parse.key_analyses_label
        calculations_label = stage2_parse.calculations_label

        if dimension == "mathematical_concepts":
            concepts_label = label
        elif dimension == "key_analyses":
            analyses_label = label
        else:
            calculations_label = label

        return Stage2Parse(
            mathematical_concepts_label=concepts_label,
            key_analyses_label=analyses_label,
            calculations_label=calculations_label,
            success=stage2_parse.success,
            error=stage2_parse.error,
        )

    @classmethod
    def _replace_stage2_labels(
        cls,
        stage2_parse: Stage2Parse,
        *,
        source_label: str,
        target_label: str,
    ) -> Stage2Parse:
        updated = stage2_parse
        for dimension, label in stage2_parse.principle_labels.items():
            if label == source_label:
                updated = cls._with_stage2_label(updated, dimension=dimension, label=target_label)
        return updated

    @staticmethod
    def _stage2_specialist_status(tool_trace: list[dict[str, Any]]) -> dict[str, Any]:
        specialist_tool_names = {
            "alternative_route_verifier_tool",
            "equivalence_substitution_verifier_tool",
            "condition_obligation_verifier_tool",
        }
        hard_sources: list[str] = []
        alternative_sources: list[str] = []
        preferred_dimensions: list[str] = []
        evidence: list[str] = []

        for item in tool_trace:
            if not isinstance(item, dict) or item.get("discarded"):
                continue
            tool_name = item.get("tool_name")
            result = item.get("result")
            if not isinstance(tool_name, str) or not isinstance(result, dict):
                continue

            result_evidence = result.get("evidence", [])
            if isinstance(result_evidence, list):
                evidence.extend(str(entry) for entry in result_evidence if isinstance(entry, str))

            hard_contradiction = bool(result.get("hard_contradiction", False))
            if result.get("status") == "hard_contradiction":
                hard_contradiction = True
            if tool_name == "contradiction_probe_tool" and result.get("contradiction_level") == "hard_contradiction":
                hard_contradiction = True
            if tool_name == "condition_binding_tool" and result.get("binding_conflict") == "hard":
                hard_contradiction = True

            if hard_contradiction:
                hard_sources.append(tool_name)
                preferred_dimension = result.get("preferred_dimension")
                if isinstance(preferred_dimension, str) and preferred_dimension:
                    preferred_dimensions.append(preferred_dimension)

            valid_alternative = bool(
                result.get("valid_alternative", False)
                or result.get("valid_equivalent_transformation", False)
                or (tool_name == "equivalence_check_tool" and result.get("relation") == "alternative_valid")
            )
            if valid_alternative and not hard_contradiction:
                alternative_sources.append(tool_name)
                preferred_dimension = result.get("preferred_dimension")
                if isinstance(preferred_dimension, str) and preferred_dimension:
                    preferred_dimensions.append(preferred_dimension)

        return {
            "stage2_specialist_enabled": any(
                isinstance(item, dict) and item.get("tool_name") in specialist_tool_names for item in tool_trace
            ),
            "stage2_specialist_hard_contradiction": len(hard_sources) > 0,
            "stage2_specialist_valid_alternative": len(alternative_sources) > 0,
            "stage2_specialist_hard_sources": hard_sources,
            "stage2_specialist_alternative_sources": alternative_sources,
            "stage2_specialist_preferred_dimensions": preferred_dimensions,
            "stage2_specialist_evidence": evidence,
        }

    @classmethod
    def _apply_stage2_specialist_adjustment(
        cls,
        stage2_parse: Stage2Parse,
        tool_trace: list[dict[str, Any]],
    ) -> tuple[Stage2Parse, dict[str, Any]]:
        status = cls._stage2_specialist_status(tool_trace)
        adjusted = stage2_parse
        action = "none"

        has_contradiction_label = "contradiction-found" in stage2_parse.principle_labels.values()
        has_hard_contradiction = bool(status.get("stage2_specialist_hard_contradiction", False))
        has_valid_alternative = bool(status.get("stage2_specialist_valid_alternative", False))

        if has_valid_alternative and not has_hard_contradiction and has_contradiction_label:
            adjusted = cls._replace_stage2_labels(
                stage2_parse,
                source_label="contradiction-found",
                target_label="reasonable-but-incomplete",
            )
            action = "downgraded_contradiction_from_valid_alternative"
        elif has_hard_contradiction and not has_contradiction_label:
            preferred_dimensions = status.get("stage2_specialist_preferred_dimensions", [])
            preferred_dimension = (
                preferred_dimensions[0]
                if isinstance(preferred_dimensions, list) and preferred_dimensions
                else "calculations"
            )
            adjusted = cls._with_stage2_label(
                stage2_parse,
                dimension=preferred_dimension,
                label="contradiction-found",
            )
            action = f"upgraded_{preferred_dimension}_from_hard_tool_evidence"

        status["stage2_specialist_adjustment_applied"] = action != "none"
        status["stage2_specialist_adjustment_action"] = action
        status["stage2_specialist_adjusted_labels"] = adjusted.principle_labels
        return adjusted, status

    def _run_stage2_review(
        self,
        *,
        example: TraceExample,
        step_index: int,
        model: str,
        stage2_response: str,
        stage2_parse: Stage2Parse,
        stage2_tool_trace: list[dict[str, Any]],
    ) -> tuple[Stage2Parse, str | None, str | None, int, dict[str, Any]]:
        if not self._stage2_review_enabled(example):
            return stage2_parse, None, None, 0, {
                "stage2_review_enabled": False,
                "stage2_review_applied": False,
                "stage2_review_success": True,
            }

        tool_evidence = self._stage2_tool_evidence(stage2_tool_trace)
        prompt_text = self.prompt_builder.build_stage2_review_prompt(
            question=example.question,
            previous_steps=example.steps[:step_index],
            step_index=step_index,
            current_step=example.steps[step_index],
            stage2_response=stage2_response,
            stage2_labels=stage2_parse.principle_labels,
            tool_evidence=tool_evidence,
        )
        response_text, attempt, _, from_cache = self._cached_or_generate(
            example=example,
            step_index=step_index,
            stage=STAGE_2_REVIEW,
            model=model,
            prompt_text=prompt_text,
            metadata={
                "dataset": example.dataset,
                "example_id": example.example_id,
                "step_index": step_index,
                "stage": STAGE_2_REVIEW,
                "stage2_tools_mode": self.stage2_tools_mode,
                "tool_call_policy": TOOL_CALL_POLICY_VERSION,
                "stage2_review": True,
            },
            tools=None,
            tool_handlers=None,
            required_tool_names=None,
            cache_context=(
                f"stage2_review=true;"
                f"stage2_tools_mode={self.stage2_tools_mode};"
                f"tool_call_policy={TOOL_CALL_POLICY_VERSION}"
            ),
        )
        review_parse = parse_stage2_response(response_text)
        fresh_attempts = 0 if from_cache else 1
        while not review_parse.success and fresh_attempts < self.max_stage_attempts:
            response_text, attempt, _ = self._generate_retry(
                example=example,
                step_index=step_index,
                stage=STAGE_2_REVIEW,
                model=model,
                prompt_text=prompt_text,
                prior_attempt=attempt,
                metadata={
                    "dataset": example.dataset,
                    "example_id": example.example_id,
                    "step_index": step_index,
                    "stage": STAGE_2_REVIEW,
                    "stage2_tools_mode": self.stage2_tools_mode,
                    "tool_call_policy": TOOL_CALL_POLICY_VERSION,
                    "stage2_review": True,
                },
                tools=None,
                tool_handlers=None,
                required_tool_names=None,
                cache_context=(
                    f"stage2_review=true;"
                    f"stage2_tools_mode={self.stage2_tools_mode};"
                    f"tool_call_policy={TOOL_CALL_POLICY_VERSION}"
                ),
            )
            fresh_attempts += 1
            review_parse = parse_stage2_response(response_text)

        if not review_parse.success:
            return stage2_parse, prompt_text, response_text, attempt, {
                "stage2_review_enabled": True,
                "stage2_review_applied": False,
                "stage2_review_success": False,
                "stage2_review_error": review_parse.error,
                "stage2_review_fallback_to_original": True,
                "stage2_review_tool_evidence": tool_evidence,
            }

        return review_parse, prompt_text, response_text, attempt, {
            "stage2_review_enabled": True,
            "stage2_review_applied": True,
            "stage2_review_success": True,
            "stage2_review_tool_evidence": tool_evidence,
        }

    @staticmethod
    def _stage2_specialist_review_enabled(
        tool_trace: list[dict[str, Any]],
        stage2_tool_status: dict[str, Any],
    ) -> bool:
        if not bool(stage2_tool_status.get("stage2_tool_required", False)):
            return False
        if not bool(stage2_tool_status.get("stage2_route_trigger_specialist_review", True)):
            return False
        specialist_tools = {
            "alternative_route_verifier_tool",
            "equivalence_substitution_verifier_tool",
            "condition_obligation_verifier_tool",
        }
        for item in tool_trace:
            if not isinstance(item, dict) or item.get("discarded"):
                continue
            tool_name = item.get("tool_name")
            if isinstance(tool_name, str) and tool_name in specialist_tools:
                return True
        return False

    def _run_stage2_specialist_review(
        self,
        *,
        example: TraceExample,
        step_index: int,
        model: str,
        stage2_response: str,
        stage2_parse: Stage2Parse,
        stage2_tool_trace: list[dict[str, Any]],
        stage2_tool_status: dict[str, Any],
    ) -> tuple[Stage2Parse, str | None, str | None, int, dict[str, Any]]:
        if not self._stage2_specialist_review_enabled(stage2_tool_trace, stage2_tool_status):
            return stage2_parse, None, None, 0, {
                "stage2_specialist_review_enabled": False,
                "stage2_specialist_review_applied": False,
                "stage2_specialist_review_success": True,
            }

        specialist_evidence = self._stage2_specialist_tool_evidence(stage2_tool_trace)
        step_type_meta = {
            "step_type": stage2_tool_status.get("stage2_step_type"),
            "reasoning": stage2_tool_status.get("stage2_step_type_reasoning"),
            "risk_flags": stage2_tool_status.get("stage2_step_type_risk_flags", []),
            "specialists": stage2_tool_status.get("stage2_step_type_specialists", []),
            "source": stage2_tool_status.get("stage2_step_type_source"),
            "confidence": stage2_tool_status.get("stage2_step_type_confidence"),
        }
        route_meta = {
            "mode": stage2_tool_status.get("stage2_route_mode"),
            "source": stage2_tool_status.get("stage2_route_source"),
            "confidence": stage2_tool_status.get("stage2_route_confidence"),
            "selected_specialists": stage2_tool_status.get("stage2_route_selected_specialists", []),
            "trigger_specialist_review": stage2_tool_status.get("stage2_route_trigger_specialist_review", False),
            "model_name": stage2_tool_status.get("stage2_route_model_name"),
            "fallback_used": stage2_tool_status.get("stage2_route_fallback_used", False),
            "fallback_reason": stage2_tool_status.get("stage2_route_fallback_reason", ""),
        }
        prompt_text = self.prompt_builder.build_stage2_specialist_review_prompt(
            question=example.question,
            previous_steps=example.steps[:step_index],
            step_index=step_index,
            current_step=example.steps[step_index],
            stage2_response=stage2_response,
            stage2_labels=stage2_parse.principle_labels,
            step_type_meta=step_type_meta,
            route_meta=route_meta,
            specialist_evidence=specialist_evidence,
        )
        response_text, attempt, _, from_cache = self._cached_or_generate(
            example=example,
            step_index=step_index,
            stage=STAGE_2_SPECIALIST_REVIEW,
            model=model,
            prompt_text=prompt_text,
            metadata={
                "dataset": example.dataset,
                "example_id": example.example_id,
                "step_index": step_index,
                "stage": STAGE_2_SPECIALIST_REVIEW,
                "stage2_tools_mode": self.stage2_tools_mode,
                "tool_call_policy": TOOL_CALL_POLICY_VERSION,
                "stage2_specialist_review": True,
                "stage2_step_type": stage2_tool_status.get("stage2_step_type"),
            },
            tools=None,
            tool_handlers=None,
            required_tool_names=None,
            cache_context=(
                f"stage2_specialist_review=true;"
                f"stage2_tools_mode={self.stage2_tools_mode};"
                f"stage2_step_type={stage2_tool_status.get('stage2_step_type', '')};"
                f"tool_call_policy={TOOL_CALL_POLICY_VERSION}"
            ),
        )
        review_parse = parse_stage2_response(response_text)
        fresh_attempts = 0 if from_cache else 1
        while not review_parse.success and fresh_attempts < self.max_stage_attempts:
            response_text, attempt, _ = self._generate_retry(
                example=example,
                step_index=step_index,
                stage=STAGE_2_SPECIALIST_REVIEW,
                model=model,
                prompt_text=prompt_text,
                prior_attempt=attempt,
                metadata={
                    "dataset": example.dataset,
                    "example_id": example.example_id,
                    "step_index": step_index,
                    "stage": STAGE_2_SPECIALIST_REVIEW,
                    "stage2_tools_mode": self.stage2_tools_mode,
                    "tool_call_policy": TOOL_CALL_POLICY_VERSION,
                    "stage2_specialist_review": True,
                    "stage2_step_type": stage2_tool_status.get("stage2_step_type"),
                },
                tools=None,
                tool_handlers=None,
                required_tool_names=None,
                cache_context=(
                    f"stage2_specialist_review=true;"
                    f"stage2_tools_mode={self.stage2_tools_mode};"
                    f"stage2_step_type={stage2_tool_status.get('stage2_step_type', '')};"
                    f"tool_call_policy={TOOL_CALL_POLICY_VERSION}"
                ),
            )
            fresh_attempts += 1
            review_parse = parse_stage2_response(response_text)

        if not review_parse.success:
            return stage2_parse, prompt_text, response_text, attempt, {
                "stage2_specialist_review_enabled": True,
                "stage2_specialist_review_applied": False,
                "stage2_specialist_review_success": False,
                "stage2_specialist_review_error": review_parse.error,
                "stage2_specialist_review_fallback_to_original": True,
                "stage2_specialist_review_evidence": specialist_evidence,
            }

        return review_parse, prompt_text, response_text, attempt, {
            "stage2_specialist_review_enabled": True,
            "stage2_specialist_review_applied": True,
            "stage2_specialist_review_success": True,
            "stage2_specialist_review_evidence": specialist_evidence,
        }

    @staticmethod
    def _step_label(stage2_parse: Stage2Parse) -> int:
        labels = stage2_parse.principle_labels.values()
        return 0 if "contradiction-found" in labels else 1

    def predict_trace(self, example: TraceExample, model: str) -> TracePrediction:
        step_rows: list[dict] = []
        pred_first_mistake_index: int | None = None

        for step_index, _ in enumerate(example.steps):
            (
                stage1_response,
                stage1_parse,
                stage1_attempts,
                stage1_prompt,
                stage1_tool_trace,
                stage1_tool_errors,
                stage1_tool_status,
            ) = self._run_stage1(
                example=example,
                step_index=step_index,
                model=model,
            )
            if not stage1_parse.success:
                parse_status = {
                    "stage1_success": False,
                    "stage2_success": False,
                    **stage1_tool_status,
                }
                step_rows.append(
                    StepPrediction(
                        step_index=step_index,
                        pred_step_label=None,
                        stage1_prompt=stage1_prompt,
                        stage2_prompt=None,
                        stage1_raw_response=stage1_response,
                        stage2_raw_response=None,
                        stage1_parse=stage1_parse.to_dict(),
                        stage2_parse=None,
                        stage1_tool_trace=stage1_tool_trace,
                        stage1_tool_errors=stage1_tool_errors,
                        stage1_attempts=stage1_attempts,
                        stage2_attempts=0,
                        parse_status=parse_status,
                    ).to_dict()
                )
                return TracePrediction(
                    example_id=example.example_id,
                    dataset=example.dataset,
                    model=model,
                    pred_first_mistake_index=None,
                    pred_trace_label=None,
                    gold_first_mistake_index=example.gold_first_mistake_index,
                    gold_trace_label=example.gold_trace_label,
                    steps=step_rows,
                    completed=False,
                    error=stage1_parse.error,
                )

            stage1_tool_success = bool(stage1_tool_status.get("stage1_tool_success", True))
            stage1_degraded_ok = bool(
                stage1_tool_status.get("stage1_tool_degraded", False)
                and stage1_tool_status.get("stage1_tool_degrade_fallback_success", False)
            )
            if (not stage1_tool_success) and (not stage1_degraded_ok):
                parse_status = {
                    "stage1_success": False,
                    "stage2_success": False,
                    **stage1_tool_status,
                }
                step_rows.append(
                    StepPrediction(
                        step_index=step_index,
                        pred_step_label=None,
                        stage1_prompt=stage1_prompt,
                        stage2_prompt=None,
                        stage1_raw_response=stage1_response,
                        stage2_raw_response=None,
                        stage1_parse=stage1_parse.to_dict(),
                        stage2_parse=None,
                        stage1_tool_trace=stage1_tool_trace,
                        stage1_tool_errors=stage1_tool_errors,
                        stage1_attempts=stage1_attempts,
                        stage2_attempts=0,
                        parse_status=parse_status,
                    ).to_dict()
                )
                return TracePrediction(
                    example_id=example.example_id,
                    dataset=example.dataset,
                    model=model,
                    pred_first_mistake_index=None,
                    pred_trace_label=None,
                    gold_first_mistake_index=example.gold_first_mistake_index,
                    gold_trace_label=example.gold_trace_label,
                    steps=step_rows,
                    completed=False,
                    error="Stage-1 required tools failed after retries.",
                )

            (
                stage2_response,
                stage2_parse,
                stage2_attempts,
                stage2_prompt,
                stage2_tool_trace,
                stage2_tool_errors,
                stage2_tool_status,
            ) = self._run_stage2(
                example=example,
                step_index=step_index,
                model=model,
                stage1_parse=stage1_parse,
            )
            stage2_step_type_meta = {
                "reasoning": stage2_tool_status.get("stage2_step_type_reasoning"),
                "risk_flags": list(stage2_tool_status.get("stage2_step_type_risk_flags", [])),
                "specialists": list(stage2_tool_status.get("stage2_step_type_specialists", [])),
                "source": stage2_tool_status.get("stage2_step_type_source"),
                "confidence": stage2_tool_status.get("stage2_step_type_confidence"),
                "mode": stage2_tool_status.get("stage2_step_type_mode"),
                "llm_used": stage2_tool_status.get("stage2_step_type_llm_used", False),
                "llm_success": stage2_tool_status.get("stage2_step_type_llm_success", False),
                "llm_attempts": stage2_tool_status.get("stage2_step_type_llm_attempts", 0),
                "fallback_to_heuristic": stage2_tool_status.get("stage2_step_type_fallback_to_heuristic", False),
                "fallback_reason": stage2_tool_status.get("stage2_step_type_fallback_reason", ""),
                "heuristic_step_type": stage2_tool_status.get("stage2_step_type_heuristic_step_type"),
            }
            stage2_route_meta = {
                "mode": stage2_tool_status.get("stage2_route_mode"),
                "source": stage2_tool_status.get("stage2_route_source"),
                "confidence": stage2_tool_status.get("stage2_route_confidence"),
                "selected_specialists": list(stage2_tool_status.get("stage2_route_selected_specialists", [])),
                "trigger_specialist_review": stage2_tool_status.get("stage2_route_trigger_specialist_review", False),
                "model_name": stage2_tool_status.get("stage2_route_model_name"),
                "candidate_scores": dict(stage2_tool_status.get("stage2_route_candidate_scores", {})),
                "success": stage2_tool_status.get("stage2_route_success", False),
                "error": stage2_tool_status.get("stage2_route_error"),
                "fallback_used": stage2_tool_status.get("stage2_route_fallback_used", False),
                "fallback_reason": stage2_tool_status.get("stage2_route_fallback_reason", ""),
            }
            if not stage2_parse.success:
                step_rows.append(
                    StepPrediction(
                        step_index=step_index,
                        pred_step_label=None,
                        stage1_prompt=stage1_prompt,
                        stage2_prompt=stage2_prompt,
                        stage1_raw_response=stage1_response,
                        stage2_raw_response=stage2_response,
                        stage1_parse=stage1_parse.to_dict(),
                        stage2_parse=stage2_parse.to_dict(),
                        stage2_step_type=stage2_tool_status.get("stage2_step_type"),
                        stage2_step_type_meta=stage2_step_type_meta,
                        stage2_route_meta=stage2_route_meta,
                        principle_labels=stage2_parse.principle_labels,
                        stage1_tool_trace=stage1_tool_trace,
                        stage1_tool_errors=stage1_tool_errors,
                        stage2_tool_trace=stage2_tool_trace,
                        stage2_tool_errors=stage2_tool_errors,
                        stage1_attempts=stage1_attempts,
                        stage2_attempts=stage2_attempts,
                        parse_status={
                            "stage1_success": True,
                            "stage2_success": False,
                            **stage1_tool_status,
                            **stage2_tool_status,
                        },
                    ).to_dict()
                )
                return TracePrediction(
                    example_id=example.example_id,
                    dataset=example.dataset,
                    model=model,
                    pred_first_mistake_index=None,
                    pred_trace_label=None,
                    gold_first_mistake_index=example.gold_first_mistake_index,
                    gold_trace_label=example.gold_trace_label,
                    steps=step_rows,
                    completed=False,
                    error=stage2_parse.error,
                )

            stage2_tool_success = bool(stage2_tool_status.get("stage2_tool_success", True))
            stage2_degraded_ok = bool(
                stage2_tool_status.get("stage2_tool_degraded", False)
                and stage2_tool_status.get("stage2_tool_degrade_fallback_success", False)
            )
            if (not stage2_tool_success) and (not stage2_degraded_ok):
                step_rows.append(
                    StepPrediction(
                        step_index=step_index,
                        pred_step_label=None,
                        stage1_prompt=stage1_prompt,
                        stage2_prompt=stage2_prompt,
                        stage1_raw_response=stage1_response,
                        stage2_raw_response=stage2_response,
                        stage1_parse=stage1_parse.to_dict(),
                        stage2_parse=stage2_parse.to_dict(),
                        stage2_step_type=stage2_tool_status.get("stage2_step_type"),
                        stage2_step_type_meta=stage2_step_type_meta,
                        stage2_route_meta=stage2_route_meta,
                        principle_labels=stage2_parse.principle_labels,
                        stage1_tool_trace=stage1_tool_trace,
                        stage1_tool_errors=stage1_tool_errors,
                        stage2_tool_trace=stage2_tool_trace,
                        stage2_tool_errors=stage2_tool_errors,
                        stage1_attempts=stage1_attempts,
                        stage2_attempts=stage2_attempts,
                        parse_status={
                            "stage1_success": True,
                            "stage2_success": False,
                            **stage1_tool_status,
                            **stage2_tool_status,
                        },
                    ).to_dict()
                )
                return TracePrediction(
                    example_id=example.example_id,
                    dataset=example.dataset,
                    model=model,
                    pred_first_mistake_index=None,
                    pred_trace_label=None,
                    gold_first_mistake_index=example.gold_first_mistake_index,
                    gold_trace_label=example.gold_trace_label,
                    steps=step_rows,
                    completed=False,
                    error="Stage-2 required tools failed after retries.",
                )

            original_stage2_parse = stage2_parse
            (
                reviewed_stage2_parse,
                stage2_review_prompt,
                stage2_review_response,
                stage2_review_attempts,
                stage2_review_status,
            ) = self._run_stage2_review(
                example=example,
                step_index=step_index,
                model=model,
                stage2_response=stage2_response,
                stage2_parse=stage2_parse,
                stage2_tool_trace=stage2_tool_trace,
            )
            (
                specialist_reviewed_stage2_parse,
                stage2_specialist_review_prompt,
                stage2_specialist_review_response,
                stage2_specialist_review_attempts,
                stage2_specialist_review_status,
            ) = self._run_stage2_specialist_review(
                example=example,
                step_index=step_index,
                model=model,
                stage2_response=stage2_response,
                stage2_parse=reviewed_stage2_parse,
                stage2_tool_trace=stage2_tool_trace,
                stage2_tool_status=stage2_tool_status,
            )
            stage2_parse, stage2_specialist_status = self._apply_stage2_specialist_adjustment(
                specialist_reviewed_stage2_parse,
                stage2_tool_trace,
            )

            pred_step_label = self._step_label(stage2_parse)
            step_prediction = StepPrediction(
                step_index=step_index,
                pred_step_label=pred_step_label,
                stage1_prompt=stage1_prompt,
                stage2_prompt=stage2_prompt,
                stage1_raw_response=stage1_response,
                stage2_raw_response=stage2_response,
                stage1_parse=stage1_parse.to_dict(),
                stage2_parse=stage2_parse.to_dict(),
                stage2_step_type=stage2_tool_status.get("stage2_step_type"),
                stage2_step_type_meta=stage2_step_type_meta,
                stage2_route_meta=stage2_route_meta,
                principle_labels=stage2_parse.principle_labels,
                stage1_tool_trace=stage1_tool_trace,
                stage1_tool_errors=stage1_tool_errors,
                stage2_tool_trace=stage2_tool_trace,
                stage2_tool_errors=stage2_tool_errors,
                stage1_attempts=stage1_attempts,
                stage2_attempts=stage2_attempts,
                stage2_review_prompt=stage2_review_prompt,
                stage2_review_raw_response=stage2_review_response,
                stage2_review_parse=reviewed_stage2_parse.to_dict() if stage2_review_prompt is not None else None,
                stage2_review_attempts=stage2_review_attempts,
                stage2_original_parse=original_stage2_parse.to_dict()
                if stage2_review_prompt is not None
                else None,
                stage2_review_applied=bool(stage2_review_status.get("stage2_review_applied", False)),
                stage2_specialist_review_prompt=stage2_specialist_review_prompt,
                stage2_specialist_review_raw_response=stage2_specialist_review_response,
                stage2_specialist_review_parse=(
                    specialist_reviewed_stage2_parse.to_dict()
                    if stage2_specialist_review_prompt is not None
                    else None
                ),
                stage2_specialist_review_attempts=stage2_specialist_review_attempts,
                stage2_specialist_review_applied=bool(
                    stage2_specialist_review_status.get("stage2_specialist_review_applied", False)
                ),
                parse_status={
                    "stage1_success": True,
                    "stage2_success": True,
                    **stage1_tool_status,
                    **stage2_tool_status,
                    **stage2_review_status,
                    **stage2_specialist_review_status,
                    **stage2_specialist_status,
                },
            )
            step_rows.append(step_prediction.to_dict())

            if pred_step_label == 0:
                pred_first_mistake_index = step_index
                break

        pred_trace_label = POSITIVE_TRACE_LABEL if pred_first_mistake_index is None else NEGATIVE_TRACE_LABEL
        return TracePrediction(
            example_id=example.example_id,
            dataset=example.dataset,
            model=model,
            pred_first_mistake_index=pred_first_mistake_index,
            pred_trace_label=pred_trace_label,
            gold_first_mistake_index=example.gold_first_mistake_index,
            gold_trace_label=example.gold_trace_label,
            steps=step_rows,
            completed=True,
        )
