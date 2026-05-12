from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from tenacity import RetryError, Retrying, stop_after_attempt, wait_exponential


class OpenAIResponsesClient:
    def __init__(self, timeout: float = 120.0) -> None:
        load_dotenv()
        self.timeout = timeout
        self._client = None
        self._resolved_config: dict[str, str] | None = None

    @staticmethod
    def _provider_defaults(provider: str) -> dict[str, str]:
        normalized = provider.strip().lower()
        if normalized == "siliconflow":
            return {
                "base_url": "https://api.siliconflow.cn/v1",
                "api_style": "chat_completions",
            }
        if normalized == "aihubmix":
            return {
                "base_url": "https://aihubmix.com/v1",
                "api_style": "chat_completions",
            }
        if normalized == "autodl":
            return {
                "base_url": "https://www.autodl.art/api/v1",
                "api_style": "chat_completions",
            }
        if normalized == "volcengine":
            return {
                "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                "api_style": "auto",
            }
        return {
            "base_url": "",
            "api_style": "auto",
        }

    def _resolve_configuration(self) -> dict[str, str]:
        if self._resolved_config is not None:
            return self._resolved_config

        provider = os.getenv("MATHCHECKER_PROVIDER", "").strip().lower()
        if provider:
            defaults = self._provider_defaults(provider)
            if provider == "volcengine":
                api_key = os.getenv("VOLCENGINE_API_KEY", "").strip() or os.getenv("ARK_API_KEY", "").strip()
                base_url = (
                    os.getenv("VOLCENGINE_BASE_URL", "").strip()
                    or os.getenv("ARK_BASE_URL", "").strip()
                    or defaults["base_url"]
                )
                api_style = (
                    os.getenv("VOLCENGINE_API_STYLE", "").strip().lower()
                    or os.getenv("ARK_API_STYLE", "").strip().lower()
                    or defaults["api_style"]
                )
            else:
                prefix = provider.upper()
                api_key = os.getenv(f"{prefix}_API_KEY", "").strip()
                base_url = os.getenv(f"{prefix}_BASE_URL", "").strip() or defaults["base_url"]
                api_style = os.getenv(f"{prefix}_API_STYLE", "").strip().lower() or defaults["api_style"]
            if api_key:
                self._resolved_config = {
                    "provider": provider,
                    "api_key": api_key,
                    "base_url": base_url,
                    "api_style": api_style,
                }
                return self._resolved_config

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        api_style = os.getenv("OPENAI_API_STYLE", "auto").strip().lower()
        self._resolved_config = {
            "provider": provider or "generic",
            "api_key": api_key,
            "base_url": base_url,
            "api_style": api_style,
        }
        return self._resolved_config

    def validate_configuration(self) -> None:
        self._get_client()

    def _get_client(self):
        config = self._resolve_configuration()
        api_key = config["api_key"]
        if not api_key:
            provider = config["provider"]
            if provider and provider != "generic":
                if provider == "volcengine":
                    raise RuntimeError(
                        "VOLCENGINE_API_KEY or ARK_API_KEY is not set for MATHCHECKER_PROVIDER=volcengine."
                    )
                raise RuntimeError(f"{provider.upper()}_API_KEY is not set for MATHCHECKER_PROVIDER={provider}.")
            raise RuntimeError("OPENAI_API_KEY is not set.")
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai package is not installed. Run `pip install -e .`.") from exc

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": self.timeout,
        }
        base_url = config["base_url"]
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)
        return self._client

    @staticmethod
    def _extract_chat_completion_text(response: Any) -> str:
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")
        if not choices:
            return ""

        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message", {})
        else:
            message = getattr(first_choice, "message", None)

        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)

        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            texts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if isinstance(block.get("text"), str):
                        texts.append(block["text"])
                    elif block.get("type") == "text" and isinstance(block.get("text"), str):
                        texts.append(block["text"])
                else:
                    text = getattr(block, "text", None)
                    if isinstance(text, str):
                        texts.append(text)
            return "\n".join(texts).strip()
        return ""

    @staticmethod
    def _extract_chat_completion_message(response: Any) -> Any:
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")
        if not choices:
            return None
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            return first_choice.get("message")
        return getattr(first_choice, "message", None)

    @staticmethod
    def _message_content_to_text(message: Any) -> str:
        if message is None:
            return ""
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            texts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        texts.append(text)
                else:
                    text = getattr(block, "text", None)
                    if isinstance(text, str):
                        texts.append(text)
            return "\n".join(texts).strip()
        return ""

    @staticmethod
    def _message_reasoning_content_to_text(message: Any) -> str:
        if message is None:
            return ""
        if isinstance(message, dict):
            reasoning_content = message.get("reasoning_content")
        else:
            reasoning_content = getattr(message, "reasoning_content", None)
        if isinstance(reasoning_content, str):
            return reasoning_content
        if isinstance(reasoning_content, list):
            texts: list[str] = []
            for block in reasoning_content:
                if isinstance(block, dict):
                    text = block.get("text") or block.get("reasoning_content")
                    if isinstance(text, str):
                        texts.append(text)
                else:
                    text = getattr(block, "text", None) or getattr(block, "reasoning_content", None)
                    if isinstance(text, str):
                        texts.append(text)
            return "".join(texts)
        return ""

    @staticmethod
    def _message_tool_calls(message: Any) -> list[dict[str, Any]]:
        if message is None:
            return []
        if isinstance(message, dict):
            raw_tool_calls = message.get("tool_calls", []) or []
        else:
            raw_tool_calls = getattr(message, "tool_calls", None) or []

        normalized: list[dict[str, Any]] = []
        for index, tool_call in enumerate(raw_tool_calls):
            if isinstance(tool_call, dict):
                call_id = tool_call.get("id")
                function_payload = tool_call.get("function", {})
                name = function_payload.get("name")
                arguments = function_payload.get("arguments", "{}")
            else:
                call_id = getattr(tool_call, "id", None)
                function_payload = getattr(tool_call, "function", None)
                if isinstance(function_payload, dict):
                    name = function_payload.get("name")
                    arguments = function_payload.get("arguments", "{}")
                else:
                    name = getattr(function_payload, "name", None)
                    arguments = getattr(function_payload, "arguments", "{}")
            if not call_id:
                call_id = f"call_{index}"
            normalized.append(
                {
                    "id": str(call_id),
                    "name": str(name) if name is not None else "",
                    "arguments": arguments,
                }
            )
        return normalized

    @staticmethod
    def _needs_siliconflow_reasoning_replay(*, provider: str, model: str) -> bool:
        if provider.strip().lower() != "siliconflow":
            return False
        normalized_model = model.strip().lower()
        return "deepseek-v3.2" in normalized_model or "glm-4.7" in normalized_model

    @staticmethod
    def _attach_reasoning_content(
        assistant_message: dict[str, Any],
        *,
        reasoning_content: str,
        preserve_reasoning_content: bool,
    ) -> dict[str, Any]:
        if preserve_reasoning_content and reasoning_content:
            assistant_message["reasoning_content"] = reasoning_content
        return assistant_message

    @staticmethod
    def _extract_output_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text
        output = getattr(response, "output", None) or []
        texts: list[str] = []
        for item in output:
            if isinstance(item, dict):
                content = item.get("content", [])
            else:
                content = getattr(item, "content", None) or []
            for block in content:
                text = getattr(block, "text", None)
                if text is None and isinstance(block, dict):
                    text = block.get("text")
                if text:
                    texts.append(text)
        return "\n".join(texts).strip()

    def _request_chat_completion(
        self,
        client: Any,
        model: str,
        prompt: str,
        *,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> Any:
        payload_messages = messages if messages is not None else [{"role": "user", "content": prompt}]
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": payload_messages,
            "temperature": 0,
        }
        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = tool_choice or "required"
        return client.chat.completions.create(**request_kwargs)

    def _request_responses(self, client: Any, model: str, prompt: str) -> Any:
        return client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
            temperature=0,
        )

    @staticmethod
    def _sanitize_prompt_for_gateway(prompt: str) -> str:
        """Normalize prompt text for provider gateways with strict parameter validators."""
        cleaned = prompt
        cleaned = re.sub(r"\^\{\\prime\}", "'", cleaned)
        cleaned = re.sub(r"\^\\prime", "'", cleaned)
        cleaned = cleaned.replace("\\prime", "'")
        cleaned = cleaned.replace("$", "")
        # Fall back to plain-text math commands when gateways reject backslash tokens.
        cleaned = re.sub(r"\\([A-Za-z]+)", r"\1", cleaned)
        return cleaned

    @staticmethod
    def _looks_like_gateway_parameter_error(exc: Exception) -> bool:
        message = str(exc)
        lower = message.lower()
        if "apitype:openai.chat parameter" in lower:
            return True
        if "bad_response_status_code" in lower:
            return True
        if "invalid_request_error" in lower and "parameter" in lower:
            return True
        return False

    def _request_with_retries(
        self,
        *,
        client: Any,
        model: str,
        prompt: str,
        api_style: str,
        use_tools: bool,
        tools: list[dict[str, Any]] | None,
        handlers: dict[str, Any],
        required: set[str],
    ) -> tuple[Any, str | None, str, list[dict[str, Any]], list[str]]:
        last_style: str | None = None
        text = ""
        tool_trace: list[dict[str, Any]] = []
        tool_errors: list[str] = []
        response: Any | None = None

        for attempt in Retrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            reraise=True,
        ):
            with attempt:
                if api_style == "chat_completions":
                    last_style = "chat_completions"
                    if use_tools:
                        text, response, tool_trace, tool_errors = self._run_chat_completion_with_tools(
                            client=client,
                            model=model,
                            prompt=prompt,
                            tools=tools or [],
                            tool_handlers=handlers,
                            required_tool_names=required,
                        )
                    else:
                        response = self._request_chat_completion(client, model, prompt)
                        text = self._extract_chat_completion_text(response)
                elif api_style == "responses":
                    last_style = "responses"
                    response = self._request_responses(client, model, prompt)
                    text = self._extract_output_text(response)
                    if use_tools:
                        tool_errors = ["Tool-calling is not supported when using responses API style."]
                        tool_trace = []
                else:
                    try:
                        last_style = "chat_completions"
                        if use_tools:
                            text, response, tool_trace, tool_errors = self._run_chat_completion_with_tools(
                                client=client,
                                model=model,
                                prompt=prompt,
                                tools=tools or [],
                                tool_handlers=handlers,
                                required_tool_names=required,
                            )
                        else:
                            response = self._request_chat_completion(client, model, prompt)
                            text = self._extract_chat_completion_text(response)
                    except Exception:  # noqa: BLE001
                        last_style = "responses"
                        response = self._request_responses(client, model, prompt)
                        text = self._extract_output_text(response)
                        if use_tools:
                            tool_errors = ["Provider fallback to responses API skipped tool-calling."]
                            tool_trace = []

        assert response is not None
        return response, last_style, text, tool_trace, tool_errors

    @staticmethod
    def _parse_tool_arguments(raw_arguments: Any) -> tuple[dict[str, Any], str | None]:
        if isinstance(raw_arguments, dict):
            return raw_arguments, None
        if not isinstance(raw_arguments, str):
            return {}, "Tool arguments must be a JSON object string."
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            return {}, f"Tool arguments are not valid JSON: {exc.msg}"
        if not isinstance(parsed, dict):
            return {}, "Tool arguments JSON must decode into an object."
        return parsed, None

    @staticmethod
    def _tool_schema_name(tool_schema: dict[str, Any]) -> str:
        function = tool_schema.get("function", {})
        name = function.get("name")
        return str(name) if isinstance(name, str) else ""

    @staticmethod
    def _tool_phase_instruction(tool_name: str) -> str:
        if tool_name == "logic_check_tool":
            return (
                "Call logic_check_tool now. This call is only for Stage1 sections 1 and 2:\n"
                "1. Mathematical Concepts to Apply\n"
                "2. Key Analyses for the Next Step\n"
                "Do not use this call for section 3 calculations."
            )
        if tool_name == "domain_guard_tool":
            return (
                "Call domain_guard_tool now. This call is for Stage1 section 3 calculations.\n"
                "Check local domain constraints (log/sqrt/division) and return auditable constraints/conflicts."
            )
        if tool_name == "symbolic_relation_tool":
            return (
                "Call symbolic_relation_tool now. This call is for Stage1 section 3 calculations.\n"
                "Verify local symbolic/algebraic relations for extracted expressions."
            )
        if tool_name == "python_calc_tool":
            return (
                "Call python_calc_tool now. This call is only for Stage1 section 3:\n"
                "3. Mathematical Expressions to Compute\n"
                "Focus on local expression verification and computed results."
            )
        if tool_name == "prm_constraint_tool":
            return (
                "Call prm_constraint_tool now. This call is for Stage1 section 3 calculations.\n"
                "Do local constraint/numeric-symbolic verification for unresolved expressions."
            )
        if tool_name == "unit_ratio_tool":
            return (
                "Call unit_ratio_tool now. This call is for Stage1 section 3 calculations.\n"
                "Extract unit or ratio conversion evidence when relevant."
            )
        if tool_name == "gsm_expr_reference_tool":
            return (
                "Call gsm_expr_reference_tool now. This call is for Stage1 section 3 calculations.\n"
                "Verify GSM8K-style calculation tags/equalities from previous steps and summarize numeric state."
            )
        if tool_name == "answer_obligation_tool":
            return (
                "Call answer_obligation_tool now for the current step.\n"
                "Check whether the step fulfills the obligation to provide a valid answer/conclusion."
            )
        if tool_name == "condition_binding_tool":
            return (
                "Call condition_binding_tool now for the current step.\n"
                "Check variable-condition binding consistency against question conditions."
            )
        if tool_name == "contradiction_probe_tool":
            return (
                "Call contradiction_probe_tool now for the current step.\n"
                "Probe for hard local contradiction evidence. Return evidence only; do not decide the final label."
            )
        if tool_name == "equivalence_check_tool":
            return (
                "Call equivalence_check_tool now for the current step.\n"
                "Check alignment or alternative-path signals as advisory evidence only."
            )
        if tool_name == "alternative_route_verifier_tool":
            return (
                "Call alternative_route_verifier_tool now for the current step.\n"
                "Check whether the step is a different-but-valid solution route.\n"
                "Do not treat reference mismatch alone as contradiction."
            )
        if tool_name == "equivalence_substitution_verifier_tool":
            return (
                "Call equivalence_substitution_verifier_tool now for the current step.\n"
                "Check whether rewrites, substitutions, or decompositions are equivalent or provably false."
            )
        if tool_name == "condition_obligation_verifier_tool":
            return (
                "Call condition_obligation_verifier_tool now for the current step.\n"
                "Check question-condition consistency and whether a local conclusion/branch is auditable."
            )
        if tool_name == "gsm_expr_check_tool":
            return (
                "Call gsm_expr_check_tool now for the current step.\n"
                "Check GSM8K-style <<expression=result>> tags and explicit arithmetic equalities."
            )
        if tool_name == "gsm_final_answer_check_tool":
            return (
                "Call gsm_final_answer_check_tool now for the current step.\n"
                "If a final answer is stated, compare it with locally verified calculation evidence.\n"
                "This tool is advisory only; do not use it alone to decide contradiction-found."
            )
        if tool_name == "gsm_unsupported_number_tool":
            return (
                "Call gsm_unsupported_number_tool now for the current step.\n"
                "Conservatively check whether numeric claims are grounded in the question or prior calculations.\n"
                "This tool is advisory only; do not use it alone to decide contradiction-found."
            )
        return f"Call {tool_name} now."

    @staticmethod
    def _parse_formatted_steps(text: str) -> list[str]:
        cleaned = text.strip()
        if not cleaned:
            return []
        marker_iter = list(re.finditer(r"\(step\s+\d+\)\s", cleaned))
        if not marker_iter:
            return [cleaned]
        parts: list[str] = []
        for index, marker in enumerate(marker_iter):
            start = marker.end()
            end = marker_iter[index + 1].start() if index + 1 < len(marker_iter) else len(cleaned)
            chunk = cleaned[start:end].strip()
            if chunk:
                parts.append(chunk)
        return parts

    @classmethod
    def _extract_prompt_context(cls, prompt: str) -> dict[str, Any]:
        question_match = re.search(r"Question:\s*(.*?)\nInitial steps:\n", prompt, flags=re.S)
        question = question_match.group(1).strip() if question_match else ""

        current_step_match = re.search(
            r"The actual next step is:\n(.*?)\nExecute the following instructions sequentially\.",
            prompt,
            flags=re.S,
        )
        current_step_block = current_step_match.group(1).strip() if current_step_match else ""
        current_step = re.sub(r"^\(step\s+\d+\)\s*", "", current_step_block).strip()

        if current_step_match:
            initial_match = re.search(r"Initial steps:\n(.*?)\nThe actual next step is:\n", prompt, flags=re.S)
        else:
            initial_match = re.search(
                r"Initial steps:\n(.*?)\nExecute the following instructions sequentially\.",
                prompt,
                flags=re.S,
            )
        initial_block = initial_match.group(1).strip() if initial_match else ""
        previous_steps = cls._parse_formatted_steps(initial_block)

        return {
            "question": question,
            "previous_steps": previous_steps,
            "current_step": current_step,
        }

    @staticmethod
    def _extract_candidate_expressions(text: str, limit: int = 12) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()

        def add(raw: str) -> None:
            item = raw.strip().rstrip(".,;:")
            if not item:
                return
            if len(item) > 220:
                return
            if item in seen:
                return
            seen.add(item)
            candidates.append(item)

        for segment in re.findall(r"\$([^$]+)\$", text, flags=re.S):
            for chunk in re.split(r"[,\n;]", segment):
                if any(op in chunk for op in ["+", "-", "*", "/", "^", "=", "<", ">", r"\frac", r"\sqrt"]):
                    add(chunk)

        for match in re.finditer(r"([A-Za-z0-9\\\(\)\^\+\-\*/\.]+(?:<=|>=|=|<|>)[A-Za-z0-9\\\(\)\^\+\-\*/\.]+)", text):
            add(match.group(1))

        for match in re.finditer(
            r"(?<![A-Za-z\\])(-?\d+(?:\.\d+)?(?:\s*[\+\-\*/]\s*-?\d+(?:\.\d+)?)+)(?![A-Za-z])",
            text,
        ):
            add(match.group(1))

        if not candidates:
            for match in re.finditer(r"([A-Za-z][A-Za-z0-9]*\s*\([^)]+\)\s*=\s*[^,\n]+)", text):
                add(match.group(1))
                if len(candidates) >= limit:
                    break

        return candidates[:limit]

    @classmethod
    def _fallback_tool_arguments(cls, *, tool_name: str, prompt: str) -> dict[str, Any] | None:
        context = cls._extract_prompt_context(prompt)
        question = str(context.get("question", ""))
        previous_steps = context.get("previous_steps")
        current_step = str(context.get("current_step", ""))
        if not isinstance(previous_steps, list):
            previous_steps = []
        safe_previous_steps = [str(step) for step in previous_steps]

        if tool_name == "logic_check_tool":
            return {
                "question": question,
                "previous_steps": safe_previous_steps,
                "candidate_focus": "next-step planning",
            }
        if tool_name == "python_calc_tool":
            source = "\n".join([question, *safe_previous_steps]).strip()
            expressions = cls._extract_candidate_expressions(source)
            return {"expressions": expressions}
        if tool_name == "prm_constraint_tool":
            source = "\n".join([question, *safe_previous_steps]).strip()
            expressions = cls._extract_candidate_expressions(source)
            return {
                "question": question,
                "previous_steps": safe_previous_steps,
                "expressions": expressions,
            }
        if tool_name in {"domain_guard_tool", "symbolic_relation_tool", "unit_ratio_tool"}:
            source = "\n".join([question, *safe_previous_steps]).strip()
            expressions = cls._extract_candidate_expressions(source)
            return {
                "question": question,
                "previous_steps": safe_previous_steps,
                "expressions": expressions,
            }
        if tool_name == "gsm_expr_reference_tool":
            return {
                "question": question,
                "previous_steps": safe_previous_steps,
            }
        if tool_name in {
            "answer_obligation_tool",
            "condition_binding_tool",
            "contradiction_probe_tool",
            "equivalence_check_tool",
            "alternative_route_verifier_tool",
            "equivalence_substitution_verifier_tool",
            "condition_obligation_verifier_tool",
            "gsm_expr_check_tool",
            "gsm_final_answer_check_tool",
            "gsm_unsupported_number_tool",
            "bb_arithmetic_chain_tool",
            "bb_variable_state_tool",
            "bb_substitution_tool",
            "bb_decomposition_equivalence_tool",
        }:
            return {
                "question": question,
                "current_step": current_step,
                "previous_steps": safe_previous_steps,
            }
        return None

    @staticmethod
    def _tool_order(required: set[str], available: list[dict[str, Any]]) -> list[str]:
        available_names = [OpenAIResponsesClient._tool_schema_name(schema) for schema in available]
        ordered: list[str] = []
        for preferred in [
            "logic_check_tool",
            "domain_guard_tool",
            "symbolic_relation_tool",
            "python_calc_tool",
            "prm_constraint_tool",
            "unit_ratio_tool",
            "gsm_expr_reference_tool",
            "answer_obligation_tool",
            "condition_binding_tool",
            "contradiction_probe_tool",
            "equivalence_check_tool",
            "alternative_route_verifier_tool",
            "equivalence_substitution_verifier_tool",
            "condition_obligation_verifier_tool",
            "gsm_expr_check_tool",
            "gsm_final_answer_check_tool",
            "gsm_unsupported_number_tool",
            "bb_decomposition_equivalence_tool",
            "bb_arithmetic_chain_tool",
            "bb_variable_state_tool",
            "bb_substitution_tool",
        ]:
            if preferred in required and preferred in available_names:
                ordered.append(preferred)
        for name in available_names:
            if name in required and name not in ordered:
                ordered.append(name)
        return ordered

    def _execute_tool_call(
        self,
        *,
        round_index: int,
        tool_call: dict[str, Any],
        tool_handlers: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], str | None]:
        call_id = tool_call["id"]
        tool_name = tool_call["name"]
        raw_arguments = tool_call.get("arguments", "{}")
        arguments, args_error = self._parse_tool_arguments(raw_arguments)
        error: str | None = args_error
        if error is None:
            handler = tool_handlers.get(tool_name)
            if handler is None:
                error = f"Unknown tool requested by model: {tool_name}"
                result: dict[str, Any] = {"status": "error", "error": error}
            else:
                try:
                    result = handler(arguments)
                except Exception as exc:  # noqa: BLE001
                    error = f"{tool_name} execution failed: {exc}"
                    result = {"status": "error", "error": error}
        else:
            result = {"status": "error", "error": error}

        trace_row = {
            "round": round_index,
            "tool_call_id": call_id,
            "tool_name": tool_name,
            "args": arguments,
            "result": result,
            "error": error,
        }
        tool_message = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(result, ensure_ascii=False),
        }
        return trace_row, tool_message, error

    @staticmethod
    def _tool_result_has_payload(result: dict[str, Any]) -> bool:
        evidence_fields = (
            "auditable_conclusions",
            "evidence",
            "numeric_values",
            "symbolic_results",
            "not_verifiable",
            "domain_constraints",
            "conflicts",
            "conversion_hints",
            "required_focus",
        )
        for field in evidence_fields:
            value = result.get(field)
            if isinstance(value, (list, dict, tuple, set)) and len(value) > 0:
                return True
            if isinstance(value, str) and value.strip():
                return True

        count_fields = (
            "verified_count",
            "not_verifiable_count",
            "verified_numeric_count",
            "verified_symbolic_count",
            "checked_signal_count",
        )
        for field in count_fields:
            value = result.get(field)
            if isinstance(value, (int, float)) and value > 0:
                return True
        return False

    @classmethod
    def _is_effective_tool_result(
        cls,
        *,
        tool_name: str,
        result: dict[str, Any],
        error: str | None,
    ) -> bool:
        if error:
            return False
        status = result.get("status")
        if not isinstance(status, str) or not status.strip():
            return False
        if status in {"tool_error", "error"}:
            return False

        # Stage-2 tools can legitimately return a clean "no conflict" result with
        # little evidence text; the status fields are still auditable.
        if tool_name in {
            "answer_obligation_tool",
            "condition_binding_tool",
            "contradiction_probe_tool",
            "equivalence_check_tool",
            "gsm_expr_check_tool",
            "gsm_final_answer_check_tool",
            "gsm_unsupported_number_tool",
            "bb_arithmetic_chain_tool",
            "bb_variable_state_tool",
            "bb_substitution_tool",
            "bb_decomposition_equivalence_tool",
        }:
            return True
        if tool_name == "logic_check_tool":
            return True
        return cls._tool_result_has_payload(result)

    def _execute_fallback_tool_call(
        self,
        *,
        round_index: int,
        tool_name: str,
        prompt: str,
        tool_handlers: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
        fallback_args = self._fallback_tool_arguments(tool_name=tool_name, prompt=prompt)
        if fallback_args is None:
            return None, None, f"Required tool was not called after retries: {tool_name}"
        fallback_call_id = f"fallback_{round_index}_{tool_name}"
        trace_row, tool_message, error = self._execute_tool_call(
            round_index=round_index,
            tool_call={
                "id": fallback_call_id,
                "name": tool_name,
                "arguments": fallback_args,
            },
            tool_handlers=tool_handlers,
        )
        trace_row["fallback"] = True
        return trace_row, tool_message, error

    def _run_required_tools_then_finalize(
        self,
        *,
        client: Any,
        model: str,
        prompt: str,
        tools: list[dict[str, Any]],
        tool_handlers: dict[str, Any],
        required_tool_names: set[str],
    ) -> tuple[str, Any, list[dict[str, Any]], list[str]]:
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        tool_trace: list[dict[str, Any]] = []
        tool_errors: list[str] = []
        round_index = 0
        final_response: Any | None = None

        available_by_name = {self._tool_schema_name(schema): schema for schema in tools}
        ordered_required = self._tool_order(required_tool_names, tools)
        config = self._resolve_configuration()
        preserve_reasoning_content = self._needs_siliconflow_reasoning_replay(
            provider=config["provider"],
            model=model,
        )

        missing_required = sorted(required_tool_names - set(ordered_required))
        for missing_name in missing_required:
            tool_errors.append(f"Required tool schema is missing: {missing_name}")

        for tool_name in ordered_required:
            tool_schema = available_by_name.get(tool_name)
            if tool_schema is None:
                continue

            messages.append({"role": "user", "content": self._tool_phase_instruction(tool_name)})
            round_index += 1
            response = self._request_chat_completion(
                client=client,
                model=model,
                prompt=prompt,
                messages=messages,
                tools=[tool_schema],
                tool_choice="required",
            )
            final_response = response
            message = self._extract_chat_completion_message(response)
            assistant_text = self._message_content_to_text(message)
            reasoning_content = self._message_reasoning_content_to_text(message)
            tool_calls_all = self._message_tool_calls(message)
            tool_calls = [item for item in tool_calls_all if item.get("name") == tool_name]

            effective_trace: dict[str, Any] | None = None
            effective_tool_message: dict[str, Any] | None = None
            effective_assistant_message: dict[str, Any] | None = None
            if tool_calls:
                tool_call = tool_calls[0]
                call_id = tool_call["id"]
                raw_arguments = tool_call.get("arguments", "{}")
                candidate_assistant_message = {
                    "role": "assistant",
                    "content": assistant_text or "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": raw_arguments if isinstance(raw_arguments, str) else json.dumps(raw_arguments),
                            },
                        }
                    ],
                }
                self._attach_reasoning_content(
                    candidate_assistant_message,
                    reasoning_content=reasoning_content,
                    preserve_reasoning_content=preserve_reasoning_content,
                )
                trace_row, tool_message, error = self._execute_tool_call(
                    round_index=round_index,
                    tool_call=tool_call,
                    tool_handlers=tool_handlers,
                )
                if self._is_effective_tool_result(
                    tool_name=tool_name,
                    result=trace_row.get("result", {}),
                    error=error,
                ):
                    effective_trace = trace_row
                    effective_tool_message = tool_message
                    effective_assistant_message = candidate_assistant_message
                else:
                    trace_row["discarded"] = True
                    trace_row["discard_reason"] = "ineffective_or_error_replaced_by_fallback"
                    tool_trace.append(trace_row)
            else:
                messages.append(
                    self._attach_reasoning_content(
                        {"role": "assistant", "content": assistant_text or ""},
                        reasoning_content=reasoning_content,
                        preserve_reasoning_content=preserve_reasoning_content,
                    )
                )

            if effective_trace is None:
                round_index += 1
                fallback_trace, fallback_message, fallback_error = self._execute_fallback_tool_call(
                    round_index=round_index,
                    tool_name=tool_name,
                    prompt=prompt,
                    tool_handlers=tool_handlers,
                )
                if fallback_trace is None or fallback_message is None:
                    tool_errors.append(fallback_error or f"Required tool was not called after retries: {tool_name}")
                    continue
                fallback_assistant_message = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": fallback_trace["tool_call_id"],
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(fallback_trace.get("args", {}), ensure_ascii=False),
                            },
                        }
                    ],
                }
                if self._is_effective_tool_result(
                    tool_name=tool_name,
                    result=fallback_trace.get("result", {}),
                    error=fallback_error,
                ):
                    effective_trace = fallback_trace
                    effective_tool_message = fallback_message
                    effective_assistant_message = fallback_assistant_message
                else:
                    tool_errors.append(f"{tool_name} did not return an effective result.")
                    effective_trace = fallback_trace
                    effective_tool_message = fallback_message
                    effective_assistant_message = fallback_assistant_message

            tool_trace.append(effective_trace)
            messages.append(effective_assistant_message)
            messages.append(effective_tool_message)

        messages.append(
            {
                "role": "user",
                "content": self._finalization_instruction(required_tool_names),
            }
        )
        final_response = self._request_chat_completion(
            client=client,
            model=model,
            prompt=prompt,
            messages=messages,
            tools=None,
            tool_choice=None,
        )
        final_text = self._extract_chat_completion_text(final_response)
        return final_text, final_response, tool_trace, tool_errors

    @staticmethod
    def _finalization_instruction(required_tool_names: set[str]) -> str:
        stage1_tools = {
            "logic_check_tool",
            "python_calc_tool",
            "prm_constraint_tool",
            "domain_guard_tool",
            "symbolic_relation_tool",
            "unit_ratio_tool",
            "gsm_expr_reference_tool",
        }
        if required_tool_names and required_tool_names.issubset(stage1_tools):
            return (
                "Now provide the final response for the original prompt.\n"
                "Use logic_check_tool outputs for sections 1 and 2.\n"
                "Use Stage1 math-tool outputs for section 3.\n"
                "Output exactly the required three sections and do not call tools."
            )
        gsm_stage2_tools = {
            "gsm_expr_check_tool",
            "gsm_final_answer_check_tool",
            "gsm_unsupported_number_tool",
        }
        specialist_stage2_tools = {
            "alternative_route_verifier_tool",
            "equivalence_substitution_verifier_tool",
            "condition_obligation_verifier_tool",
        }
        if required_tool_names and required_tool_names.issubset(gsm_stage2_tools | specialist_stage2_tools):
            return (
                "Now provide the final response for the original prompt.\n"
                "Use gsm_expr_check_tool arithmetic contradictions as hard calculation evidence.\n"
                "Treat gsm_final_answer_check_tool and gsm_unsupported_number_tool outputs as advisory only.\n"
                "If a specialist verifier reports a valid alternative route and no hard contradiction exists, "
                "do not label contradiction-found based on reference mismatch alone.\n"
                "Keep the required output format exactly and do not call tools."
            )
        big_bench_stage2_tools = {
            "bb_arithmetic_chain_tool",
            "bb_variable_state_tool",
            "bb_substitution_tool",
            "bb_decomposition_equivalence_tool",
        }
        if required_tool_names and required_tool_names.issubset(big_bench_stage2_tools | specialist_stage2_tools):
            return (
                "Now provide the final response for the original prompt.\n"
                "For BIG-Bench arithmetic, treat status=hard_contradiction from any bb_* tool "
                "as hard mathematical evidence.\n"
                "If alternative_route_verifier_tool or equivalence_substitution_verifier_tool supports a "
                "different-but-valid route and no hard contradiction exists, avoid contradiction-found.\n"
                "Do not label contradiction-found merely because a step is incomplete, not yet computed, "
                "or less detailed than the reference.\n"
                "Keep the required output format exactly and do not call tools."
            )
        return (
            "Now provide the final response for the original prompt.\n"
            "Use tool outputs as evidence.\n"
            "Treat contradiction_probe_tool and any tool with hard_contradiction=true as hard evidence.\n"
            "Treat alternative_route_verifier_tool support for a valid alternative path as advisory evidence "
            "against contradiction-found when no hard conflict exists.\n"
            "Keep the required output format exactly and do not call tools."
        )

    def _run_chat_completion_with_tools(
        self,
        *,
        client: Any,
        model: str,
        prompt: str,
        tools: list[dict[str, Any]],
        tool_handlers: dict[str, Any],
        required_tool_names: set[str] | None = None,
        max_rounds: int = 8,
    ) -> tuple[str, Any, list[dict[str, Any]], list[str]]:
        required = set(required_tool_names or set())
        if required:
            return self._run_required_tools_then_finalize(
                client=client,
                model=model,
                prompt=prompt,
                tools=tools,
                tool_handlers=tool_handlers,
                required_tool_names=required,
            )

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        tool_trace: list[dict[str, Any]] = []
        tool_errors: list[str] = []
        final_response: Any = None
        final_text = ""
        required = set(required_tool_names or set())
        called_tool_names: set[str] = set()
        config = self._resolve_configuration()
        preserve_reasoning_content = self._needs_siliconflow_reasoning_replay(
            provider=config["provider"],
            model=model,
        )

        for round_index in range(max_rounds):
            response = self._request_chat_completion(
                client=client,
                model=model,
                prompt=prompt,
                messages=messages,
                tools=tools,
                tool_choice="required" if required else "auto",
            )
            final_response = response
            message = self._extract_chat_completion_message(response)
            final_text = self._message_content_to_text(message)
            reasoning_content = self._message_reasoning_content_to_text(message)
            tool_calls = self._message_tool_calls(message)
            if not tool_calls:
                missing = sorted(required - called_tool_names)
                if missing and round_index < max_rounds - 1:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You must call all required tools before finalizing. "
                                f"Missing tools: {', '.join(missing)}. "
                                "Call the missing tools now, then continue."
                            ),
                        }
                    )
                    continue
                if missing:
                    tool_errors.append(f"Required tools were not called: {', '.join(missing)}")
                return final_text, response, tool_trace, tool_errors

            assistant_message = {
                "role": "assistant",
                "content": final_text or "",
                "tool_calls": [],
            }
            self._attach_reasoning_content(
                assistant_message,
                reasoning_content=reasoning_content,
                preserve_reasoning_content=preserve_reasoning_content,
            )
            for tool_call in tool_calls:
                call_id = tool_call["id"]
                tool_name = tool_call["name"]
                raw_arguments = tool_call.get("arguments", "{}")
                assistant_message["tool_calls"].append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": raw_arguments if isinstance(raw_arguments, str) else json.dumps(raw_arguments),
                        },
                    }
                )
            messages.append(assistant_message)

            for tool_call in tool_calls:
                call_id = tool_call["id"]
                tool_name = tool_call["name"]
                raw_arguments = tool_call.get("arguments", "{}")
                if tool_name:
                    called_tool_names.add(tool_name)
                arguments, args_error = self._parse_tool_arguments(raw_arguments)
                error: str | None = args_error

                if error is None:
                    handler = tool_handlers.get(tool_name)
                    if handler is None:
                        error = f"Unknown tool requested by model: {tool_name}"
                        result: dict[str, Any] = {"status": "error", "error": error}
                    else:
                        try:
                            result = handler(arguments)
                        except Exception as exc:  # noqa: BLE001
                            error = f"{tool_name} execution failed: {exc}"
                            result = {"status": "error", "error": error}
                else:
                    result = {"status": "error", "error": error}

                if error:
                    tool_errors.append(error)

                tool_trace.append(
                    {
                        "round": round_index + 1,
                        "tool_call_id": call_id,
                        "tool_name": tool_name,
                        "args": arguments,
                        "result": result,
                        "error": error,
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

            missing_after_round = sorted(required - called_tool_names)
            if missing_after_round and round_index < max_rounds - 1:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Required tools are still missing. "
                            f"Please call these tools next: {', '.join(missing_after_round)}."
                        ),
                    }
                )

        if final_response is None:
            raise RuntimeError("Tool-calling loop ended without any model response.")
        missing = sorted(required - called_tool_names)
        if missing:
            tool_errors.append(f"Required tools were not called: {', '.join(missing)}")
        tool_errors.append("Tool-calling loop reached max rounds before completion.")
        return final_text, final_response, tool_trace, tool_errors

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        metadata: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_handlers: dict[str, Any] | None = None,
        required_tool_names: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        client = self._get_client()
        config = self._resolve_configuration()
        api_style = config["api_style"]
        use_tools = bool(tools)
        handlers = tool_handlers or {}
        required = set(required_tool_names or [])
        used_gateway_sanitized_prompt = False
        try:
            response, last_style, text, tool_trace, tool_errors = self._request_with_retries(
                client=client,
                model=model,
                prompt=prompt,
                api_style=api_style,
                use_tools=use_tools,
                tools=tools,
                handlers=handlers,
                required=required,
            )
        except RetryError as exc:
            last_attempt = getattr(exc, "last_attempt", None)
            last_exception = last_attempt.exception() if last_attempt is not None else exc
            if isinstance(last_exception, Exception) and self._looks_like_gateway_parameter_error(last_exception):
                sanitized_prompt = self._sanitize_prompt_for_gateway(prompt)
                if sanitized_prompt != prompt:
                    used_gateway_sanitized_prompt = True
                    try:
                        response, last_style, text, tool_trace, tool_errors = self._request_with_retries(
                            client=client,
                            model=model,
                            prompt=sanitized_prompt,
                            api_style=api_style,
                            use_tools=use_tools,
                            tools=tools,
                            handlers=handlers,
                            required=required,
                        )
                    except RetryError as retry_exc:
                        raise RuntimeError("OpenAI request failed after retries.") from retry_exc
                else:
                    raise RuntimeError("OpenAI request failed after retries.") from exc
            else:
                raise RuntimeError("OpenAI request failed after retries.") from exc

        meta = {
            "response_id": getattr(response, "id", None),
            "model": getattr(response, "model", model),
            "metadata": metadata or {},
            "api_style": last_style,
            "provider": config["provider"],
            "gateway_prompt_sanitized": used_gateway_sanitized_prompt,
        }
        if use_tools:
            meta["tool_trace"] = tool_trace
            meta["tool_errors"] = tool_errors
            meta["tools_enabled"] = True
            meta["required_tool_names"] = sorted(required)
        usage = getattr(response, "usage", None)
        if usage is not None:
            if hasattr(usage, "model_dump"):
                meta["usage"] = usage.model_dump()
            elif isinstance(usage, dict):
                meta["usage"] = usage
        return text, meta
