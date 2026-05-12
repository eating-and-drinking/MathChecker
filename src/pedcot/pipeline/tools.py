from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable

STAGE1_TOOLS_MODES = {"none", "python", "logic", "both"}
STAGE2_TOOLS_MODES = {"none", "triad"}
BIG_BENCH_DATASET_NAME = "big-bench-mistake"
PRM800K_DATASET_NAME = "prm800k"
MR_GSM8K_ORIGINAL_DATASET_NAME = "mr-gsm8k-original"

ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(slots=True, frozen=True)
class Stage1Tooling:
    schemas: list[dict[str, Any]]
    handlers: dict[str, ToolHandler]
    required_tool_names: set[str]
    route_required_tool_names: Callable[[str, list[str]], set[str]] | None = field(default=None)


_PYTHON_CALC_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "python_calc_tool",
        "description": (
            "Use safe arithmetic/symbolic-style local verification for sub-expressions. "
            "This tool must NOT perform natural-language logic checking. "
            "Do not use this tool to directly solve the entire question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expressions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "A list of local arithmetic expressions to verify.",
                }
            },
            "required": ["expressions"],
            "additionalProperties": False,
        },
    },
}

_PRM_CONSTRAINT_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "prm_constraint_tool",
        "description": (
            "Use robust local constraint verification for PRM-style symbolic steps. "
            "Handles inequalities, integer-range constraints, and symbolic/local numeric checks. "
            "Must NOT perform natural-language logic checking."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expressions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional local expressions/constraints to verify.",
                },
                "question": {
                    "type": "string",
                    "description": "Optional question text used to infer integer interval constraints.",
                },
                "previous_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional prior steps used to infer constraints.",
                },
            },
            "additionalProperties": False,
        },
    },
}

_SYMBOLIC_RELATION_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "symbolic_relation_tool",
        "description": (
            "Verify symbolic relations/equivalences for local algebraic expressions. "
            "Does not do natural-language logic judgement."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expressions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional expression list for symbolic relation checks.",
                },
                "question": {"type": "string"},
                "previous_steps": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
    },
}

_DOMAIN_GUARD_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "domain_guard_tool",
        "description": (
            "Check local domain constraints for log/sqrt/division expressions and produce auditable domain evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "previous_steps": {"type": "array", "items": {"type": "string"}},
                "expressions": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
    },
}

_UNIT_RATIO_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "unit_ratio_tool",
        "description": (
            "Extract and verify local unit-conversion ratios (length/time/money/percent) as auditable conversion hints."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "previous_steps": {"type": "array", "items": {"type": "string"}},
                "expressions": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
    },
}

_LOGIC_CHECK_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "logic_check_tool",
        "description": (
            "Check local step-level reasoning consistency and identify the next unresolved "
            "focus, without solving the full question. This tool owns logic verification."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "previous_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "All steps before the current step.",
                },
                "candidate_focus": {
                    "type": "string",
                    "description": "Optional planning intent for the immediate next subtask.",
                },
            },
            "required": ["question", "previous_steps"],
            "additionalProperties": False,
        },
    },
}

_GSM_EXPR_REFERENCE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "gsm_expr_reference_tool",
        "description": (
            "Verify GSM8K-style calculation tags from previous steps and summarize "
            "auditable numeric state for Stage1 section 3. Do not solve the full problem."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "previous_steps": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question", "previous_steps"],
            "additionalProperties": False,
        },
    },
}


_ANSWER_OBLIGATION_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "answer_obligation_tool",
        "description": (
            "Check whether the current step satisfies the task obligation to provide "
            "an answer or actionable conclusion. Return auditable evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "current_step": {"type": "string"},
                "previous_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional prior steps for local context.",
                },
            },
            "required": ["question", "current_step"],
            "additionalProperties": False,
        },
    },
}

_CONDITION_BINDING_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "condition_binding_tool",
        "description": (
            "Check whether branch conditions and variable bindings in the current "
            "step are consistent with the question definitions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "current_step": {"type": "string"},
                "previous_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["question", "current_step"],
            "additionalProperties": False,
        },
    },
}

_EQUIVALENCE_CHECK_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "equivalence_check_tool",
        "description": (
            "Check whether the current step is plausibly aligned or an alternative valid path. "
            "This is advisory evidence only and must not directly decide contradiction."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "current_step": {"type": "string"},
                "previous_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["question", "current_step"],
            "additionalProperties": False,
        },
    },
}


_CONTRADICTION_PROBE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "contradiction_probe_tool",
        "description": (
            "Probe for hard, auditable contradictions in the current step using only local "
            "evidence from the question, previous steps, and current step. Return evidence; "
            "do not overwrite the final label."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "current_step": {"type": "string"},
                "previous_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["question", "current_step"],
            "additionalProperties": False,
        },
    },
}

_ALTERNATIVE_ROUTE_VERIFIER_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "alternative_route_verifier_tool",
        "description": (
            "Check whether the current step is a mathematically compatible alternative solution route. "
            "Do not treat a different-but-valid path as contradiction without hard evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "current_step": {"type": "string"},
                "previous_steps": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question", "current_step"],
            "additionalProperties": False,
        },
    },
}

_EQUIVALENCE_SUBSTITUTION_VERIFIER_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "equivalence_substitution_verifier_tool",
        "description": (
            "Check whether rewrites, substitutions, and decompositions in the current step are "
            "equivalent or provably contradictory. Return auditable evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "current_step": {"type": "string"},
                "previous_steps": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question", "current_step"],
            "additionalProperties": False,
        },
    },
}

_CONDITION_OBLIGATION_VERIFIER_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "condition_obligation_verifier_tool",
        "description": (
            "Check whether the current step respects question conditions and satisfies any local "
            "obligation to conclude or branch correctly. Return hard conflict evidence only when auditable."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "current_step": {"type": "string"},
                "previous_steps": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question", "current_step"],
            "additionalProperties": False,
        },
    },
}

_GSM_EXPR_CHECK_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "gsm_expr_check_tool",
        "description": (
            "Check GSM8K-style local calculations in the current step, especially "
            "<<expression=result>> tags and explicit arithmetic equalities. Return evidence only."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "current_step": {"type": "string"},
                "previous_steps": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question", "current_step"],
            "additionalProperties": False,
        },
    },
}

_GSM_FINAL_ANSWER_CHECK_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "gsm_final_answer_check_tool",
        "description": (
            "When the current step states a final answer, compare it with the last locally "
            "verified calculation result from the trace. Advisory evidence only; do not use "
            "this tool alone to decide contradiction."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "current_step": {"type": "string"},
                "previous_steps": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question", "current_step"],
            "additionalProperties": False,
        },
    },
}

_GSM_UNSUPPORTED_NUMBER_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "gsm_unsupported_number_tool",
        "description": (
            "Conservatively check whether the current step introduces numeric claims not "
            "grounded in the question, previous steps, or local calculation tags. Advisory evidence only; "
            "do not use this tool alone to decide contradiction."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "current_step": {"type": "string"},
                "previous_steps": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question", "current_step"],
            "additionalProperties": False,
        },
    },
}

_BB_ARITHMETIC_CHAIN_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bb_arithmetic_chain_tool",
        "description": (
            "For BIG-Bench arithmetic steps, verify adjacent equality-chain transformations "
            "with safe local arithmetic evaluation. Returns hard_contradiction only for "
            "provably unequal adjacent expressions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "current_step": {"type": "string"},
                "previous_steps": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question", "current_step"],
            "additionalProperties": False,
        },
    },
}

_BB_VARIABLE_STATE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bb_variable_state_tool",
        "description": (
            "For BIG-Bench arithmetic traces, build A/B/C/D numeric state from previous "
            "calculation steps and check whether the current step conflicts with that state."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "current_step": {"type": "string"},
                "previous_steps": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question", "current_step"],
            "additionalProperties": False,
        },
    },
}

_BB_SUBSTITUTION_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bb_substitution_tool",
        "description": (
            "For BIG-Bench final-combination steps, verify variable substitution and "
            "final arithmetic against prior computed variable values."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "current_step": {"type": "string"},
                "previous_steps": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question", "current_step"],
            "additionalProperties": False,
        },
    },
}

_BB_DECOMPOSITION_EQUIVALENCE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bb_decomposition_equivalence_tool",
        "description": (
            "For BIG-Bench decomposition steps, check whether the claimed A/B/C/D "
            "decomposition is numerically equivalent to the original expression. "
            "Different granularity is allowed; provable non-equivalence is hard evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "current_step": {"type": "string"},
                "previous_steps": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question", "current_step"],
            "additionalProperties": False,
        },
    },
}


def build_stage1_tooling(mode: str, dataset: str | None = None) -> Stage1Tooling:
    normalized = mode.strip().lower()
    if normalized not in STAGE1_TOOLS_MODES:
        raise ValueError(f"Unsupported stage1 tools mode: {mode}")

    if normalized == "none":
        return Stage1Tooling(schemas=[], handlers={}, required_tool_names=set())

    schemas: list[dict[str, Any]] = []
    handlers: dict[str, ToolHandler] = {}
    required: set[str] = set()
    dataset_name = (dataset or "").strip().lower()
    use_prm_constraint_tool = dataset_name == PRM800K_DATASET_NAME
    use_gsm_tool = dataset_name == MR_GSM8K_ORIGINAL_DATASET_NAME

    if normalized in {"python", "both"}:
        if use_gsm_tool:
            schemas.append(_GSM_EXPR_REFERENCE_TOOL_SCHEMA)
            handlers["gsm_expr_reference_tool"] = gsm_expr_reference_tool
            required.add("gsm_expr_reference_tool")
        elif use_prm_constraint_tool:
            schemas.extend(
                [
                    _DOMAIN_GUARD_TOOL_SCHEMA,
                    _SYMBOLIC_RELATION_TOOL_SCHEMA,
                    _PRM_CONSTRAINT_TOOL_SCHEMA,
                    _UNIT_RATIO_TOOL_SCHEMA,
                ]
            )
            handlers["domain_guard_tool"] = domain_guard_tool
            handlers["symbolic_relation_tool"] = symbolic_relation_tool
            handlers["prm_constraint_tool"] = prm_constraint_tool
            handlers["unit_ratio_tool"] = unit_ratio_tool
            required.add("prm_constraint_tool")
        else:
            schemas.append(_PYTHON_CALC_TOOL_SCHEMA)
            handlers["python_calc_tool"] = python_calc_tool
            required.add("python_calc_tool")

    if normalized in {"logic", "both"}:
        schemas.append(_LOGIC_CHECK_TOOL_SCHEMA)
        handlers["logic_check_tool"] = logic_check_tool
        required.add("logic_check_tool")

    route = None
    if use_prm_constraint_tool and normalized in {"python", "both"}:
        route = _build_prm_stage1_router(include_logic=(normalized in {"logic", "both"}))
    return Stage1Tooling(
        schemas=schemas,
        handlers=handlers,
        required_tool_names=required,
        route_required_tool_names=route,
    )


def build_stage2_tooling(mode: str, dataset: str | None = None) -> Stage1Tooling:
    normalized = mode.strip().lower()
    if normalized not in STAGE2_TOOLS_MODES:
        raise ValueError(f"Unsupported stage2 tools mode: {mode}")

    if normalized == "none":
        return Stage1Tooling(schemas=[], handlers={}, required_tool_names=set())

    dataset_name = (dataset or "").strip().lower()
    specialist_schemas = [
        _ALTERNATIVE_ROUTE_VERIFIER_TOOL_SCHEMA,
        _EQUIVALENCE_SUBSTITUTION_VERIFIER_TOOL_SCHEMA,
        _CONDITION_OBLIGATION_VERIFIER_TOOL_SCHEMA,
    ]
    specialist_handlers: dict[str, ToolHandler] = {
        "alternative_route_verifier_tool": alternative_route_verifier_tool,
        "equivalence_substitution_verifier_tool": equivalence_substitution_verifier_tool,
        "condition_obligation_verifier_tool": condition_obligation_verifier_tool,
    }
    if dataset_name == BIG_BENCH_DATASET_NAME:
        schemas = [
            _BB_ARITHMETIC_CHAIN_TOOL_SCHEMA,
            _BB_VARIABLE_STATE_TOOL_SCHEMA,
            _BB_SUBSTITUTION_TOOL_SCHEMA,
            _BB_DECOMPOSITION_EQUIVALENCE_TOOL_SCHEMA,
            *specialist_schemas,
        ]
        handlers: dict[str, ToolHandler] = {
            "bb_arithmetic_chain_tool": bb_arithmetic_chain_tool,
            "bb_variable_state_tool": bb_variable_state_tool,
            "bb_substitution_tool": bb_substitution_tool,
            "bb_decomposition_equivalence_tool": bb_decomposition_equivalence_tool,
            **specialist_handlers,
        }
        required = {
            "bb_arithmetic_chain_tool",
            "bb_variable_state_tool",
            "bb_substitution_tool",
            "bb_decomposition_equivalence_tool",
            "alternative_route_verifier_tool",
            "equivalence_substitution_verifier_tool",
            "condition_obligation_verifier_tool",
        }
        return Stage1Tooling(
            schemas=schemas,
            handlers=handlers,
            required_tool_names=required,
            route_required_tool_names=_build_big_bench_stage2_router(),
        )

    if dataset_name == MR_GSM8K_ORIGINAL_DATASET_NAME:
        schemas = [
            _GSM_EXPR_CHECK_TOOL_SCHEMA,
            _GSM_FINAL_ANSWER_CHECK_TOOL_SCHEMA,
            _GSM_UNSUPPORTED_NUMBER_TOOL_SCHEMA,
            *specialist_schemas,
        ]
        handlers: dict[str, ToolHandler] = {
            "gsm_expr_check_tool": gsm_expr_check_tool,
            "gsm_final_answer_check_tool": gsm_final_answer_check_tool,
            "gsm_unsupported_number_tool": gsm_unsupported_number_tool,
            **specialist_handlers,
        }
        required = {
            "gsm_expr_check_tool",
            "gsm_final_answer_check_tool",
            "gsm_unsupported_number_tool",
            "alternative_route_verifier_tool",
            "equivalence_substitution_verifier_tool",
            "condition_obligation_verifier_tool",
        }
        return Stage1Tooling(
            schemas=schemas,
            handlers=handlers,
            required_tool_names=required,
            route_required_tool_names=_build_mr_stage2_router(),
        )

    schemas = [
        _CONTRADICTION_PROBE_TOOL_SCHEMA,
        *specialist_schemas,
    ]
    handlers: dict[str, ToolHandler] = {
        "contradiction_probe_tool": contradiction_probe_tool,
        **specialist_handlers,
    }
    required = {
        "contradiction_probe_tool",
        "alternative_route_verifier_tool",
        "equivalence_substitution_verifier_tool",
        "condition_obligation_verifier_tool",
    }
    return Stage1Tooling(
        schemas=schemas,
        handlers=handlers,
        required_tool_names=required,
        route_required_tool_names=_build_generic_stage2_router(),
    )


def _build_big_bench_stage2_router() -> Callable[[str, list[str]], set[str]]:
    def route(question: str, context_steps: list[str]) -> set[str]:
        del question
        current_step = context_steps[-1] if context_steps else ""
        if _bb_is_decomposition_step(current_step):
            required = {"bb_decomposition_equivalence_tool"}
            return _augment_stage2_route_with_specialists(required, current_step)
        if _bb_is_final_step(current_step):
            required = {
                "bb_arithmetic_chain_tool",
                "bb_variable_state_tool",
                "bb_substitution_tool",
            }
            return _augment_stage2_route_with_specialists(required, current_step)
        if _bb_is_calculation_step(current_step):
            required = {
                "bb_arithmetic_chain_tool",
                "bb_variable_state_tool",
            }
            return _augment_stage2_route_with_specialists(required, current_step)
        return _augment_stage2_route_with_specialists({"bb_arithmetic_chain_tool"}, current_step)

    return route


def _build_mr_stage2_router() -> Callable[[str, list[str]], set[str]]:
    def route(question: str, context_steps: list[str]) -> set[str]:
        del question
        current_step = context_steps[-1] if context_steps else ""
        required = {
            "gsm_expr_check_tool",
            "gsm_final_answer_check_tool",
            "gsm_unsupported_number_tool",
        }
        return _augment_stage2_route_with_specialists(required, current_step)

    return route


def _build_generic_stage2_router() -> Callable[[str, list[str]], set[str]]:
    def route(question: str, context_steps: list[str]) -> set[str]:
        del question
        current_step = context_steps[-1] if context_steps else ""
        return _augment_stage2_route_with_specialists({"contradiction_probe_tool"}, current_step)

    return route


def _augment_stage2_route_with_specialists(required: set[str], current_step: str) -> set[str]:
    lowered = current_step.lower()
    augmented = set(required)
    augmented.add("alternative_route_verifier_tool")

    if _generic_is_transformation_step(lowered):
        augmented.add("equivalence_substitution_verifier_tool")

    if _generic_is_condition_or_conclusion_step(lowered):
        augmented.add("condition_obligation_verifier_tool")

    if not {"equivalence_substitution_verifier_tool", "condition_obligation_verifier_tool"} & augmented:
        augmented.add("equivalence_substitution_verifier_tool")

    return augmented


def _generic_is_transformation_step(lowered_step: str) -> bool:
    return any(
        marker in lowered_step
        for marker in [
            "equivalently",
            "equivalent",
            "rewrite",
            "rewritten",
            "can be written as",
            "substitute",
            "substituting",
            "factor",
            "expand",
            "let ",
            "set ",
            "denote",
        ]
    )


def _generic_is_condition_or_conclusion_step(lowered_step: str) -> bool:
    return any(
        marker in lowered_step
        for marker in [
            "if ",
            "when ",
            "case ",
            "cases",
            "assume",
            "suppose",
            "therefore",
            "hence",
            "thus",
            "so the answer is",
            "final answer",
            "final equation",
        ]
    )


def _bb_context(args: dict[str, Any]) -> tuple[str, str, list[str]]:
    question = str(args.get("question", "")).strip()
    current_step = str(args.get("current_step", "")).strip()
    previous_steps = _coerce_previous_steps(args.get("previous_steps"))
    return question, current_step, previous_steps


def _bb_is_decomposition_step(step: str) -> bool:
    return "this equation can be written as" in step.lower()


def _bb_is_calculation_step(step: str) -> bool:
    return "let's calculate" in step.lower() or "lets calculate" in step.lower()


def _bb_is_final_step(step: str) -> bool:
    lowered = step.lower()
    return "final equation" in lowered or "so the answer is" in lowered


def _bb_number(value: Fraction) -> int | float:
    if value.denominator == 1:
        return int(value)
    return float(value)


def _bb_eval_expression(expression: str, variables: dict[str, Fraction] | None = None) -> Fraction:
    normalized = expression.strip().replace("−", "-")
    normalized = normalized.replace("^", "**")
    if not re.fullmatch(r"[A-Z0-9\+\-\*/\(\)\.\s]+", normalized):
        raise ValueError(f"Unsupported expression characters: {expression}")
    parsed = ast.parse(normalized, mode="eval")
    return _bb_eval_ast(parsed.body, variables or {})


def _bb_eval_ast(node: ast.AST, variables: dict[str, Fraction]) -> Fraction:
    if isinstance(node, ast.BinOp):
        left = _bb_eval_ast(node.left, variables)
        right = _bb_eval_ast(node.right, variables)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ValueError("Division by zero.")
            return left / right
        raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
    if isinstance(node, ast.UnaryOp):
        operand = _bb_eval_ast(node.operand, variables)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise ValueError("Boolean constants are not arithmetic values.")
        if isinstance(node.value, int):
            return Fraction(node.value, 1)
        if isinstance(node.value, float):
            return Fraction(str(node.value))
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        raise ValueError(f"Unknown variable: {node.id}")
    raise ValueError(f"Unsupported syntax: {type(node).__name__}")


def _bb_trim_expression(text: str) -> str | None:
    cleaned = text.strip().strip("`").strip().rstrip(".,;:")
    cleaned = re.sub(r"^\s*[-*]\s+", "", cleaned)
    start = re.search(r"[A-Z0-9\(\+\-]", cleaned)
    if not start:
        return None
    cleaned = cleaned[start.start() :].strip()
    match = re.match(r"([A-Z0-9\+\-\*/\(\)\.\s]+)", cleaned)
    if not match:
        return None
    expression = match.group(1).strip().rstrip(".,;:")
    if not expression or not re.search(r"[A-Z0-9]", expression):
        return None
    return expression


def _bb_extract_equality_chain(step: str) -> list[str]:
    if _bb_is_decomposition_step(step):
        return []
    text = step.strip().replace('"', "")
    text = re.sub(r"^.*?Let'?s calculate\s+", "", text, flags=re.I)
    text = re.sub(r"^.*?Then,\s*the final equation is\s+", "", text, flags=re.I)
    text = re.sub(r"\.\s*So the answer is.*$", "", text, flags=re.I | re.S)
    if "=" not in text:
        return []
    parts = [part.strip() for part in text.split("=")]
    if len(parts) >= 2 and re.fullmatch(r"[A-Z]", parts[0]):
        parts = parts[1:]
    chain: list[str] = []
    for part in parts:
        expression = _bb_trim_expression(part)
        if expression is not None:
            chain.append(expression)
    return chain


def _bb_chain_checks(step: str, variables: dict[str, Fraction] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    chain = _bb_extract_equality_chain(step)
    evaluated: list[tuple[str, Fraction]] = []
    not_verifiable: list[dict[str, Any]] = []
    variables = variables or {}
    for expression in chain:
        try:
            evaluated.append((expression, _bb_eval_expression(expression, variables)))
        except Exception as exc:  # noqa: BLE001
            not_verifiable.append({"expression": expression, "reason": str(exc)})

    checks: list[dict[str, Any]] = []
    contradictions: list[dict[str, Any]] = []
    for (left_expr, left_value), (right_expr, right_value) in zip(evaluated, evaluated[1:]):
        row = {
            "left_expression": left_expr,
            "right_expression": right_expr,
            "left_value": _bb_number(left_value),
            "right_value": _bb_number(right_value),
            "equivalent": left_value == right_value,
        }
        checks.append(row)
        if left_value != right_value:
            contradictions.append(
                {
                    **row,
                    "conflict": (
                        f"Adjacent equality-chain expressions are not equal: "
                        f"{left_expr} = {_bb_number(left_value)} but "
                        f"{right_expr} = {_bb_number(right_value)}."
                    ),
                }
            )
    return checks, contradictions, not_verifiable


def _bb_assignment_variable(step: str) -> str | None:
    match = re.search(r"Let'?s calculate\s+([A-Z])\s*=", step, flags=re.I)
    return match.group(1).upper() if match else None


def _bb_variable_state(previous_steps: list[str]) -> tuple[dict[str, Fraction], list[dict[str, Any]]]:
    variables: dict[str, Fraction] = {}
    evidence: list[dict[str, Any]] = []
    for index, step in enumerate(previous_steps):
        variable = _bb_assignment_variable(step)
        if variable is None:
            continue
        chain = _bb_extract_equality_chain(step)
        if not chain:
            continue
        try:
            value = _bb_eval_expression(chain[-1], variables)
        except Exception as exc:  # noqa: BLE001
            evidence.append(
                {
                    "step_index": index,
                    "variable": variable,
                    "status": "not_verifiable",
                    "reason": str(exc),
                }
            )
            continue
        variables[variable] = value
        evidence.append(
            {
                "step_index": index,
                "variable": variable,
                "claimed_expression": chain[-1],
                "claimed_value": _bb_number(value),
                "status": "recorded",
            }
        )
    return variables, evidence


def _bb_original_expression(question: str) -> str | None:
    raw = str(question).split("=", maxsplit=1)[0].strip()
    return _bb_trim_expression(raw)


def _bb_claimed_decomposition(step: str) -> tuple[str | None, dict[str, str], str | None]:
    claim_match = re.search(r"written as\s+\"([^\"]+)\"", step, flags=re.I)
    claim = claim_match.group(1).strip() if claim_match else None
    where_match = re.search(r"\bwhere\b\s*(.*)$", step, flags=re.I | re.S)
    if not where_match:
        return claim, {}, "No where-clause found."
    definition_text = where_match.group(1).strip().rstrip(".")
    matches = list(re.finditer(r"\b([A-Z])\s*=", definition_text))
    if not matches:
        return claim, {}, "No variable definitions found."
    definitions: dict[str, str] = {}
    for index, match in enumerate(matches):
        variable = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(definition_text)
        expression = definition_text[start:end].strip()
        expression = re.sub(r",?\s*(?:and\s*)?$", "", expression, flags=re.I).strip()
        expression = expression.strip(", ")
        if expression:
            definitions[variable] = expression
    return claim, definitions, None


def bb_arithmetic_chain_tool(args: dict[str, Any]) -> dict[str, Any]:
    _, current_step, previous_steps = _bb_context(args)
    variables, variable_evidence = _bb_variable_state(previous_steps)
    checks, contradictions, not_verifiable = _bb_chain_checks(current_step, variables)
    if contradictions:
        status = "hard_contradiction"
    elif checks:
        status = "verified"
    else:
        status = "handled_not_verifiable"
    return {
        "status": status,
        "verification_type": "big_bench_arithmetic_chain",
        "hard_contradiction": status == "hard_contradiction",
        "chain_checks": checks,
        "contradictions": contradictions,
        "variable_state": {key: _bb_number(value) for key, value in variables.items()},
        "variable_state_evidence": variable_evidence,
        "not_verifiable": not_verifiable if not_verifiable else {},
        "checked_signal_count": len(checks),
        "errors": {},
    }


def bb_variable_state_tool(args: dict[str, Any]) -> dict[str, Any]:
    _, current_step, previous_steps = _bb_context(args)
    variables, variable_evidence = _bb_variable_state(previous_steps)
    variable = _bb_assignment_variable(current_step)
    checks: list[dict[str, Any]] = []
    contradictions: list[dict[str, Any]] = []
    if variable is not None:
        chain = _bb_extract_equality_chain(current_step)
        if chain:
            try:
                claimed_value = _bb_eval_expression(chain[-1], variables)
            except Exception as exc:  # noqa: BLE001
                return {
                    "status": "handled_not_verifiable",
                    "verification_type": "big_bench_variable_state",
                    "hard_contradiction": False,
                    "variable_state": {key: _bb_number(value) for key, value in variables.items()},
                    "variable_state_evidence": variable_evidence,
                    "not_verifiable": {"current_step": str(exc)},
                    "checked_signal_count": 0,
                    "errors": {},
                }
            checks.append(
                {
                    "variable": variable,
                    "claimed_expression": chain[-1],
                    "claimed_value": _bb_number(claimed_value),
                    "previous_value": _bb_number(variables[variable]) if variable in variables else None,
                    "matches_previous": variable not in variables or variables[variable] == claimed_value,
                }
            )
            if variable in variables and variables[variable] != claimed_value:
                contradictions.append(
                    {
                        "variable": variable,
                        "previous_value": _bb_number(variables[variable]),
                        "current_claimed_value": _bb_number(claimed_value),
                        "conflict": f"{variable} conflicts with a prior computed value.",
                    }
                )
    checks.extend(_bb_final_variable_usage_checks(current_step, variables))
    contradictions.extend([item for item in checks if item.get("equivalent") is False])
    if contradictions:
        status = "hard_contradiction"
    elif checks or variable_evidence:
        status = "verified"
    else:
        status = "handled_not_verifiable"
    return {
        "status": status,
        "verification_type": "big_bench_variable_state",
        "hard_contradiction": status == "hard_contradiction",
        "variable_state": {key: _bb_number(value) for key, value in variables.items()},
        "variable_state_evidence": variable_evidence,
        "checks": checks,
        "contradictions": contradictions,
        "not_verifiable": {},
        "checked_signal_count": len(checks),
        "errors": {},
    }


def _bb_final_variable_usage_checks(current_step: str, variables: dict[str, Fraction]) -> list[dict[str, Any]]:
    if not _bb_is_final_step(current_step) or not variables:
        return []
    checks, contradictions, _ = _bb_chain_checks(current_step, variables)
    return [*checks, *contradictions]


def bb_substitution_tool(args: dict[str, Any]) -> dict[str, Any]:
    _, current_step, previous_steps = _bb_context(args)
    if not _bb_is_final_step(current_step):
        return {
            "status": "handled_not_verifiable",
            "verification_type": "big_bench_substitution",
            "hard_contradiction": False,
            "not_verifiable": {"current_step": "Current step is not a final-combination step."},
            "checked_signal_count": 0,
            "errors": {},
        }
    variables, variable_evidence = _bb_variable_state(previous_steps)
    checks, contradictions, not_verifiable = _bb_chain_checks(current_step, variables)
    if contradictions:
        status = "hard_contradiction"
    elif checks:
        status = "verified"
    else:
        status = "handled_not_verifiable"
    return {
        "status": status,
        "verification_type": "big_bench_substitution",
        "hard_contradiction": status == "hard_contradiction",
        "variable_state": {key: _bb_number(value) for key, value in variables.items()},
        "variable_state_evidence": variable_evidence,
        "substitution_checks": checks,
        "contradictions": contradictions,
        "not_verifiable": not_verifiable if not_verifiable else {},
        "checked_signal_count": len(checks),
        "errors": {},
    }


def bb_decomposition_equivalence_tool(args: dict[str, Any]) -> dict[str, Any]:
    question, current_step, _ = _bb_context(args)
    if not _bb_is_decomposition_step(current_step):
        return {
            "status": "handled_not_verifiable",
            "verification_type": "big_bench_decomposition_equivalence",
            "hard_contradiction": False,
            "not_verifiable": {"current_step": "Current step is not a decomposition step."},
            "checked_signal_count": 0,
            "errors": {},
        }
    original_expression = _bb_original_expression(question)
    claimed_structure, definitions, parse_error = _bb_claimed_decomposition(current_step)
    if original_expression is None or claimed_structure is None or parse_error:
        return {
            "status": "handled_not_verifiable",
            "verification_type": "big_bench_decomposition_equivalence",
            "hard_contradiction": False,
            "original_expression": original_expression,
            "claimed_structure": claimed_structure,
            "definitions": definitions,
            "not_verifiable": {"parse": parse_error or "Could not parse original expression or claimed structure."},
            "checked_signal_count": 0,
            "errors": {},
        }
    variable_values: dict[str, Fraction] = {}
    definition_evidence: list[dict[str, Any]] = []
    for variable, expression in definitions.items():
        try:
            value = _bb_eval_expression(expression)
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "handled_not_verifiable",
                "verification_type": "big_bench_decomposition_equivalence",
                "hard_contradiction": False,
                "original_expression": original_expression,
                "claimed_structure": claimed_structure,
                "definitions": definitions,
                "not_verifiable": {variable: str(exc)},
                "checked_signal_count": 0,
                "errors": {},
            }
        variable_values[variable] = value
        definition_evidence.append(
            {
                "variable": variable,
                "expression": expression,
                "value": _bb_number(value),
            }
        )
    try:
        original_value = _bb_eval_expression(original_expression)
        claimed_value = _bb_eval_expression(claimed_structure, variable_values)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "handled_not_verifiable",
            "verification_type": "big_bench_decomposition_equivalence",
            "hard_contradiction": False,
            "original_expression": original_expression,
            "claimed_structure": claimed_structure,
            "definitions": definitions,
            "definition_evidence": definition_evidence,
            "not_verifiable": {"evaluation": str(exc)},
            "checked_signal_count": 0,
            "errors": {},
        }
    equivalent = original_value == claimed_value
    contradiction = None
    if not equivalent:
        contradiction = {
            "original_expression": original_expression,
            "original_value": _bb_number(original_value),
            "claimed_structure": claimed_structure,
            "claimed_value": _bb_number(claimed_value),
            "conflict": "Claimed decomposition is not equivalent to the original expression.",
        }
    return {
        "status": "verified" if equivalent else "hard_contradiction",
        "verification_type": "big_bench_decomposition_equivalence",
        "hard_contradiction": not equivalent,
        "original_expression": original_expression,
        "claimed_structure": claimed_structure,
        "definition_evidence": definition_evidence,
        "original_value": _bb_number(original_value),
        "claimed_value": _bb_number(claimed_value),
        "equivalent": equivalent,
        "contradictions": [contradiction] if contradiction else [],
        "not_verifiable": {},
        "checked_signal_count": 1,
        "errors": {},
    }


def python_calc_tool(args: dict[str, Any]) -> dict[str, Any]:
    expressions = args.get("expressions")
    if not isinstance(expressions, list) or not expressions:
        return {
            "status": "tool_error",
            "verification_type": "numeric_symbolic",
            "numeric_values": {},
            "symbolic_results": {},
            "auditable_conclusions": [],
            "not_verifiable": {},
            "errors": {"expressions": "Expected a non-empty list of expression strings."},
        }

    numeric_values: dict[str, int | float] = {}
    symbolic_results: dict[str, dict[str, Any]] = {}
    auditable_conclusions: list[dict[str, Any]] = []
    not_verifiable: dict[str, str] = {}
    errors: dict[str, str] = {}

    for raw_expression in expressions:
        if not isinstance(raw_expression, str) or not raw_expression.strip():
            errors[str(raw_expression)] = "Expression must be a non-empty string."
            continue
        expression = raw_expression.strip()

        if _is_numeric_expression(expression):
            try:
                numeric_value = evaluate_arithmetic_expression(expression)
                numeric_values[expression] = numeric_value
                auditable_conclusions.append(
                    {
                        "expression": expression,
                        "mode": "numeric",
                        "result_type": "value",
                        "value": numeric_value,
                    }
                )
                continue
            except Exception as exc:  # noqa: BLE001
                not_verifiable[expression] = f"Numeric evaluation unavailable: {exc}"
                auditable_conclusions.append(
                    {
                        "expression": expression,
                        "mode": "numeric",
                        "result_type": "not_verifiable",
                        "reason": str(exc),
                    }
                )
                continue

        symbolic = _safe_symbolic_analyze_expression(expression)
        symbolic_status = symbolic.get("status")
        if symbolic_status == "verified":
            symbolic_results[expression] = symbolic
            auditable_conclusions.append(
                {
                    "expression": expression,
                    "mode": "symbolic",
                    "result_type": symbolic.get("result_type"),
                    "evidence": symbolic.get("evidence"),
                }
            )
            continue

        reason = str(symbolic.get("reason", "Symbolic verification unavailable."))
        not_verifiable[expression] = reason
        auditable_conclusions.append(
            {
                "expression": expression,
                "mode": "symbolic",
                "result_type": "not_verifiable",
                "reason": reason,
            }
        )

    verified_numeric = len(numeric_values)
    verified_symbolic = len(symbolic_results)
    verified_total = verified_numeric + verified_symbolic
    not_verifiable_total = len(not_verifiable)

    if errors:
        status = "tool_error"
    elif verified_total == 0 and not_verifiable_total == 0:
        status = "tool_error"
    elif verified_numeric > 0 and verified_symbolic == 0 and not_verifiable_total == 0:
        status = "numeric_verified"
    elif verified_symbolic > 0 and verified_numeric == 0 and not_verifiable_total == 0:
        status = "symbolic_verified"
    elif verified_total > 0 and not_verifiable_total == 0:
        status = "mixed_verified"
    elif verified_total > 0 and not_verifiable_total > 0:
        status = "partial_verified"
    else:
        status = "handled_not_verifiable"

    return {
        "status": status,
        "verification_type": "numeric_symbolic",
        "numeric_values": numeric_values,
        "symbolic_results": symbolic_results,
        "auditable_conclusions": auditable_conclusions,
        "not_verifiable": not_verifiable,
        "errors": errors,
        "verified_numeric_count": verified_numeric,
        "verified_symbolic_count": verified_symbolic,
        "verified_count": verified_total,
        "not_verifiable_count": len(not_verifiable),
        "error_count": len(errors),
    }


def prm_constraint_tool(args: dict[str, Any]) -> dict[str, Any]:
    raw_expressions = args.get("expressions")
    expressions: list[str] = []
    if isinstance(raw_expressions, list):
        expressions = [str(item).strip() for item in raw_expressions if str(item).strip()]

    question = str(args.get("question", "")).strip()
    previous_steps_raw = args.get("previous_steps")
    previous_steps: list[str] = []
    if isinstance(previous_steps_raw, list):
        previous_steps = [str(item).strip() for item in previous_steps_raw if str(item).strip()]

    if not expressions:
        source = "\n".join([question, *previous_steps]).strip()
        expressions = _extract_constraint_candidates(source)

    numeric_values: dict[str, int | float] = {}
    symbolic_results: dict[str, dict[str, Any]] = {}
    auditable_conclusions: list[dict[str, Any]] = []
    not_verifiable: dict[str, str] = {}
    errors: dict[str, str] = {}

    if not expressions:
        return {
            "status": "handled_not_verifiable",
            "verification_type": "constraint_symbolic",
            "numeric_values": {},
            "symbolic_results": {},
            "auditable_conclusions": [],
            "not_verifiable": {"_global": "No usable expression/constraint candidates were extracted."},
            "errors": {},
            "verified_numeric_count": 0,
            "verified_symbolic_count": 0,
            "verified_count": 0,
            "not_verifiable_count": 1,
            "error_count": 0,
        }

    for expression in expressions:
        if _is_numeric_expression(expression):
            try:
                value = evaluate_arithmetic_expression(expression)
            except Exception as exc:  # noqa: BLE001
                not_verifiable[expression] = f"Numeric evaluation unavailable: {exc}"
                auditable_conclusions.append(
                    {
                        "expression": expression,
                        "mode": "numeric",
                        "result_type": "not_verifiable",
                        "reason": str(exc),
                    }
                )
            else:
                numeric_values[expression] = value
                auditable_conclusions.append(
                    {
                        "expression": expression,
                        "mode": "numeric",
                        "result_type": "value",
                        "value": value,
                    }
                )
            continue

        constraint_result = _analyze_chain_inequality(
            expression=expression,
            question=question,
        )
        if constraint_result is not None:
            symbolic_results[expression] = constraint_result
            auditable_conclusions.append(
                {
                    "expression": expression,
                    "mode": "constraint",
                    "result_type": constraint_result.get("result_type"),
                    "evidence": constraint_result.get("evidence"),
                }
            )
            continue

        symbolic = _safe_symbolic_analyze_expression(expression)
        symbolic_status = symbolic.get("status")
        if symbolic_status == "verified":
            symbolic_results[expression] = symbolic
            auditable_conclusions.append(
                {
                    "expression": expression,
                    "mode": "symbolic",
                    "result_type": symbolic.get("result_type"),
                    "evidence": symbolic.get("evidence"),
                }
            )
            continue

        reason = str(symbolic.get("reason", "Constraint/symbolic verification unavailable."))
        not_verifiable[expression] = reason
        auditable_conclusions.append(
            {
                "expression": expression,
                "mode": "symbolic",
                "result_type": "not_verifiable",
                "reason": reason,
            }
        )

    verified_numeric = len(numeric_values)
    verified_symbolic = len(symbolic_results)
    verified_total = verified_numeric + verified_symbolic
    not_verifiable_total = len(not_verifiable)

    if errors:
        status = "tool_error"
    elif verified_numeric > 0 and verified_symbolic == 0 and not_verifiable_total == 0:
        status = "numeric_verified"
    elif verified_symbolic > 0 and verified_numeric == 0 and not_verifiable_total == 0:
        status = "symbolic_verified"
    elif verified_total > 0 and not_verifiable_total == 0:
        status = "mixed_verified"
    elif verified_total > 0 and not_verifiable_total > 0:
        status = "partial_verified"
    else:
        status = "handled_not_verifiable"

    return {
        "status": status,
        "verification_type": "constraint_symbolic",
        "numeric_values": numeric_values,
        "symbolic_results": symbolic_results,
        "auditable_conclusions": auditable_conclusions,
        "not_verifiable": not_verifiable,
        "errors": errors,
        "verified_numeric_count": verified_numeric,
        "verified_symbolic_count": verified_symbolic,
        "verified_count": verified_total,
        "not_verifiable_count": len(not_verifiable),
        "error_count": len(errors),
    }


_UNIT_KEYWORDS = (
    "inch",
    "inches",
    "foot",
    "feet",
    "minute",
    "minutes",
    "hour",
    "hours",
    "second",
    "seconds",
    "percent",
    "dollar",
    "dollars",
    "cent",
    "cents",
    "%",
)


def _coerce_expression_inputs(args: dict[str, Any]) -> list[str]:
    expressions_raw = args.get("expressions")
    if not isinstance(expressions_raw, list):
        return []
    expressions: list[str] = []
    for item in expressions_raw:
        text = str(item).strip()
        if text:
            expressions.append(text)
    return _unique_preserve_order(expressions)[:24]


def _coerce_previous_steps(previous_steps_raw: Any) -> list[str]:
    if not isinstance(previous_steps_raw, list):
        return []
    steps: list[str] = []
    for item in previous_steps_raw:
        text = str(item).strip()
        if text:
            steps.append(text)
    return steps[:64]


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _extract_candidate_expressions_from_text(text: str, limit: int = 16) -> list[str]:
    source = text.strip()
    if not source:
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    def add(raw_item: str) -> None:
        item = raw_item.strip().rstrip(".,;:")
        if not item:
            return
        if len(item) > 240:
            return
        if item in seen:
            return
        seen.add(item)
        candidates.append(item)

    for item in _extract_constraint_candidates(source):
        add(item)
        if len(candidates) >= limit:
            return candidates[:limit]

    for match in re.finditer(
        r"([A-Za-z0-9\\\(\)\^\+\-\*/\.]+(?:<=|>=|=|<|>)[A-Za-z0-9\\\(\)\^\+\-\*/\.]+)",
        source,
    ):
        add(match.group(1))
        if len(candidates) >= limit:
            return candidates[:limit]

    for match in re.finditer(
        r"(?<![A-Za-z\\])(-?\d+(?:\.\d+)?(?:\s*[\+\-\*/]\s*-?\d+(?:\.\d+)?)+)(?![A-Za-z])",
        source,
    ):
        add(match.group(1))
        if len(candidates) >= limit:
            return candidates[:limit]

    return candidates[:limit]


def _contains_domain_signal(text: str) -> bool:
    lowered = text.lower()
    if any(token in lowered for token in ("\\sqrt", "sqrt(", "\\log", "\\ln", "log(", "ln(")):
        return True
    if re.search(r"/\s*[A-Za-z0-9_\(]", text):
        return True
    return False


def _contains_symbolic_signal(text: str) -> bool:
    if re.search(r"(<=|>=|=|<|>)", text):
        return True
    if re.search(r"[A-Za-z]", text) and any(op in text for op in ("+", "-", "*", "/", "^")):
        return True
    if "\\frac" in text or "\\sqrt" in text:
        return True
    return False


def _contains_unit_signal(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in _UNIT_KEYWORDS)


def _build_prm_stage1_router(*, include_logic: bool) -> Callable[[str, list[str]], set[str]]:
    def route(question: str, previous_steps: list[str]) -> set[str]:
        safe_question = str(question).strip()
        safe_previous_steps = [str(step).strip() for step in previous_steps if str(step).strip()]
        source = "\n".join([safe_question, *safe_previous_steps]).strip()
        required_names: set[str] = {"prm_constraint_tool"}
        if include_logic:
            required_names.add("logic_check_tool")

        if not source:
            return required_names

        if _contains_domain_signal(source):
            required_names.add("domain_guard_tool")
            # Domain checks should happen before symbolic relation checks.
            required_names.add("symbolic_relation_tool")
        elif _contains_symbolic_signal(source):
            required_names.add("symbolic_relation_tool")

        if _contains_unit_signal(source):
            required_names.add("unit_ratio_tool")

        return required_names

    return route


def symbolic_relation_tool(args: dict[str, Any]) -> dict[str, Any]:
    expressions = _coerce_expression_inputs(args)
    question = str(args.get("question", "")).strip()
    previous_steps = _coerce_previous_steps(args.get("previous_steps"))
    if not expressions:
        expressions = _extract_candidate_expressions_from_text("\n".join([question, *previous_steps]))

    symbolic_results: dict[str, dict[str, Any]] = {}
    auditable_conclusions: list[dict[str, Any]] = []
    not_verifiable: dict[str, str] = {}

    if not expressions:
        return {
            "status": "handled_not_verifiable",
            "verification_type": "symbolic_relation",
            "symbolic_results": {},
            "auditable_conclusions": [],
            "not_verifiable": {"_global": "No symbolic candidates extracted."},
            "errors": {},
            "verified_count": 0,
            "not_verifiable_count": 1,
            "error_count": 0,
        }

    for expression in expressions:
        analyzed = _safe_symbolic_analyze_expression(expression)
        if analyzed.get("status") == "verified":
            symbolic_results[expression] = analyzed
            auditable_conclusions.append(
                {
                    "expression": expression,
                    "mode": "symbolic",
                    "result_type": analyzed.get("result_type"),
                    "evidence": analyzed.get("evidence"),
                }
            )
        else:
            reason = str(analyzed.get("reason", "Symbolic verification unavailable."))
            not_verifiable[expression] = reason
            auditable_conclusions.append(
                {
                    "expression": expression,
                    "mode": "symbolic",
                    "result_type": "not_verifiable",
                    "reason": reason,
                }
            )

    verified_count = len(symbolic_results)
    not_verifiable_count = len(not_verifiable)
    if verified_count > 0 and not_verifiable_count == 0:
        status = "symbolic_verified"
    elif verified_count > 0:
        status = "partial_verified"
    else:
        status = "handled_not_verifiable"
    return {
        "status": status,
        "verification_type": "symbolic_relation",
        "symbolic_results": symbolic_results,
        "auditable_conclusions": auditable_conclusions,
        "not_verifiable": not_verifiable,
        "errors": {},
        "verified_count": verified_count,
        "not_verifiable_count": not_verifiable_count,
        "error_count": 0,
    }


def domain_guard_tool(args: dict[str, Any]) -> dict[str, Any]:
    question = str(args.get("question", "")).strip()
    previous_steps = _coerce_previous_steps(args.get("previous_steps"))
    expressions = _coerce_expression_inputs(args)
    source = "\n".join([question, *previous_steps, *expressions]).strip()

    domain_constraints: list[str] = []
    conflicts: list[str] = []
    evidence: list[str] = []

    for log_arg in re.findall(r"(?:\\log|\\ln|log|ln)\s*\(([^)]+)\)", source):
        cleaned = log_arg.strip()
        if cleaned:
            domain_constraints.append(f"{cleaned} > 0")
            evidence.append(f"log/ln domain requires {cleaned} > 0")

    for sqrt_arg in re.findall(r"(?:\\sqrt|sqrt)\s*\(([^)]+)\)", source):
        cleaned = sqrt_arg.strip()
        if cleaned:
            domain_constraints.append(f"{cleaned} >= 0")
            evidence.append(f"sqrt domain requires {cleaned} >= 0")

    for match in re.finditer(r"/\s*([A-Za-z0-9_]+)", source):
        denominator = match.group(1).strip()
        if denominator:
            domain_constraints.append(f"{denominator} != 0")
            evidence.append(f"division requires {denominator} != 0")

    if re.search(r"(?:\\log|\\ln|log|ln)\s*\(\s*0\s*\)", source):
        conflicts.append("log/ln argument is 0, domain violated.")
    if re.search(r"(?:\\sqrt|sqrt)\s*\(\s*-\s*\d", source):
        conflicts.append("sqrt argument appears negative, domain violated.")
    if re.search(r"/\s*0(?:[^0-9]|$)", source):
        conflicts.append("Division by zero pattern detected.")

    domain_constraints = _unique_preserve_order(domain_constraints)
    evidence = _unique_preserve_order(evidence)
    status = "domain_conflict" if conflicts else "domain_checked"
    return {
        "status": status,
        "verification_type": "domain",
        "domain_constraints": domain_constraints,
        "conflicts": conflicts,
        "evidence": evidence,
        "checked_signal_count": len(domain_constraints),
    }


def unit_ratio_tool(args: dict[str, Any]) -> dict[str, Any]:
    question = str(args.get("question", "")).strip()
    previous_steps = _coerce_previous_steps(args.get("previous_steps"))
    expressions = _coerce_expression_inputs(args)
    source = "\n".join([question, *previous_steps, *expressions]).lower()

    hints: list[str] = []
    if "foot" in source or "feet" in source:
        hints.append("1 foot = 12 inches")
    if "inch" in source or "inches" in source:
        hints.append("12 inches = 1 foot")
    if "minute" in source or "minutes" in source:
        hints.append("1 minute = 60 seconds")
    if "hour" in source or "hours" in source:
        hints.append("1 hour = 60 minutes = 3600 seconds")
    if "percent" in source or "%" in source:
        hints.append("percent means divide by 100")
    if "dollar" in source or "dollars" in source:
        hints.append("1 dollar = 100 cents")

    hints = _unique_preserve_order(hints)
    status = "unit_checked" if hints else "handled_not_verifiable"
    return {
        "status": status,
        "verification_type": "unit_ratio",
        "conversion_hints": hints,
        "evidence": hints,
        "not_verifiable": {} if hints else {"_global": "No unit-conversion signals detected."},
    }


def gsm_expr_reference_tool(args: dict[str, Any]) -> dict[str, Any]:
    question = str(args.get("question", "")).strip()
    previous_steps = _coerce_previous_steps(args.get("previous_steps"))
    source = "\n".join(previous_steps)
    checks = _extract_gsm_calculation_checks(source)
    incorrect = [item for item in checks if item.get("verdict") == "arithmetic_error"]
    verified = [item for item in checks if item.get("verdict") == "verified"]
    not_verifiable = [item for item in checks if item.get("verdict") == "not_verifiable"]

    status = "gsm_reference_ready"
    if incorrect:
        status = "gsm_reference_conflict"
    elif not checks:
        status = "handled_not_verifiable"

    return {
        "status": status,
        "verification_type": "gsm_expression_reference",
        "question_numbers": _extract_numeric_strings(question),
        "previous_calculation_checks": checks,
        "verified_previous_count": len(verified),
        "incorrect_previous_count": len(incorrect),
        "not_verifiable_count": len(not_verifiable),
        "last_verified_value": verified[-1].get("computed_result") if verified else None,
        "auditable_conclusions": [
            "Previous GSM calculation tags/equalities were checked locally."
            if checks
            else "No previous GSM calculation tag/equality was available for local checking."
        ],
        "checked_signal_count": len(checks),
        "errors": {},
    }


def gsm_expr_check_tool(args: dict[str, Any]) -> dict[str, Any]:
    current_step = str(args.get("current_step", "")).strip()
    if not current_step:
        return {
            "status": "tool_error",
            "verification_type": "gsm_expression_check",
            "errors": {"current_step": "Expected a non-empty current_step string."},
        }

    checks = _extract_gsm_calculation_checks(current_step)
    incorrect = [item for item in checks if item.get("verdict") == "arithmetic_error"]
    verified = [item for item in checks if item.get("verdict") == "verified"]
    not_verifiable = [item for item in checks if item.get("verdict") == "not_verifiable"]
    if incorrect:
        status = "gsm_calculation_contradiction"
    elif verified and not not_verifiable:
        status = "gsm_calculation_verified"
    elif verified:
        status = "partial_verified"
    elif not_verifiable:
        status = "handled_not_verifiable"
    else:
        status = "handled_not_verifiable"

    return {
        "status": status,
        "verification_type": "gsm_expression_check",
        "calculation_checks": checks,
        "contradictions": incorrect,
        "verified_count": len(verified),
        "not_verifiable_count": len(not_verifiable),
        "checked_signal_count": len(checks),
        "errors": {},
    }


def gsm_final_answer_check_tool(args: dict[str, Any]) -> dict[str, Any]:
    current_step = str(args.get("current_step", "")).strip()
    previous_steps = _coerce_previous_steps(args.get("previous_steps"))
    final_answer = _extract_gsm_final_answer(current_step)
    all_checks = _extract_gsm_calculation_checks("\n".join([*previous_steps, current_step]))
    verified = [item for item in all_checks if item.get("verdict") == "verified"]
    last_verified_value = verified[-1].get("computed_result") if verified else None

    if final_answer is None:
        return {
            "status": "handled_not_verifiable",
            "verification_type": "gsm_final_answer_check",
            "final_answer_present": False,
            "not_verifiable": {"current_step": "Current step does not state a final answer."},
            "checked_signal_count": 0,
            "errors": {},
        }

    if last_verified_value is None:
        return {
            "status": "handled_not_verifiable",
            "verification_type": "gsm_final_answer_check",
            "final_answer_present": True,
            "final_answer": final_answer,
            "not_verifiable": {"trace": "No locally verified prior calculation is available for comparison."},
            "checked_signal_count": 1,
            "errors": {},
        }

    matches = _numeric_strings_equal(final_answer, str(last_verified_value))
    return {
        "status": "gsm_final_answer_advisory",
        "verification_type": "gsm_final_answer_check",
        "advisory_only": True,
        "hard_contradiction": False,
        "advisory_level": "supports_trace" if matches else "possible_final_answer_mismatch",
        "final_answer_present": True,
        "final_answer": final_answer,
        "last_verified_value": last_verified_value,
        "matches_last_verified_value": matches,
        "note": (
            "Final answer matches the last locally verified calculation."
            if matches
            else "Final answer differs from the last locally verified calculation; this is weak evidence only."
        ),
        "checked_signal_count": 1,
        "errors": {},
    }


def gsm_unsupported_number_tool(args: dict[str, Any]) -> dict[str, Any]:
    question = str(args.get("question", "")).strip()
    current_step = str(args.get("current_step", "")).strip()
    previous_steps = _coerce_previous_steps(args.get("previous_steps"))

    grounded = set(_extract_numeric_strings(question))
    previous_checks = _extract_gsm_calculation_checks("\n".join(previous_steps))
    for check in previous_checks:
        for key in ("computed_result", "claimed_result"):
            value = check.get(key)
            if value is not None:
                grounded.add(_normalize_number_token(str(value)))
    for number in _extract_numeric_strings("\n".join(previous_steps)):
        grounded.add(number)

    current_without_tags = re.sub(r"<<[^<>]+>>", " ", current_step)
    current_without_tags = re.sub(r"^\s*Step\s+\d+\s*:\s*", "", current_without_tags, flags=re.I)
    current_numbers = _extract_numeric_strings(current_without_tags)
    # Ignore step numbering and numbers that are locally justified by a calculation tag in this step.
    local_checks = _extract_gsm_calculation_checks(current_step)
    locally_justified = set()
    for check in local_checks:
        for key in ("computed_result", "claimed_result"):
            value = check.get(key)
            if value is not None:
                locally_justified.add(_normalize_number_token(str(value)))

    unsupported = [
        number
        for number in current_numbers
        if number not in grounded and number not in locally_justified
    ]
    unsupported = _unique_preserve_order(unsupported)

    return {
        "status": "gsm_number_grounding_advisory",
        "verification_type": "gsm_number_grounding",
        "advisory_only": True,
        "hard_contradiction": False,
        "advisory_level": "possible_ungrounded_number" if unsupported else "no_obvious_ungrounded_number",
        "grounded_numbers": sorted(grounded),
        "current_numbers": current_numbers,
        "unsupported_numbers": unsupported,
        "note": (
            "Some numeric claims were not directly grounded by question/prior numbers/local calculation tags."
            if unsupported
            else "No obvious ungrounded numeric claim was detected."
        ),
        "checked_signal_count": len(current_numbers),
        "errors": {},
    }


_GSM_TAG_PATTERN = re.compile(r"<<([^<>]+)>>")
_EXPLICIT_ARITH_EQUALITY_PATTERN = re.compile(
    r"(?<![A-Za-z])"
    r"([-+]?\d[\d,]*(?:\.\d+)?(?:\s*[\+\-\*/]\s*[-+]?\d[\d,]*(?:\.\d+)?)+)"
    r"\s*=\s*"
    r"([-+]?\d[\d,]*(?:\.\d+)?)"
)
_NUMBER_TOKEN_PATTERN = re.compile(r"(?<![A-Za-z])-?\d[\d,]*(?:\.\d+)?")


def _extract_gsm_calculation_checks(text: str) -> list[dict[str, Any]]:
    if not text.strip():
        return []

    checks: list[dict[str, Any]] = []
    occupied_spans: list[tuple[int, int]] = []

    for match in _GSM_TAG_PATTERN.finditer(text):
        occupied_spans.append(match.span())
        payload = match.group(1).strip()
        check = _check_arithmetic_claim(payload=payload, source="gsm_tag")
        if check is not None:
            checks.append(check)

    for match in _EXPLICIT_ARITH_EQUALITY_PATTERN.finditer(text):
        if any(match.start() >= start and match.end() <= end for start, end in occupied_spans):
            continue
        payload = f"{match.group(1)}={match.group(2)}"
        check = _check_arithmetic_claim(payload=payload, source="explicit_equality")
        if check is not None:
            checks.append(check)

    return checks


def _check_arithmetic_claim(*, payload: str, source: str) -> dict[str, Any] | None:
    if "=" not in payload:
        return {
            "source": source,
            "raw": payload,
            "verdict": "not_verifiable",
            "reason": "Calculation claim does not contain '='.",
        }
    expression_raw, claimed_raw = payload.rsplit("=", maxsplit=1)
    expression = _normalize_gsm_arithmetic_expression(expression_raw)
    claimed = _normalize_gsm_number_or_expression(claimed_raw)
    if not expression or not claimed:
        return {
            "source": source,
            "raw": payload,
            "verdict": "not_verifiable",
            "reason": "Expression or claimed result is empty after normalization.",
        }

    try:
        computed = evaluate_arithmetic_expression(expression)
    except Exception as exc:  # noqa: BLE001
        return {
            "source": source,
            "raw": payload,
            "expression": expression,
            "claimed_result": claimed,
            "verdict": "not_verifiable",
            "reason": f"Expression could not be evaluated safely: {exc}",
        }

    matches = _numeric_strings_equal(str(computed), claimed)
    return {
        "source": source,
        "raw": payload,
        "expression": expression,
        "claimed_result": claimed,
        "computed_result": computed,
        "verdict": "verified" if matches else "arithmetic_error",
    }


def _normalize_gsm_arithmetic_expression(text: str) -> str:
    normalized = text.strip()
    normalized = normalized.replace("$", "").replace(",", "")
    normalized = normalized.replace("×", "*").replace("÷", "/")
    normalized = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _normalize_gsm_number_or_expression(text: str) -> str:
    normalized = text.strip()
    normalized = normalized.replace("$", "").replace(",", "")
    normalized = normalized.replace("×", "*").replace("÷", "/")
    normalized = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", normalized)
    normalized = normalized.strip().rstrip(".")
    return normalized


def _numeric_strings_equal(left: str, right: str) -> bool:
    try:
        left_value = evaluate_arithmetic_expression(_normalize_gsm_number_or_expression(left))
        right_value = evaluate_arithmetic_expression(_normalize_gsm_number_or_expression(right))
    except Exception:  # noqa: BLE001
        return _normalize_number_token(left) == _normalize_number_token(right)
    return abs(float(left_value) - float(right_value)) <= 1e-9


def _extract_gsm_final_answer(text: str) -> str | None:
    marker_match = re.search(r"####\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)", text)
    if marker_match:
        return _normalize_gsm_number_or_expression(marker_match.group(1))
    answer_match = re.search(
        r"(?:answer is|final answer is|therefore,?\s+the answer is)\s+\$?(-?\d[\d,]*(?:\.\d+)?)",
        text,
        flags=re.I,
    )
    if answer_match:
        return _normalize_gsm_number_or_expression(answer_match.group(1))
    return None


def _extract_numeric_strings(text: str) -> list[str]:
    numbers = [_normalize_number_token(match.group(0)) for match in _NUMBER_TOKEN_PATTERN.finditer(text)]
    return _unique_preserve_order([number for number in numbers if number])


def _normalize_number_token(text: str) -> str:
    normalized = text.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        value = evaluate_arithmetic_expression(normalized)
    except Exception:  # noqa: BLE001
        return normalized
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return str(value)


_ASSIGNMENT_PATTERN = re.compile(r"\b([A-Z])\s*=\s*(.*)")
_FINAL_NUMERIC_PATTERN = re.compile(r"=\s*([-+]?\d+(?:\.\d+)?)\s*$")


def logic_check_tool(args: dict[str, Any]) -> dict[str, Any]:
    question = str(args.get("question", "")).strip()
    previous_steps_raw = args.get("previous_steps", [])
    if not isinstance(previous_steps_raw, list):
        previous_steps: list[str] = []
    else:
        previous_steps = [str(step) for step in previous_steps_raw]

    candidate_focus = args.get("candidate_focus")
    solved_components, assignment_conflicts = _parse_solved_components(previous_steps)
    all_components = _extract_question_components(question)
    if not all_components and solved_components:
        last_component = sorted(solved_components)[-1]
        if "A" <= last_component < "Z":
            all_components = [chr(ord(last_component) + 1)]
    next_unresolved_component = next((comp for comp in all_components if comp not in solved_components), None)

    required_focus = _derive_required_focus(
        question=question,
        next_unresolved_component=next_unresolved_component,
        candidate_focus=candidate_focus,
    )
    expected_value_hint = _estimate_expected_value_hint(question, next_unresolved_component)

    logic_warnings: list[str] = []
    if not previous_steps:
        logic_warnings.append("No previous step context provided.")
    logic_warnings.extend(assignment_conflicts)

    is_consistent = len(assignment_conflicts) == 0
    status = "logic_verified" if is_consistent else "logic_inconsistent"

    return {
        "status": status,
        "verification_type": "logic",
        "is_consistent": is_consistent,
        "next_unresolved_component": next_unresolved_component,
        "required_focus": required_focus,
        "logic_warnings": logic_warnings,
        "expected_value_hint": expected_value_hint,
    }


def answer_obligation_tool(args: dict[str, Any]) -> dict[str, Any]:
    question = str(args.get("question", "")).strip()
    current_step = str(args.get("current_step", "")).strip()
    previous_steps_raw = args.get("previous_steps", [])
    previous_steps = [str(item) for item in previous_steps_raw] if isinstance(previous_steps_raw, list) else []

    if not question or not current_step:
        return {
            "status": "tool_error",
            "verification_type": "obligation",
            "obligation_satisfied": False,
            "conflict_type": "invalid_input",
            "evidence": ["Both question and current_step are required non-empty strings."],
        }

    lowered_step = current_step.lower()
    lowered_question = question.lower()

    requires_answer = bool(
        re.search(r"\b(find|calculate|compute|what is|determine|solve)\b", lowered_question)
    )
    answer_marker = bool(
        re.search(r"#\s*answer|\banswer\b|\btherefore\b|\bso\b", lowered_step)
    )
    empty_answer = bool(re.search(r"\bno answer\b|\bcannot answer\b|\bcan't answer\b", lowered_step))
    has_math_payload = bool(re.search(r"[0-9]|[A-Za-z]\s*=", current_step))
    has_previous_context = len(previous_steps) > 0

    evidence: list[str] = []
    if requires_answer:
        evidence.append("Question requests a concrete result.")
    if answer_marker:
        evidence.append("Current step appears to be a conclusion/answer step.")
    if empty_answer:
        evidence.append("Current step explicitly states no answer.")
    if not has_math_payload:
        evidence.append("Current step has little or no verifiable math payload.")
    if has_previous_context:
        evidence.append("Previous steps exist and a concrete continuation is expected.")

    obligation_satisfied = True
    conflict_type = "none"
    if requires_answer and (empty_answer or (answer_marker and not has_math_payload)):
        obligation_satisfied = False
        conflict_type = "missing_required_answer"

    status = "obligation_ok" if obligation_satisfied else "obligation_conflict"
    return {
        "status": status,
        "verification_type": "obligation",
        "obligation_satisfied": obligation_satisfied,
        "conflict_type": conflict_type,
        "evidence": evidence,
    }


def condition_binding_tool(args: dict[str, Any]) -> dict[str, Any]:
    question = str(args.get("question", "")).strip()
    current_step = str(args.get("current_step", "")).strip()

    if not question or not current_step:
        return {
            "status": "tool_error",
            "verification_type": "condition_binding",
            "binding_conflict": "invalid_input",
            "evidence": ["Both question and current_step are required non-empty strings."],
        }

    question_vars = _extract_condition_variables(question)
    step_vars = _extract_condition_variables(current_step)
    evidence: list[str] = []
    conflict = "none"

    if question_vars:
        evidence.append(f"Question condition variables: {sorted(question_vars)}.")
    if step_vars:
        evidence.append(f"Step condition variables: {sorted(step_vars)}.")

    if question_vars and step_vars:
        out_of_scope = sorted(step_vars - question_vars)
        if out_of_scope:
            conflict = "hard"
            evidence.append(
                f"Step uses condition variables not defined by question conditions: {out_of_scope}."
            )
    elif question_vars and not step_vars and re.search(r"\b(if|when|case)\b", current_step.lower()):
        conflict = "soft"
        evidence.append("Step indicates case split but no clear condition variable was extracted.")

    status = "binding_ok" if conflict == "none" else "binding_conflict"
    return {
        "status": status,
        "verification_type": "condition_binding",
        "binding_conflict": conflict,
        "question_condition_vars": sorted(question_vars),
        "step_condition_vars": sorted(step_vars),
        "evidence": evidence,
    }


def contradiction_probe_tool(args: dict[str, Any]) -> dict[str, Any]:
    question = str(args.get("question", "")).strip()
    current_step = str(args.get("current_step", "")).strip()
    previous_steps_raw = args.get("previous_steps", [])
    previous_steps = [str(item) for item in previous_steps_raw] if isinstance(previous_steps_raw, list) else []

    if not question or not current_step:
        return {
            "status": "tool_error",
            "verification_type": "contradiction_probe",
            "contradiction_level": "none",
            "hard_conflict_types": [],
            "soft_conflict_types": ["invalid_input"],
            "evidence": ["Both question and current_step are required non-empty strings."],
        }

    lowered_question = question.lower()
    lowered_step = current_step.lower()
    evidence: list[str] = []
    hard_conflicts: list[str] = []
    soft_conflicts: list[str] = []

    requires_concrete_answer = bool(
        re.search(r"\b(find|calculate|compute|what is|determine|solve|how many|how much)\b", lowered_question)
    )
    empty_answer = bool(re.search(r"\bno answer\b|\bcannot answer\b|\bcan't answer\b", lowered_step))
    if requires_concrete_answer and empty_answer:
        hard_conflicts.append("missing_required_answer")
        evidence.append("Current step gives no answer while the question asks for a concrete result.")

    false_relations = _extract_false_ground_relations(current_step)
    if false_relations:
        hard_conflicts.append("false_ground_relation")
        evidence.extend(false_relations)

    assignment_conflicts = _extract_assignment_conflicts(previous_steps=previous_steps, current_step=current_step)
    if assignment_conflicts:
        hard_conflicts.append("assignment_conflict_with_previous_steps")
        evidence.extend(assignment_conflicts)

    if re.search(r"\bcontradict(?:s|ion)?\b|\bimpossible\b|\bno solution\b", lowered_step):
        if hard_conflicts:
            evidence.append("Current step also contains direct contradiction-like wording.")
        else:
            soft_conflicts.append("contradiction_like_wording_without_independent_evidence")
            evidence.append("Current step uses contradiction-like wording, but no independent hard conflict was found.")

    if not hard_conflicts and not soft_conflicts:
        evidence.append("No hard local contradiction pattern was detected.")

    contradiction_level = "hard_contradiction" if hard_conflicts else "soft_conflict" if soft_conflicts else "none"
    return {
        "status": "contradiction_probe_checked",
        "verification_type": "contradiction_probe",
        "contradiction_level": contradiction_level,
        "hard_conflict_types": _unique_preserve_order(hard_conflicts),
        "soft_conflict_types": _unique_preserve_order(soft_conflicts),
        "evidence": _unique_preserve_order(evidence),
    }


def equivalence_check_tool(args: dict[str, Any]) -> dict[str, Any]:
    question = str(args.get("question", "")).strip()
    current_step = str(args.get("current_step", "")).strip()

    if not question or not current_step:
        return {
            "status": "tool_error",
            "verification_type": "equivalence",
            "relation": "unknown",
            "advisory_level": "needs_review",
            "evidence": ["Both question and current_step are required non-empty strings."],
        }

    q_tokens = _keyword_tokens(question)
    s_tokens = _keyword_tokens(current_step)
    overlap = len(q_tokens & s_tokens)

    lowered_step = current_step.lower()
    relation = "aligned"
    advisory_level = "none"
    evidence: list[str] = []

    if any(marker in lowered_step for marker in ["equivalently", "rewrite", "convert", "let", "suppose"]):
        relation = "alternative_valid"
        evidence.append("Step contains typical transformation/alternative-path markers.")

    if overlap <= 1 and len(s_tokens) > 0:
        relation = "off_track"
        advisory_level = "needs_review"
        evidence.append("Low lexical overlap with question-specific math tokens; this is advisory, not contradiction evidence.")
    else:
        evidence.append(f"Token overlap score={overlap}.")

    if re.search(r"\bno answer\b|\bcontradict\b|\bimpossible\b", lowered_step):
        advisory_level = "needs_review"
        evidence.append("Step contains contradiction-like wording; use contradiction_probe_tool for hard conflict evidence.")

    status = "equivalence_checked"
    return {
        "status": status,
        "verification_type": "equivalence",
        "relation": relation,
        "advisory_level": advisory_level,
        "evidence": evidence,
    }


def alternative_route_verifier_tool(args: dict[str, Any]) -> dict[str, Any]:
    question = str(args.get("question", "")).strip()
    current_step = str(args.get("current_step", "")).strip()
    previous_steps_raw = args.get("previous_steps", [])
    previous_steps = [str(item) for item in previous_steps_raw] if isinstance(previous_steps_raw, list) else []

    if not question or not current_step:
        return {
            "status": "tool_error",
            "verification_type": "alternative_route",
            "valid_alternative": False,
            "hard_contradiction": False,
            "preferred_dimension": "key_analyses",
            "evidence": ["Both question and current_step are required non-empty strings."],
        }

    lowered_step = current_step.lower()
    markers = _extract_alternative_route_markers(lowered_step)
    equivalence = equivalence_check_tool(args)
    false_relations = _extract_false_ground_relations(current_step)
    assignment_conflicts = _extract_assignment_conflicts(previous_steps=previous_steps, current_step=current_step)
    hard_conflicts = [*false_relations, *assignment_conflicts]

    relation = str(equivalence.get("relation", "unknown"))
    valid_alternative = bool(
        not hard_conflicts
        and (
            relation == "alternative_valid"
            or len(markers) >= 2
            or (markers and relation == "aligned")
        )
    )
    advisory_level = "strong" if valid_alternative and len(markers) >= 2 else "weak" if valid_alternative else "none"

    evidence: list[str] = []
    if markers:
        evidence.append(f"Alternative-route markers: {markers}.")
    evidence.extend(str(item) for item in equivalence.get("evidence", []) if isinstance(item, str))
    evidence.extend(hard_conflicts)
    if not evidence:
        evidence.append("No explicit alternative-route marker or contradiction evidence was detected.")

    return {
        "status": "alternative_route_verified" if valid_alternative else "alternative_route_checked",
        "verification_type": "alternative_route",
        "valid_alternative": valid_alternative,
        "hard_contradiction": len(hard_conflicts) > 0,
        "preferred_dimension": "key_analyses",
        "advisory_level": advisory_level,
        "alternative_markers": markers,
        "evidence": _unique_preserve_order(evidence),
    }


def equivalence_substitution_verifier_tool(args: dict[str, Any]) -> dict[str, Any]:
    question = str(args.get("question", "")).strip()
    current_step = str(args.get("current_step", "")).strip()

    if not question or not current_step:
        return {
            "status": "tool_error",
            "verification_type": "equivalence_substitution",
            "valid_alternative": False,
            "valid_equivalent_transformation": False,
            "hard_contradiction": False,
            "preferred_dimension": "calculations",
            "evidence": ["Both question and current_step are required non-empty strings."],
        }

    lowered_step = current_step.lower()
    markers = _extract_transformation_markers(lowered_step)
    candidates = _extract_probe_relation_candidates(current_step)
    verified_equivalences: list[str] = []
    contradiction_evidence: list[str] = []
    not_verifiable: list[str] = []

    for expression in candidates:
        numeric_relation = _numeric_relation_check(expression)
        if numeric_relation is not None:
            if bool(numeric_relation.get("is_true", False)):
                verified_equivalences.append(f"Verified numeric relation: {expression}.")
            else:
                contradiction_evidence.append(f"False numeric relation: {expression}.")
            continue
        analyzed = _safe_symbolic_analyze_expression(expression)
        if analyzed.get("status") != "verified":
            reason = analyzed.get("reason")
            if isinstance(reason, str) and reason:
                not_verifiable.append(f"{expression}: {reason}")
            continue
        evidence = analyzed.get("evidence")
        if not isinstance(evidence, dict):
            continue
        if analyzed.get("result_type") == "equivalence":
            if evidence.get("is_equal") is False:
                contradiction_evidence.append(f"Non-equivalent equality claim: {expression}.")
            elif evidence.get("is_equal") is True:
                verified_equivalences.append(f"Verified equivalent relation: {expression}.")
        elif analyzed.get("result_type") == "relation" and str(evidence.get("simplified")) == "False":
            contradiction_evidence.append(f"False relation after symbolic check: {expression}.")

    valid_equivalent_transformation = bool(markers and not contradiction_evidence)
    valid_alternative = bool(valid_equivalent_transformation or verified_equivalences)
    evidence = [*verified_equivalences, *contradiction_evidence]
    if markers:
        evidence.append(f"Transformation markers: {markers}.")
    if not evidence and not_verifiable:
        evidence.append("Transformation intent detected but symbolic verification was limited.")
    if not evidence:
        evidence.append("No transformation-specific contradiction was detected.")

    return {
        "status": "hard_contradiction" if contradiction_evidence else "equivalence_substitution_checked",
        "verification_type": "equivalence_substitution",
        "valid_alternative": valid_alternative,
        "valid_equivalent_transformation": valid_equivalent_transformation,
        "hard_contradiction": len(contradiction_evidence) > 0,
        "preferred_dimension": "calculations",
        "transformation_markers": markers,
        "verified_equivalences": verified_equivalences,
        "contradictions": contradiction_evidence,
        "not_verifiable": not_verifiable[:8],
        "evidence": _unique_preserve_order(evidence),
    }


def condition_obligation_verifier_tool(args: dict[str, Any]) -> dict[str, Any]:
    question = str(args.get("question", "")).strip()
    current_step = str(args.get("current_step", "")).strip()

    if not question or not current_step:
        return {
            "status": "tool_error",
            "verification_type": "condition_obligation",
            "hard_contradiction": False,
            "valid_progression": False,
            "preferred_dimension": "key_analyses",
            "evidence": ["Both question and current_step are required non-empty strings."],
        }

    obligation = answer_obligation_tool(args)
    binding = condition_binding_tool(args)
    contradiction = contradiction_probe_tool(args)
    lowered_step = current_step.lower()
    has_progress_signal = bool(
        re.search(r"[0-9]|[A-Za-z]\s*=", current_step)
        or any(marker in lowered_step for marker in ["if ", "case ", "assume", "suppose", "therefore", "hence", "thus"])
    )

    hard_contradiction = bool(
        str(obligation.get("conflict_type", "none")) != "none"
        or str(binding.get("binding_conflict", "none")) == "hard"
        or str(contradiction.get("contradiction_level", "none")) == "hard_contradiction"
    )
    valid_progression = bool(has_progress_signal and not hard_contradiction)

    evidence: list[str] = []
    evidence.extend(str(item) for item in obligation.get("evidence", []) if isinstance(item, str))
    evidence.extend(str(item) for item in binding.get("evidence", []) if isinstance(item, str))
    evidence.extend(str(item) for item in contradiction.get("evidence", []) if isinstance(item, str))
    if not evidence:
        evidence.append("No condition or obligation conflict was detected.")

    return {
        "status": "hard_contradiction" if hard_contradiction else "condition_obligation_checked",
        "verification_type": "condition_obligation",
        "hard_contradiction": hard_contradiction,
        "valid_progression": valid_progression,
        "preferred_dimension": "key_analyses",
        "obligation_satisfied": bool(obligation.get("obligation_satisfied", True)),
        "binding_conflict": str(binding.get("binding_conflict", "none")),
        "contradiction_level": str(contradiction.get("contradiction_level", "none")),
        "evidence": _unique_preserve_order(evidence),
    }


def _extract_alternative_route_markers(lowered_step: str) -> list[str]:
    markers: list[str] = []
    for marker in [
        "equivalently",
        "equivalent",
        "rewrite",
        "rewritten",
        "can be written as",
        "substitute",
        "factor",
        "expand",
        "let ",
        "set ",
        "denote",
        "assume",
        "suppose",
        "consider",
        "case",
    ]:
        if marker in lowered_step:
            markers.append(marker.strip())
    return _unique_preserve_order(markers)


def _extract_transformation_markers(lowered_step: str) -> list[str]:
    markers: list[str] = []
    for marker in [
        "equivalently",
        "rewrite",
        "can be written as",
        "substitute",
        "factor",
        "expand",
        "let ",
        "set ",
        "denote",
    ]:
        if marker in lowered_step:
            markers.append(marker.strip())
    return _unique_preserve_order(markers)


def _extract_false_ground_relations(text: str) -> list[str]:
    findings: list[str] = []
    for expression in _extract_probe_relation_candidates(text):
        if not _is_ground_relation_candidate(expression):
            continue
        numeric_relation = _numeric_relation_check(expression)
        if numeric_relation is not None:
            if not bool(numeric_relation.get("is_true", False)):
                findings.append(f"Ground relation is false: {expression}.")
            continue
        analyzed = _safe_symbolic_analyze_expression(expression)
        if analyzed.get("status") != "verified":
            continue
        evidence = analyzed.get("evidence")
        if not isinstance(evidence, dict):
            continue
        if analyzed.get("result_type") == "equivalence" and evidence.get("is_equal") is False:
            findings.append(f"Ground equality is false: {expression}.")
        elif analyzed.get("result_type") == "relation" and str(evidence.get("simplified")) == "False":
            findings.append(f"Ground relation is false: {expression}.")
    return _unique_preserve_order(findings)


def _extract_probe_relation_candidates(text: str) -> list[str]:
    candidates = _extract_candidate_expressions_from_text(text, limit=12)
    for segment in re.findall(r"\$([^$]+)\$", text, flags=re.S):
        candidates.append(segment)
    for match in re.finditer(
        r"([-+()0-9.\s*/^]+(?:<=|>=|=|<|>)[-+()0-9.\s*/^]+)",
        text,
    ):
        candidates.append(match.group(1))
    return _unique_preserve_order([candidate.strip() for candidate in candidates if candidate.strip()])[:12]


def _is_ground_relation_candidate(expression: str) -> bool:
    prepared = _prepare_symbolic_text(expression)
    if not prepared or _detect_relation_operator(prepared) is None:
        return False
    names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", prepared))
    math_names = {"sqrt", "pi", "sin", "cos", "tan", "log", "ln", "exp", "E", "I"}
    return len(names - math_names) == 0


def _numeric_relation_check(expression: str) -> dict[str, Any] | None:
    operator = _detect_relation_operator(expression)
    if operator is None:
        return None
    parts = expression.split(operator, maxsplit=1)
    if len(parts) != 2:
        return None
    left_raw, right_raw = parts[0].strip(), parts[1].strip()
    if not left_raw or not right_raw:
        return None
    if not _is_numeric_expression(left_raw) or not _is_numeric_expression(right_raw):
        return None

    try:
        left_value = evaluate_arithmetic_expression(left_raw)
        right_value = evaluate_arithmetic_expression(right_raw)
    except Exception:  # noqa: BLE001
        return None

    if operator == "=":
        is_true = left_value == right_value
    elif operator == "<":
        is_true = left_value < right_value
    elif operator == ">":
        is_true = left_value > right_value
    elif operator == "<=":
        is_true = left_value <= right_value
    elif operator == ">=":
        is_true = left_value >= right_value
    else:
        return None

    return {
        "status": "verified",
        "result_type": "numeric_relation",
        "is_true": is_true,
        "left_value": left_value,
        "right_value": right_value,
        "operator": operator,
    }


def _extract_assignment_conflicts(*, previous_steps: list[str], current_step: str) -> list[str]:
    lowered_current = current_step.lower()
    if re.search(r"\b(if|when|suppose|assume|case|let)\b", lowered_current):
        return []

    previous_assignments: dict[str, Fraction] = {}
    for step in previous_steps:
        for name, value in _extract_simple_numeric_assignments(step).items():
            previous_assignments[name] = value

    conflicts: list[str] = []
    for name, value in _extract_simple_numeric_assignments(current_step).items():
        previous_value = previous_assignments.get(name)
        if previous_value is not None and previous_value != value:
            conflicts.append(f"Current step assigns {name}={value}, but previous steps assigned {name}={previous_value}.")
    return conflicts


def _extract_simple_numeric_assignments(text: str) -> dict[str, Fraction]:
    assignments: dict[str, Fraction] = {}
    for match in re.finditer(r"\b([A-Za-z][A-Za-z0-9_]*)\s*=\s*([-+]?\d+(?:\.\d+)?)\b", text):
        name = match.group(1)
        if name.lower() in {"step", "case"}:
            continue
        try:
            value = Fraction(match.group(2))
        except Exception:  # noqa: BLE001
            continue
        assignments[name] = value
    return assignments


def _extract_condition_variables(text: str) -> set[str]:
    variables: set[str] = set()
    # Examples: x <= 3, y>0, if z = 2
    for match in re.finditer(r"\b([a-zA-Z])\s*(<=|>=|=|<|>)", text):
        variables.add(match.group(1).lower())
    return variables


def _keyword_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[A-Za-z]{2,}", text.lower()))
    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "then",
        "when",
        "what",
        "find",
        "solve",
        "calculate",
        "compute",
    }
    return {token for token in tokens if token not in stop_words}


def _is_numeric_expression(expression: str) -> bool:
    compact = expression.replace(" ", "")
    if "\\" in compact or "$" in compact:
        return False
    if re.search(r"[A-Za-z]", compact):
        return False
    if "^" in compact:
        return False
    return bool(re.fullmatch(r"[0-9\.\+\-\*\/%\(\)\s]+", expression))


def _safe_symbolic_analyze_expression(expression: str) -> dict[str, Any]:
    try:
        return _symbolic_analyze_expression(expression)
    except Exception as exc:  # noqa: BLE001
        return {"status": "not_verifiable", "reason": f"Symbolic runtime error: {exc}"}


_CHAIN_INEQUALITY_PATTERN = re.compile(
    r"^\s*([-+]?\d+)\s*(<=|<)\s*([A-Za-z][A-Za-z0-9_]*)\s*(<=|<)\s*([-+]?\d+)\s*$"
)
_QUESTION_INTERVAL_PATTERN = re.compile(r"[\[\(]\s*(-?\d+)\s*,\s*(-?\d+)\s*[\]\)]")


def _extract_constraint_candidates(source: str) -> list[str]:
    if not source.strip():
        return []
    candidates: list[str] = []
    seen: set[str] = set()

    def add(item: str) -> None:
        value = item.strip().rstrip(".,;:")
        if not value or value in seen:
            return
        seen.add(value)
        candidates.append(value)

    for segment in re.findall(r"\$([^$]+)\$", source, flags=re.S):
        if "<" in segment or ">" in segment or "=" in segment:
            add(segment)

    for match in re.finditer(r"(-?\d+\s*(?:<=|<)\s*[A-Za-z][A-Za-z0-9_]*\s*(?:<=|<)\s*-?\d+)", source):
        add(match.group(1))

    for match in re.finditer(r"([A-Za-z][A-Za-z0-9_]*\s*=\s*[^,\n]+)", source):
        add(match.group(1))
        if len(candidates) >= 16:
            break

    return candidates[:16]


def _extract_integer_domain_from_question(question: str) -> tuple[int, int] | None:
    lowered = question.lower()
    if "integer" not in lowered and "integers" not in lowered:
        return None
    match = _QUESTION_INTERVAL_PATTERN.search(question)
    if not match:
        return None
    left = int(match.group(1))
    right = int(match.group(2))
    if left <= right:
        return left, right
    return right, left


def _analyze_chain_inequality(*, expression: str, question: str) -> dict[str, Any] | None:
    normalized = expression.strip()
    match = _CHAIN_INEQUALITY_PATTERN.match(normalized)
    if not match:
        return None

    left_value = int(match.group(1))
    left_op = match.group(2)
    variable = match.group(3)
    right_op = match.group(4)
    right_value = int(match.group(5))
    if left_value > right_value:
        return None

    low = left_value + (1 if left_op == "<" else 0)
    high = right_value - (1 if right_op == "<" else 0)
    base_values = list(range(low, high + 1)) if low <= high else []

    domain = _extract_integer_domain_from_question(question)
    if domain is not None:
        domain_low, domain_high = domain
        satisfying_values = [value for value in base_values if domain_low <= value <= domain_high]
    else:
        satisfying_values = base_values

    evidence_values: list[int] | str
    if len(satisfying_values) <= 30:
        evidence_values = satisfying_values
    else:
        evidence_values = (
            f"{satisfying_values[:5]} ... {satisfying_values[-5:]} "
            f"(total={len(satisfying_values)})"
        )

    return {
        "status": "verified",
        "result_type": "integer_constraint",
        "evidence": {
            "expression": normalized,
            "variable": variable,
            "interval_bounds": [left_value, right_value],
            "interval_ops": [left_op, right_op],
            "integer_domain": list(domain) if domain is not None else None,
            "satisfying_count": len(satisfying_values),
            "satisfying_values": evidence_values,
        },
    }


def _symbolic_analyze_expression(expression: str) -> dict[str, Any]:
    if any(marker in expression for marker in ("\\begin", "\\end", "\\require", "\\enclose", "\\overline")):
        return {"status": "not_verifiable", "reason": "Unsupported display-math or decorated latex block."}

    prepared = _prepare_symbolic_text(expression)
    if not prepared:
        return {"status": "not_verifiable", "reason": "Empty symbolic expression after normalization."}

    try:
        sympy_ctx = _import_sympy()
    except Exception as exc:  # noqa: BLE001
        return {"status": "not_verifiable", "reason": f"SymPy unavailable: {exc}"}

    sp = sympy_ctx["sp"]
    parse_expr = sympy_ctx["parse_expr"]
    transformations = sympy_ctx["transformations"]

    relation_operator = _detect_relation_operator(prepared)
    if relation_operator is not None:
        parts = prepared.split(relation_operator, maxsplit=1)
        if len(parts) != 2:
            return {"status": "not_verifiable", "reason": "Relation could not be split into two sides."}
        left_raw, right_raw = parts[0].strip(), parts[1].strip()
        if not left_raw or not right_raw:
            return {"status": "not_verifiable", "reason": "Relation missing left or right side."}
        left, left_error = _sympy_parse(
            expression=left_raw,
            sp=sp,
            parse_expr=parse_expr,
            transformations=transformations,
        )
        right, right_error = _sympy_parse(
            expression=right_raw,
            sp=sp,
            parse_expr=parse_expr,
            transformations=transformations,
        )
        if left_error or right_error:
            return {
                "status": "not_verifiable",
                "reason": f"Symbolic parse failed: left_error={left_error}, right_error={right_error}",
            }
        if relation_operator == "=":
            difference = sp.simplify(left - right)
            is_equal = bool(difference == 0)
            return {
                "status": "verified",
                "result_type": "equivalence",
                "evidence": {
                    "normalized_left": str(left),
                    "normalized_right": str(right),
                    "difference": str(difference),
                    "is_equal": is_equal,
                },
            }
        relation = sp.Rel(left, right, relation_operator)
        simplified = sp.simplify(relation)
        return {
            "status": "verified",
            "result_type": "relation",
            "evidence": {
                "normalized_left": str(left),
                "normalized_right": str(right),
                "relation": relation_operator,
                "simplified": str(simplified),
            },
        }

    parsed, parse_error = _sympy_parse(
        expression=prepared,
        sp=sp,
        parse_expr=parse_expr,
        transformations=transformations,
    )
    if parse_error:
        return {"status": "not_verifiable", "reason": f"Symbolic parse failed: {parse_error}"}
    simplified = sp.simplify(parsed)
    return {
        "status": "verified",
        "result_type": "simplified",
        "evidence": {
            "normalized_expression": str(parsed),
            "simplified": str(simplified),
        },
    }


def _prepare_symbolic_text(expression: str) -> str:
    text = expression.strip()
    if not text:
        return ""

    text = text.replace("\u2212", "-").replace("−", "-")
    text = text.replace("\u00d7", "*").replace("\u00f7", "/")
    text = text.replace("（", "(").replace("）", ")")

    math_blocks = re.findall(r"\$\$(.*?)\$\$", text, flags=re.DOTALL)
    if math_blocks:
        text = math_blocks[0]
    else:
        inline_blocks = re.findall(r"\$(.*?)\$", text, flags=re.DOTALL)
        if inline_blocks:
            text = inline_blocks[0] if len(inline_blocks) == 1 else " + ".join(inline_blocks)

    text = text.strip().strip("`")
    text = re.sub(r"\\left|\\right", "", text)
    text = text.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\div", "/")
    text = text.replace(r"\pi", "pi")

    text = _replace_latex_frac(text)
    text = _replace_latex_sqrt(text)
    text = _replace_caret_power(text)
    text = text.replace("{", "(").replace("}", ")")
    if re.search(r"\\[a-zA-Z]+", text):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _replace_latex_frac(text: str) -> str:
    while True:
        index = text.find(r"\frac")
        if index < 0:
            return text
        numerator, next_index = _parse_braced_segment(text, index + len(r"\frac"))
        if numerator is None:
            return text
        denominator, end_index = _parse_braced_segment(text, next_index)
        if denominator is None:
            return text
        replacement = f"(({numerator})/({denominator}))"
        text = text[:index] + replacement + text[end_index:]


def _replace_latex_sqrt(text: str) -> str:
    while True:
        index = text.find(r"\sqrt")
        if index < 0:
            return text
        radicand, end_index = _parse_braced_segment(text, index + len(r"\sqrt"))
        if radicand is None:
            return text
        replacement = f"sqrt({radicand})"
        text = text[:index] + replacement + text[end_index:]


def _parse_braced_segment(text: str, start_index: int) -> tuple[str | None, int]:
    index = start_index
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text) or text[index] != "{":
        return None, start_index

    depth = 0
    for cursor in range(index, len(text)):
        char = text[cursor]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[index + 1 : cursor], cursor + 1
    return None, start_index


def _replace_caret_power(text: str) -> str:
    text = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", text)
    text = re.sub(r"\^([A-Za-z0-9\.\-]+)", r"**(\1)", text)
    return text


def _detect_relation_operator(text: str) -> str | None:
    for operator in ("<=", ">=", "=", "<", ">"):
        if operator in text:
            return operator
    return None


def _sympy_parse(*, sp: Any, parse_expr: Any, transformations: Any, expression: str) -> tuple[Any, str | None]:
    normalized = expression.strip()
    if not normalized:
        return None, "Empty symbolic side."

    local_dict: dict[str, Any] = {"pi": sp.pi, "E": sp.E}
    function_names = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", normalized))
    for name in function_names:
        if name in {"sin", "cos", "tan", "cot", "sec", "csc", "sqrt", "log", "ln", "exp"}:
            continue
        local_dict[name] = sp.Function(name)
    symbol_names = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", normalized))
    for name in symbol_names:
        if name in local_dict:
            continue
        if hasattr(sp, name):
            continue
        local_dict[name] = sp.Symbol(name)

    parse_input = normalized.replace("^", "**")
    try:
        parsed = parse_expr(parse_input, local_dict=local_dict, transformations=transformations, evaluate=True)
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    return parsed, None


def _import_sympy() -> dict[str, Any]:
    import sympy as sp  # type: ignore
    from sympy.parsing.sympy_parser import (  # type: ignore
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    return {
        "sp": sp,
        "parse_expr": parse_expr,
        "transformations": transformations,
    }


def _parse_solved_components(previous_steps: list[str]) -> tuple[set[str], list[str]]:
    solved_components: set[str] = set()
    value_by_component: dict[str, str] = {}
    conflicts: list[str] = []

    for index, step in enumerate(previous_steps):
        match = _ASSIGNMENT_PATTERN.search(step)
        if not match:
            continue
        component = match.group(1)
        solved_components.add(component)

        value_match = _FINAL_NUMERIC_PATTERN.search(step)
        if not value_match:
            continue
        value = value_match.group(1)
        previous_value = value_by_component.get(component)
        if previous_value is not None and previous_value != value:
            conflicts.append(
                f"Component {component} has conflicting assigned values ({previous_value} vs {value}) at step {index}."
            )
        value_by_component[component] = value

    return solved_components, conflicts


def _extract_question_components(question: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for token in re.findall(r"\b([A-Z])\b", question):
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def _derive_required_focus(
    *,
    question: str,
    next_unresolved_component: str | None,
    candidate_focus: Any,
) -> list[str]:
    focus_items: list[str] = []
    if next_unresolved_component:
        focus_items.append(f"Resolve component {next_unresolved_component} before final composition.")

    if re.search(r"-\s*-\s*", question) or "--" in question.replace(" ", ""):
        focus_items.append("Handle double negatives carefully.")

    has_mul_or_div = "*" in question or "/" in question
    has_add_or_sub = "+" in question or "-" in question
    if has_mul_or_div and has_add_or_sub:
        focus_items.append("Apply multiplication/division before addition/subtraction.")

    if "(" in question and ")" in question:
        focus_items.append("Respect parentheses from inner to outer.")

    if "**" in question or "^" in question:
        focus_items.append("Apply exponent precedence before linear operations.")

    if isinstance(candidate_focus, str) and candidate_focus.strip():
        focus_items.append(candidate_focus.strip())

    if not focus_items:
        focus_items.append("Follow operator precedence and verify the local arithmetic path.")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in focus_items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _estimate_expected_value_hint(question: str, unresolved_component: str | None) -> str | None:
    expression = question.split("=", maxsplit=1)[0].strip()
    if not expression:
        return None

    candidates = _extract_balanced_subexpressions(expression)
    if not candidates:
        candidates = [expression]

    ranked: list[str] = []
    for candidate in candidates:
        compact = candidate.strip()
        if compact and compact not in ranked:
            ranked.append(compact)

    target_candidates = ranked
    if unresolved_component and unresolved_component >= "C":
        with_mul = [item for item in ranked if "*" in item]
        if with_mul:
            target_candidates = with_mul

    for candidate in target_candidates:
        try:
            value = evaluate_arithmetic_expression(candidate)
        except Exception:  # noqa: BLE001
            continue
        subject = unresolved_component if unresolved_component else "target expression"
        return f"{subject} may evaluate to {value}."
    return None


def _extract_balanced_subexpressions(expression: str) -> list[str]:
    start_stack: list[int] = []
    candidates: list[str] = []
    for index, char in enumerate(expression):
        if char == "(":
            start_stack.append(index)
        elif char == ")" and start_stack:
            start = start_stack.pop()
            sub = expression[start + 1 : index].strip()
            if sub and any(op in sub for op in "+-*/"):
                candidates.append(sub)
    candidates.sort(key=len, reverse=True)
    return candidates


_MAX_EXPR_LENGTH = 256
_ALLOWED_CHARS_PATTERN = re.compile(r"^[0-9\.\+\-\*\/%\(\)\s]+$")


def evaluate_arithmetic_expression(expression: str) -> int | float:
    normalized = expression.strip()
    if not normalized:
        raise ValueError("Expression is empty.")
    if len(normalized) > _MAX_EXPR_LENGTH:
        raise ValueError("Expression is too long.")
    if not _ALLOWED_CHARS_PATTERN.fullmatch(normalized):
        raise ValueError("Expression contains unsupported characters.")
    if "**" in normalized:
        raise ValueError("Exponentiation is disabled for safety.")

    try:
        parsed = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Expression is not valid Python arithmetic syntax.") from exc

    value = _eval_arithmetic_ast(parsed.body)
    if value.denominator == 1:
        return int(value.numerator)
    return float(value)


def _eval_arithmetic_ast(node: ast.AST) -> Fraction:
    if isinstance(node, ast.BinOp):
        left = _eval_arithmetic_ast(node.left)
        right = _eval_arithmetic_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ValueError("Division by zero.")
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise ValueError("Division by zero.")
            return Fraction(left // right, 1)
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise ValueError("Division by zero.")
            if left.denominator != 1 or right.denominator != 1:
                raise ValueError("Modulo only supports integers.")
            return Fraction(int(left) % int(right), 1)
        raise ValueError(f"Unsupported operator: {type(node.op).__name__}")

    if isinstance(node, ast.UnaryOp):
        operand = _eval_arithmetic_ast(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Only numeric constants are supported.")
        return Fraction(str(value))

    raise ValueError(f"Unsupported syntax node: {type(node).__name__}")
