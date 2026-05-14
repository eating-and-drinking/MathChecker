"""Stage1 soft-reference consistency channel.

PRISM treats `stage1` (math concepts + key analyses + expected calculations)
as an independent evidence channel. The intuition: stage1 was generated
BEFORE looking at the current step, so it is an independent prediction of
what the next step should compute. If the actual step computes something
that materially contradicts stage1's prediction, that's strong evidence
the step is the first mistake.

This file implements a conservative extractor that turns stage1_parse +
current_step into a single scalar `inconsistency_strength in [0, 1]`. The
extractor only returns a high value when it has CONCRETE evidence (same
left-hand side, different right-hand side in a numeric equation). When
the comparison is ambiguous (text doesn't parse cleanly, stage1 makes no
specific numeric prediction), it returns 0.0 (no signal). The likelihood
in prism/likelihoods.py will then act as a no-op for that step.

Conservative-by-default keeps the false-positive rate low. The price is
that stage1 is silent on many steps -- which is fine, since the stage2 and
specialist channels are still active.
"""
from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass


# ---- public API ----

@dataclass(slots=True, frozen=True)
class Stage1ConsistencyResult:
    inconsistency_strength: float  # [0, 1]
    matched_equations: int
    contradictions: tuple[str, ...]
    source: str  # "numeric_mismatch" | "no_signal" | "missing"

    def to_dict(self) -> dict:
        return {
            "inconsistency_strength": self.inconsistency_strength,
            "matched_equations": self.matched_equations,
            "contradictions": list(self.contradictions),
            "source": self.source,
        }


def extract_stage1_inconsistency(
    *,
    stage1_calculations: str | None,
    current_step: str | None,
) -> Stage1ConsistencyResult:
    """Compare stage1's expected calculations with what the step actually did.

    Strategy:
      1. Extract numeric equations of form "lhs = rhs" from BOTH texts.
      2. Normalize lhs (whitespace, operator symbols).
      3. For each equation in the current step that shares an lhs with a stage1
         equation, compare RHS values. If they differ by more than a small
         tolerance, that is a hard inconsistency.
      4. If at least one equation in the current step is internally inconsistent
         (e.g. "5 * 20 = 110" — lhs evaluates to 100, not 110), flag too.
      5. Otherwise return no_signal.

    This intentionally does NOT try to do semantic / NLP comparison; that would
    be unreliable on free-form math reasoning text.
    """
    if not current_step:
        return Stage1ConsistencyResult(0.0, 0, (), "missing")
    if not stage1_calculations:
        # Without stage1 prediction we cannot compare; fall back to checking
        # whether the current step is internally consistent (e.g. 5*20 = 110).
        return _internal_consistency_only(current_step)

    stage1_equations = _extract_equations(stage1_calculations)
    step_equations = _extract_equations(current_step)

    contradictions: list[str] = []
    matched = 0

    if stage1_equations and step_equations:
        for lhs_norm, step_rhs in step_equations.items():
            stage1_rhs = stage1_equations.get(lhs_norm)
            if stage1_rhs is None:
                continue
            matched += 1
            if not _values_match(step_rhs, stage1_rhs):
                contradictions.append(
                    f"stage1 expected '{lhs_norm} = {stage1_rhs}' but step has '{lhs_norm} = {step_rhs}'"
                )

    # Also do an internal-consistency pass: does the step's own arithmetic check out?
    for lhs_norm, step_rhs in step_equations.items():
        evaluated = _try_eval(lhs_norm)
        if evaluated is None:
            continue
        if not _values_match(str(evaluated), step_rhs):
            contradictions.append(
                f"step claims '{lhs_norm} = {step_rhs}' but evaluation gives {evaluated}"
            )

    if contradictions:
        # Hard inconsistency. We cap at 0.92 to avoid 0/1 absolute claims.
        return Stage1ConsistencyResult(
            inconsistency_strength=0.92,
            matched_equations=matched,
            contradictions=tuple(contradictions),
            source="numeric_mismatch",
        )

    return Stage1ConsistencyResult(0.0, matched, (), "no_signal")


# ---- internal helpers ----

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
# Capture A = B where A and B can include numbers, parens, basic ops.
# We constrain LHS to look "math-ish" to avoid matching English sentences.
_EQUATION_RE = re.compile(
    r"([+\-]?[\d\.\(\)\s\*/\+\-x×÷=^]+?)\s*=\s*([+\-]?\d[\d\.\(\)\s\*/\+\-x×÷^]*)"
)
# Some additional cleanup mappings.
_OPERATOR_MAP = {"×": "*", "x": "*", "·": "*", "÷": "/", "^": "**"}


def _extract_equations(text: str) -> dict[str, str]:
    """Extract a {normalized_lhs: rhs_str} map from `text`.

    Only keeps equations whose LHS contains at least one digit and one
    arithmetic operator (so that "Step 3 = compute X" doesn't match).
    """
    found: dict[str, str] = {}
    for raw_lhs, raw_rhs in _EQUATION_RE.findall(text):
        lhs = _normalize_expr(raw_lhs)
        rhs = _normalize_expr(raw_rhs)
        # Require LHS to look like real arithmetic: digit + operator.
        if not _NUMBER_RE.search(lhs):
            continue
        if not any(op in lhs for op in ("+", "-", "*", "/", "(")):
            continue
        # Require RHS to be a pure numeric (or simple arithmetic) expression.
        if not _NUMBER_RE.search(rhs):
            continue
        # Last write wins; deliberate.
        found[lhs] = rhs
    return found


def _normalize_expr(expr: str) -> str:
    s = expr.strip()
    for k, v in _OPERATOR_MAP.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+", "", s)
    return s


# Safe arithmetic evaluator using ast.
_AST_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _try_eval(expr: str) -> float | None:
    """Safely evaluate a normalized arithmetic expression.

    Returns None on parse failure, division-by-zero, or anything non-trivial.
    """
    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    return _eval_ast(node.body)


def _eval_ast(node) -> float | None:
    if isinstance(node, ast.Num):
        return float(node.n)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if left is None or right is None:
            return None
        op = _AST_OPS.get(type(node.op))
        if op is None:
            return None
        try:
            return op(left, right)
        except ZeroDivisionError:
            return None
    if isinstance(node, ast.UnaryOp):
        v = _eval_ast(node.operand)
        if v is None:
            return None
        op = _AST_OPS.get(type(node.op))
        return op(v) if op else None
    return None


def _values_match(a: str, b: str, tol: float = 1e-6) -> bool:
    """Compare two numeric strings (or simple expressions) for equality.

    Falls back to literal equality if numeric evaluation fails.
    """
    va = _try_eval(a)
    vb = _try_eval(b)
    if va is not None and vb is not None:
        return abs(va - vb) <= tol * max(1.0, abs(va), abs(vb))
    return a.strip() == b.strip()


def _internal_consistency_only(current_step: str) -> Stage1ConsistencyResult:
    """When stage1 calculations are missing, still flag self-contradictory steps."""
    step_equations = _extract_equations(current_step)
    contradictions: list[str] = []
    for lhs_norm, step_rhs in step_equations.items():
        evaluated = _try_eval(lhs_norm)
        if evaluated is None:
            continue
        if not _values_match(str(evaluated), step_rhs):
            contradictions.append(
                f"step claims '{lhs_norm} = {step_rhs}' but evaluation gives {evaluated}"
            )
    if contradictions:
        return Stage1ConsistencyResult(0.85, 0, tuple(contradictions), "numeric_mismatch")
    return Stage1ConsistencyResult(0.0, 0, (), "no_signal")
