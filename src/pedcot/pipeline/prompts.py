from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _templates_dir() -> Path:
    return Path(__file__).with_name("templates")


def _load_template(name: str) -> str:
    return (_templates_dir() / name).read_text(encoding="utf-8")


class PedCoTPromptBuilder:
    _BOX_MARKERS = ("<|begin_of_box|>", "<|end_of_box|>")

    def __init__(self) -> None:
        self._stage1_template = _load_template("stage1.txt")
        self._stage2_template = _load_template("stage2.txt")

    @staticmethod
    def format_steps(steps: list[str]) -> str:
        if not steps:
            return ""
        return "\n".join(f"(step {index + 1}) {step}" for index, step in enumerate(steps))

    @staticmethod
    def format_current_step(step_index: int, step: str) -> str:
        return f"(step {step_index + 1}) {step}"

    @classmethod
    def _strip_box_markers(cls, text: str) -> str:
        cleaned = text.strip()
        for marker in cls._BOX_MARKERS:
            cleaned = cleaned.replace(marker, "")
        return cleaned.strip()

    @staticmethod
    def _sanitize_prompt_text(text: str) -> str:
        # Some provider gateways reject certain LaTeX-style backslash tokens as invalid request parameters.
        # Normalize common prime notation while preserving semantics for geometry/algebra questions.
        cleaned = re.sub(r"\^\{\\prime\}", "'", text)
        cleaned = re.sub(r"\^\\prime", "'", cleaned)
        cleaned = cleaned.replace("\\prime", "'")
        return cleaned

    @staticmethod
    def _normalize_stage1_section(text: str, canonical_heading: str) -> str:
        cleaned = PedCoTPromptBuilder._strip_box_markers(text)
        cleaned = PedCoTPromptBuilder._sanitize_prompt_text(cleaned)
        lines = cleaned.splitlines()
        if not lines:
            return canonical_heading

        heading_variants = [
            canonical_heading.rstrip(":"),
            "Mathematical Concepts",
            "Key Analyses",
            "Mathematical Expressions",
        ]
        first_line = lines[0].strip()
        normalized_first = re.sub(r"^\s*[123]\.\s*", "", first_line).strip().rstrip(":")
        if any(normalized_first.lower() == variant.rstrip(":").lower() for variant in heading_variants):
            lines = lines[1:]

        body = "\n".join(lines).strip()
        if not body:
            return canonical_heading
        return f"{canonical_heading}\n{body}"

    def build_stage1_prompt(self, question: str, previous_steps: list[str]) -> str:
        safe_question = self._sanitize_prompt_text(question)
        safe_previous_steps = [self._sanitize_prompt_text(step) for step in previous_steps]
        return self._stage1_template.format(
            question=safe_question,
            initial_steps=self.format_steps(safe_previous_steps),
        ).strip()

    def build_stage2_prompt(
        self,
        question: str,
        previous_steps: list[str],
        step_index: int,
        current_step: str,
        stage1_concepts: str,
        stage1_analyses: str,
        stage1_calculations: str,
    ) -> str:
        safe_question = self._sanitize_prompt_text(question)
        safe_previous_steps = [self._sanitize_prompt_text(step) for step in previous_steps]
        safe_current_step = self._sanitize_prompt_text(current_step)
        return self._stage2_template.format(
            question=safe_question,
            initial_steps=self.format_steps(safe_previous_steps),
            actual_next_step=self.format_current_step(step_index, safe_current_step),
            stage1_concepts=self._normalize_stage1_section(
                stage1_concepts,
                "Mathematical Concepts to Apply:",
            ),
            stage1_analyses=self._normalize_stage1_section(
                stage1_analyses,
                "Key Analyses for the Next Step:",
            ),
            stage1_calculations=self._normalize_stage1_section(
                stage1_calculations,
                "Mathematical Expressions to Compute:",
            ),
        ).strip()

    def build_stage2_step_type_prompt(
        self,
        *,
        question: str,
        previous_steps: list[str],
        step_index: int,
        current_step: str,
    ) -> str:
        safe_question = self._sanitize_prompt_text(question)
        safe_previous_steps = [self._sanitize_prompt_text(step) for step in previous_steps]
        safe_current_step = self._sanitize_prompt_text(current_step)
        return f"""You are classifying the type of one math reasoning step for verifier routing.
Classify the current step in context. Do not judge whether the step is correct. Do not solve the full question.

Question: {safe_question}
Previous steps:
{self.format_steps(safe_previous_steps)}
Current step:
{self.format_current_step(step_index, safe_current_step)}

Choose exactly one step_type from:
- decomposition
- final_conclusion
- condition_case
- substitution
- algebraic_transformation
- arithmetic
- reasoning_transition

Meaning of the labels:
- decomposition: split or rewrite one expression into components.
- final_conclusion: present the final answer or concluding statement.
- condition_case: introduce a case split, assumption, branch, or conditional obligation.
- substitution: plug in, replace, or substitute a known value/expression.
- algebraic_transformation: rewrite, factor, expand, denote, or equivalently transform an expression.
- arithmetic: mainly perform numeric calculation or simplification.
- reasoning_transition: general progress that does not strongly match the other categories.

Output requirements:
- Output exactly one JSON object and nothing else.
- Use this schema:
{{
  "step_type": "<one allowed label>",
  "reasoning": "<short explanation>",
  "risk_flags": ["<short snake_case flag>", "..."],
  "confidence": <number between 0 and 1>
}}
""".strip()

    def build_stage2_review_prompt(
        self,
        *,
        question: str,
        previous_steps: list[str],
        step_index: int,
        current_step: str,
        stage2_response: str,
        stage2_labels: dict[str, str | None],
        tool_evidence: list[dict],
    ) -> str:
        safe_question = self._sanitize_prompt_text(question)
        safe_previous_steps = [self._sanitize_prompt_text(step) for step in previous_steps]
        safe_current_step = self._sanitize_prompt_text(current_step)
        safe_stage2_response = self._sanitize_prompt_text(stage2_response)
        evidence_text = json.dumps(tool_evidence, ensure_ascii=False, indent=2)
        labels_text = json.dumps(stage2_labels, ensure_ascii=False, indent=2)
        return f"""You are reviewing a Stage-2 mistake-detection judgment for a BIG-Bench multistep arithmetic trace.
Use the original dataset annotations only during evaluation; do not assume any gold label here.

Question: {safe_question}
Initial steps:
{self.format_steps(safe_previous_steps)}
The actual next step is:
{self.format_current_step(step_index, safe_current_step)}

Original Stage-2 labels:
{labels_text}

Original Stage-2 response:
{safe_stage2_response}

Local arithmetic/tool evidence:
{evidence_text}

Fixed review checklist:
1. Conflict with the question conditions.
2. Conflict with previous steps.
3. Wrong theorem, concept, arithmetic rule, or precedence rule.
4. Treating a temporary assumption as a conclusion.
5. Missing required case analysis.
6. Treating incomplete reasoning as a final conclusion.
7. Calculation error.
8. Broken operator precedence, parenthesis structure, or left-to-right evaluation for equal-precedence operators.
9. Mishandled negative signs, double negatives, or subtracting a negative number.
10. Variable decomposition, variable assignment, or final substitution changes the original expression structure.

Review rules:
- Keep contradiction-found only when there is hard evidence of a mathematical conflict.
- Hard evidence includes adjacent equality-chain expressions with different values, variable values conflicting with prior computed values, final substitution conflicting with prior variable values, or a decomposition that is provably not equivalent to the original expression.
- Do not mark contradiction-found merely because a step is incomplete, does not calculate yet, uses a different but equivalent decomposition granularity, or is less detailed than the reference.
- If a tool reports status=hard_contradiction, explicitly use that evidence unless the tool result is irrelevant to the current step.
- If no hard contradiction is supported, use correct-and-aligned, reasonable-but-incomplete, or nothing-extracted as appropriate.

Output format requirements:
- Output plain text only.
- Output exactly three top-level sections and no extra top-level sections.
- Start each section on its own line with these exact headings:
1. Mathematical Concepts to Apply:
2. Key Analyses for the Next Step:
3. Mathematical Expressions to Compute:
- In each section, include exactly one final line in the format: Label: <label>.
- The <label> must be exactly one of: correct-and-aligned, reasonable-but-incomplete, nothing-extracted, contradiction-found.
- Do not add a separate summary such as "Final Labels".
""".strip()

    def build_stage2_specialist_review_prompt(
        self,
        *,
        question: str,
        previous_steps: list[str],
        step_index: int,
        current_step: str,
        stage2_response: str,
        stage2_labels: dict[str, str | None],
        step_type_meta: dict[str, Any],
        route_meta: dict[str, Any],
        specialist_evidence: list[dict[str, Any]],
    ) -> str:
        safe_question = self._sanitize_prompt_text(question)
        safe_previous_steps = [self._sanitize_prompt_text(step) for step in previous_steps]
        safe_current_step = self._sanitize_prompt_text(current_step)
        safe_stage2_response = self._sanitize_prompt_text(stage2_response)
        labels_text = json.dumps(stage2_labels, ensure_ascii=False, indent=2)
        step_type_text = json.dumps(step_type_meta, ensure_ascii=False, indent=2)
        route_text = json.dumps(route_meta, ensure_ascii=False, indent=2)
        evidence_text = json.dumps(specialist_evidence, ensure_ascii=False, indent=2)
        return f"""You are a specialist verifier agent for math-step mistake detection.
Your job is to review the original Stage-2 judgment using step-type classification and specialist verifier evidence.
Do not assume the reference path is the only valid path.

Question: {safe_question}
Initial steps:
{self.format_steps(safe_previous_steps)}
The actual next step is:
{self.format_current_step(step_index, safe_current_step)}

Original Stage-2 labels:
{labels_text}

Original Stage-2 response:
{safe_stage2_response}

Step type classifier output:
{step_type_text}

Specialist router output:
{route_text}

Specialist verifier evidence:
{evidence_text}

Specialist review rules:
- Treat the step type classifier as routing context, not as proof by itself.
- Treat the specialist router output as routing context and confidence metadata, not as proof by itself.
- Treat any verifier signal with status=hard_contradiction or hard_contradiction=true as hard mathematical evidence unless it is clearly irrelevant to the current step.
- If specialist evidence supports a different-but-valid route, an equivalent rewrite, or a conditionally valid branch, do not use contradiction-found merely because the step differs from the reference path.
- Use contradiction-found only when there is hard evidence of mathematical conflict.
- If evidence is mixed or incomplete but not contradictory, prefer reasonable-but-incomplete over contradiction-found.
- Preserve separate judgment for the three dimensions: mathematical concepts, key analyses, and calculations.

Output format requirements:
- Output plain text only.
- Output exactly three top-level sections and no extra top-level sections.
- Start each section on its own line with these exact headings:
1. Mathematical Concepts to Apply:
2. Key Analyses for the Next Step:
3. Mathematical Expressions to Compute:
- In each section, include exactly one final line in the format: Label: <label>.
- The <label> must be exactly one of: correct-and-aligned, reasonable-but-incomplete, nothing-extracted, contradiction-found.
- Do not add a separate summary such as "Final Labels".
""".strip()
