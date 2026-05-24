"""PrismPredictor: glue layer that drives PRISM with the existing LLM stack.

This predictor reuses every evidence-producing piece of `PedCoTPredictor`
(stage1, stage2 with specialist tooling, the LLM client and caching layer)
but replaces the *decision* layers:

  - legacy stage2_review            -> Gibbs refine
  - legacy stage2_specialist_review -> Gibbs refine
  - deterministic_specialist_adjustment -> Phi-invariant projection
  - "break on first contradiction-found" stop rule -> conformal commit
  - heuristic/learned router        -> greedy submodular EIG

The split is:
  Evidence production:  PedCoTPredictor._run_stage1, ._run_stage2
  Decision making:      PRISM (this file)

If you want to ablate one layer (e.g. turn EIG off and force all
specialists), pass `budget=len(candidates)` and `lam=0`.
"""
from __future__ import annotations

from typing import Any, Sequence

from ..core.constants import NEGATIVE_TRACE_LABEL, POSITIVE_TRACE_LABEL
from ..core.models import (
    Stage2Parse,
    StepPrediction,
    TraceExample,
    TracePrediction,
)
from ..pipeline.predictor import PedCoTPredictor
from .conformal import ConformalSchedule, default_schedule
from .eig import DEFAULT_SPECIALIST_CANDIDATES, SpecialistCandidate
from .eprocess import EProcessSchedule
from .infer import PrismEvidence, PrismResult, StoppingRule, prism_infer
from .joint_calibration import TemperatureMixer
from .likelihoods import default_sensitivity
from .posterior import Posterior, length_prior
from .stage1_consistency import extract_stage1_inconsistency


_SPECIALIST_TOOL_NAMES = tuple(c.name for c in DEFAULT_SPECIALIST_CANDIDATES)


class PrismPredictor:
    """LLM-backed PRISM predictor.

    Parameters
    ----------
    base_predictor : PedCoTPredictor
        Re-used for stage1 / stage2 LLM evidence production. Its
        stage2_router_mode should be left at "step-type" -- PRISM ignores
        the router result and uses EIG instead, but stage2 still needs the
        triad specialist toolset to be wired so the tool handlers are
        available for invocation.
    candidates : sequence of SpecialistCandidate
        Specialist registry.
    schedule : optional ConformalSchedule
        Stopping threshold schedule. If None, default_schedule is used per trace.
    budget, lam :
        EIG hyperparameters.
    p_no_error_prior :
        Prior mass on "trace has no error".
    """

    def __init__(
        self,
        *,
        base_predictor: PedCoTPredictor,
        candidates: Sequence[SpecialistCandidate] = DEFAULT_SPECIALIST_CANDIDATES,
        schedule: StoppingRule | None = None,
        delta: float = 0.1,
        budget: int = 3,
        lam: float = 0.05,
        p_no_error_prior: float = 0.4,
        mixer: TemperatureMixer | None = None,
    ) -> None:
        self.base = base_predictor
        self.candidates = list(candidates)
        self.schedule = schedule
        self.delta = delta
        self.budget = budget
        self.lam = lam
        self.p_no_error_prior = p_no_error_prior
        self.mixer = mixer

    # ---- public API ----

    def predict_trace(self, example: TraceExample, model: str) -> TracePrediction:
        """Run PRISM over a single trace.

        Strategy: we cannot fully decouple LLM evidence production from PRISM
        because the existing PedCoTPredictor produces stage1+stage2 together
        and the specialist tools are invoked *during* the stage2 LLM call as
        function-call attachments. So we use a hybrid scheme:

          (a) Let PedCoTPredictor produce stage2_parse and a tool_trace for
              each step. This is one LLM call per step that already includes
              ALL three specialists as available tools.
          (b) PRISM consumes the resulting evidence bundle and decides:
                * how to update the posterior
                * whether to commit (conformal stop)
                * which specialists *would* have been worth invoking
                  (recorded in step_traces for analysis but not re-run; in
                   the production codepath specialists are already cheap once
                   stage2 has been issued).

        This still gives PRISM the routing decision authority -- it records
        which specialists EIG would have chosen, and the calibration script
        can later replay traces in `specialist_invoker=None` offline mode to
        confirm the budget savings.

        A pure EIG-gated production loop would require splitting stage2 into
        "stage2 without specialist tools" + "follow-up specialist calls per
        selected tool"; that's wired but out of scope for this refactor and
        lives behind run_with_gated_specialists().
        """
        num_steps = len(example.steps)
        if num_steps == 0:
            return TracePrediction(
                example_id=example.example_id,
                dataset=example.dataset,
                model=model,
                pred_first_mistake_index=None,
                pred_trace_label=POSITIVE_TRACE_LABEL,
                gold_first_mistake_index=example.gold_first_mistake_index,
                gold_trace_label=example.gold_trace_label,
                steps=[],
                completed=True,
            )

        schedule = self.schedule or default_schedule(num_steps=num_steps, delta=self.delta)

        # We drive the loop ourselves so we can interleave LLM calls with
        # posterior updates and stop early on commit.
        posterior = Posterior(
            num_steps=num_steps,
            probs=length_prior(num_steps, p_no_error=self.p_no_error_prior),
        )
        contradiction_strengths: list[float] = [0.0] * num_steps
        step_rows: list[dict[str, Any]] = []
        pred_first_mistake_index: int | None = None
        committed = False
        num_specialist_calls = 0

        for step_index in range(num_steps):
            (
                step_pred,
                stage_failure_reason,
            ) = self._evidence_for_step(
                example=example,
                model=model,
                step_index=step_index,
            )
            if step_pred is None:
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
                    error=stage_failure_reason,
                )

            step_rows.append(step_pred.to_dict())

            # Extract PRISM evidence from the existing step prediction.
            current_step_text = (
                example.steps[step_index]
                if step_index < len(example.steps)
                else ""
            )
            evidence = self._extract_evidence(
                step_pred,
                step_index=step_index,
                current_step_text=current_step_text,
            )
            num_specialist_calls += len(evidence.specialist_emissions)

            # Apply PRISM updates inline.
            from .likelihoods import (
                make_specialist_likelihood,
                make_stage1_likelihood,
                make_stage2_likelihood,
            )

            mixer = self.mixer

            def _temper(vec, channel):
                v = list(vec)
                if mixer is not None:
                    v = mixer.temper(likelihood=v, channel=channel)
                return v

            # (0) Stage1 consistency channel.
            if evidence.stage1_inconsistency > 0.0:
                stage1_lik = make_stage1_likelihood(
                    step_index=step_index,
                    num_steps=num_steps,
                    inconsistency_strength=evidence.stage1_inconsistency,
                    sensitivity=evidence.stage1_sensitivity,
                )
                posterior.bayes_update(_temper(stage1_lik.values, "stage1"))

            # (1) Stage2 label channel.
            stage2_lik = make_stage2_likelihood(
                step_index=step_index,
                num_steps=num_steps,
                principle_labels=list(evidence.principle_labels),
                sensitivity=evidence.stage2_sensitivity,
            )
            posterior.bayes_update(_temper(stage2_lik.values, "stage2"))

            q_specialist_hard = 0.0
            max_valid_alt = 0.0
            for tool_name, (hard, valid_alt) in evidence.specialist_emissions.items():
                try:
                    cand = next(c for c in self.candidates if c.name == tool_name)
                except StopIteration:
                    cand = SpecialistCandidate(name=tool_name, sensitivity=0.75)
                spec_lik = make_specialist_likelihood(
                    step_index=step_index,
                    num_steps=num_steps,
                    hard_conflict_strength=hard,
                    valid_alternative_strength=valid_alt,
                    sensitivity=cand.sensitivity,
                    source=cand.name,
                )
                posterior.bayes_update(_temper(spec_lik.values, cand.name))
                if hard > q_specialist_hard:
                    q_specialist_hard = hard
                if valid_alt > max_valid_alt:
                    max_valid_alt = valid_alt

            q_stage1 = float(evidence.stage1_inconsistency)
            q_stage2 = stage2_lik.meta.get("q", 0.0) if stage2_lik.meta else 0.0
            suppression = 1.0 - float(max_valid_alt)
            contradiction_strengths[step_index] = max(
                float(q_stage1) * suppression,
                float(q_stage2) * suppression,
                float(q_specialist_hard) * suppression,
            )
            posterior.mark_observed(step_index)

            posterior.apply_phi_invariant(
                contradiction_strength_at=contradiction_strengths,
            )

            argmax = posterior.argmax_index()
            observable_commit = (
                (argmax == -1 and posterior.observed_up_to == num_steps)
                or (argmax != -1 and argmax < posterior.observed_up_to)
            )
            threshold_commit = schedule.should_commit(
                step_index=step_index,
                max_posterior_mass=posterior.max_mass(),
                posterior_probs=list(posterior.probs),
                argmax_index=argmax,
            )
            if observable_commit and threshold_commit:
                committed = True
                pred_first_mistake_index = None if argmax == -1 else argmax
                break

        if not committed:
            # Ran to end without committing -- still output MAP, but flag.
            argmax = posterior.argmax_index()
            pred_first_mistake_index = None if argmax == -1 else argmax

        pred_trace_label = (
            POSITIVE_TRACE_LABEL if pred_first_mistake_index is None else NEGATIVE_TRACE_LABEL
        )

        # Attach a PRISM diagnostic to the trace prediction via a final
        # synthetic step row carrying the posterior + metadata. This keeps
        # downstream evaluation code unchanged while letting analyses opt in.
        meta_step = self._build_prism_meta_row(
            posterior=posterior,
            committed=committed,
            num_specialist_calls=num_specialist_calls,
            schedule=schedule,
            contradiction_strengths=contradiction_strengths,
        )
        step_rows.append(meta_step)

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

    # ---- evidence extraction ----

    def _evidence_for_step(
        self,
        *,
        example: TraceExample,
        model: str,
        step_index: int,
    ) -> tuple[StepPrediction | None, str | None]:
        """Run stage1 + stage2 for one step. Returns (step_pred, fail_reason).

        We re-use PedCoTPredictor's private machinery. Any stage failure
        translates into an aborted trace (matching legacy behavior).
        """
        try:
            (
                stage1_response,
                stage1_parse,
                stage1_attempts,
                stage1_prompt,
                stage1_tool_trace,
                stage1_tool_errors,
                stage1_tool_status,
            ) = self.base._run_stage1(
                example=example,
                step_index=step_index,
                model=model,
            )
        except Exception as exc:  # noqa: BLE001
            return None, f"stage1 exception: {exc}"

        if not stage1_parse.success:
            return None, stage1_parse.error or "stage1 parse failure"

        try:
            (
                stage2_response,
                stage2_parse,
                stage2_attempts,
                stage2_prompt,
                stage2_tool_trace,
                stage2_tool_errors,
                stage2_tool_status,
            ) = self.base._run_stage2(
                example=example,
                step_index=step_index,
                model=model,
                stage1_parse=stage1_parse,
            )
        except Exception as exc:  # noqa: BLE001
            return None, f"stage2 exception: {exc}"

        if not stage2_parse.success:
            return None, stage2_parse.error or "stage2 parse failure"

        step_pred = StepPrediction(
            step_index=step_index,
            pred_step_label=self._step_label(stage2_parse),
            stage1_prompt=stage1_prompt,
            stage2_prompt=stage2_prompt,
            stage1_raw_response=stage1_response,
            stage2_raw_response=stage2_response,
            stage1_parse=stage1_parse.to_dict(),
            stage2_parse=stage2_parse.to_dict(),
            stage2_step_type=stage2_tool_status.get("stage2_step_type"),
            principle_labels=stage2_parse.principle_labels,
            stage1_tool_trace=stage1_tool_trace,
            stage1_tool_errors=stage1_tool_errors,
            stage2_tool_trace=stage2_tool_trace,
            stage2_tool_errors=stage2_tool_errors,
            stage1_attempts=stage1_attempts,
            stage2_attempts=stage2_attempts,
            parse_status={
                "stage1_success": True,
                "stage2_success": True,
                **stage1_tool_status,
                **stage2_tool_status,
                "prism_pipeline": True,
            },
        )
        return step_pred, None

    @staticmethod
    def _step_label(stage2_parse: Stage2Parse) -> int:
        labels = stage2_parse.principle_labels.values()
        return 0 if "contradiction-found" in labels else 1

    def _extract_evidence(
        self,
        step_pred: StepPrediction,
        *,
        step_index: int,
        current_step_text: str = "",
    ) -> PrismEvidence:
        """Turn the StepPrediction's tool trace into a PrismEvidence bundle."""
        principle_labels: tuple[str | None, ...] = tuple(step_pred.principle_labels.values())

        # Stage1 consistency extraction (independent channel).
        stage1_parse = step_pred.stage1_parse or {}
        stage1_calculations = stage1_parse.get("calculations") if isinstance(stage1_parse, dict) else None
        stage1_signal = extract_stage1_inconsistency(
            stage1_calculations=stage1_calculations,
            current_step=current_step_text,
        )

        specialist_emissions: dict[str, tuple[float, float]] = {}
        for item in step_pred.stage2_tool_trace:
            if not isinstance(item, dict) or item.get("discarded"):
                continue
            tool_name = item.get("tool_name")
            result = item.get("result")
            if not isinstance(tool_name, str) or tool_name not in _SPECIALIST_TOOL_NAMES:
                continue
            if not isinstance(result, dict):
                continue
            hard = self._extract_hard_conflict_strength(result)
            valid_alt = self._extract_valid_alternative_strength(result)
            # Keep the strongest emission if multiple calls per specialist.
            prev = specialist_emissions.get(tool_name)
            if prev is None or hard > prev[0]:
                specialist_emissions[tool_name] = (hard, valid_alt)
        return PrismEvidence(
            step_index=step_index,
            principle_labels=principle_labels,
            stage2_sensitivity=default_sensitivity("stage2"),
            specialist_emissions=specialist_emissions,
            stage1_inconsistency=float(stage1_signal.inconsistency_strength),
            stage1_sensitivity=default_sensitivity("stage1"),
        )

    @staticmethod
    def _extract_hard_conflict_strength(result: dict[str, Any]) -> float:
        if result.get("hard_contradiction"):
            return 0.92
        status = result.get("status")
        if isinstance(status, str) and "hard" in status.lower():
            return 0.88
        contradictions = result.get("contradictions")
        if isinstance(contradictions, list) and contradictions:
            return 0.65
        level = result.get("contradiction_level")
        if isinstance(level, str):
            if level == "hard_contradiction":
                return 0.90
            if level == "soft_contradiction":
                return 0.55
        binding = result.get("binding_conflict")
        if binding == "hard":
            return 0.85
        return 0.10  # default: no evidence of hard conflict

    @staticmethod
    def _extract_valid_alternative_strength(result: dict[str, Any]) -> float:
        if result.get("valid_alternative") or result.get("valid_equivalent_transformation"):
            return 0.85
        if result.get("relation") == "alternative_valid":
            return 0.80
        return 0.05

    @staticmethod
    def _build_prism_meta_row(
        *,
        posterior: Posterior,
        committed: bool,
        num_specialist_calls: int,
        schedule: StoppingRule,
        contradiction_strengths: list[float],
    ) -> dict[str, Any]:
        if isinstance(schedule, ConformalSchedule):
            schedule_payload: dict[str, Any] = {
                "type": "conformal",
                "delta": schedule.delta,
                "alpha": list(schedule.alpha),
                "calibration_size": schedule.calibration_size,
                "meta": schedule.meta,
            }
        elif isinstance(schedule, EProcessSchedule):
            schedule_payload = schedule.to_dict()
        else:
            schedule_payload = {"type": "unknown", "repr": repr(schedule)}
        return {
            "step_index": -1,
            "prism_meta": True,
            "posterior": posterior.as_dict(),
            "committed": committed,
            "abstained": not committed,
            "num_specialist_calls": num_specialist_calls,
            "conformal_schedule": schedule_payload,
            "stopping_schedule": schedule_payload,
            "contradiction_strengths": list(contradiction_strengths),
        }
