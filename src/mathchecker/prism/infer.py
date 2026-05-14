"""PRISM inference loop.

Pure-logic core. No LLM calls; all evidence is provided via callbacks. The
loop lives here so the algorithm can be tested deterministically with
synthetic evidence.

    init posterior pi over {0, ..., T-1, infty}
    for each step t:
        stage2 evidence  -> Bayes update pi
        select specialists via greedy EIG against pi
        for each selected specialist (or all in offline replay):
            invoke -> Bayes update pi
        Gibbs refine + Phi-invariant projection
        if observability gate AND max pi >= conformal alpha_t:
            return (argmax pi, committed=True)
    return (argmax pi, committed=False)  # abstain

Compared to the legacy predictor, three layers collapse into one:
  - stage2_review            )
  - stage2_specialist_review  >  -> Gibbs refine
  - deterministic_adjustment )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from .conformal import ConformalSchedule, default_schedule
from .eig import (
    DEFAULT_SPECIALIST_CANDIDATES,
    SpecialistCandidate,
    greedy_select,
)
from .likelihoods import (
    make_review_likelihood,
    make_specialist_likelihood,
    make_stage1_likelihood,
    make_stage2_likelihood,
    principle_labels_to_logits,
)
from .posterior import Posterior, length_prior


CONTRADICTION_LABEL_INDEX = 3


# ---- evidence carriers ----

@dataclass(slots=True, frozen=True)
class PrismEvidence:
    step_index: int
    principle_labels: tuple[str | None, ...]
    stage2_sensitivity: float = 0.85
    specialist_emissions: dict[str, tuple[float, float]] = field(default_factory=dict)
    review_mistake_prob: float | None = None
    # Stage1 soft-reference consistency: scalar in [0, 1]. 0 means no signal,
    # higher means stronger evidence the current step contradicts the
    # independent stage1 prediction. See prism/stage1_consistency.py.
    stage1_inconsistency: float = 0.0
    stage1_sensitivity: float = 0.55


@dataclass(slots=True)
class StepTrace:
    step_index: int
    posterior_snapshot: list[float]
    selected_specialists: list[str]
    eig_scores: dict[str, float]
    contradiction_strength: float
    committed: bool = False


@dataclass(slots=True)
class PrismResult:
    pred_first_mistake_index: int | None
    committed: bool
    abstained: bool
    final_posterior: Posterior
    step_traces: list[StepTrace]
    num_specialist_calls: int

    def to_dict(self) -> dict:
        return {
            "pred_first_mistake_index": self.pred_first_mistake_index,
            "committed": self.committed,
            "abstained": self.abstained,
            "num_specialist_calls": self.num_specialist_calls,
            "final_posterior": self.final_posterior.as_dict(),
            "step_traces": [
                {
                    "step_index": st.step_index,
                    "posterior": st.posterior_snapshot,
                    "selected_specialists": st.selected_specialists,
                    "eig_scores": st.eig_scores,
                    "contradiction_strength": st.contradiction_strength,
                    "committed": st.committed,
                }
                for st in self.step_traces
            ],
        }


SpecialistInvoker = Callable[[str, int], tuple[float, float]]


def prism_infer(
    *,
    num_steps: int,
    evidence_at_step: Callable[[int], PrismEvidence],
    specialist_invoker: SpecialistInvoker | None = None,
    candidates: Sequence[SpecialistCandidate] = DEFAULT_SPECIALIST_CANDIDATES,
    schedule: ConformalSchedule | None = None,
    budget: int = 3,
    lam: float = 0.05,
    p_no_error_prior: float = 0.4,
    early_stop: bool = True,
) -> PrismResult:
    schedule = schedule or default_schedule(num_steps=num_steps)

    posterior = Posterior(
        num_steps=num_steps,
        probs=length_prior(num_steps, p_no_error=p_no_error_prior),
    )

    step_traces: list[StepTrace] = []
    num_specialist_calls = 0
    committed = False

    per_step_logits: list[list[float]] = [
        [0.0, 0.0, 0.0, 0.0] for _ in range(num_steps)
    ]
    contradiction_strengths: list[float] = [0.0] * num_steps

    for t in range(num_steps):
        bundle = evidence_at_step(t)
        if bundle is None:
            break

        # (0) Stage1 consistency channel (independent of stage2 / specialists).
        # Fires first because stage1 is generated BEFORE seeing the step's
        # actual text, so it is an exogenous prediction the step has to match.
        stage1_lik = None
        if bundle.stage1_inconsistency > 0.0:
            stage1_lik = make_stage1_likelihood(
                step_index=t,
                num_steps=num_steps,
                inconsistency_strength=bundle.stage1_inconsistency,
                sensitivity=bundle.stage1_sensitivity,
            )
            posterior.bayes_update(stage1_lik.values)

        # (1) Stage2 channel
        stage2_lik = make_stage2_likelihood(
            step_index=t,
            num_steps=num_steps,
            principle_labels=list(bundle.principle_labels),
            sensitivity=bundle.stage2_sensitivity,
        )
        posterior.bayes_update(stage2_lik.values)
        per_step_logits[t] = principle_labels_to_logits(list(bundle.principle_labels))

        # (2) EIG-driven specialist selection (for diagnostics + production gating)
        selected, eig_scores = greedy_select(
            posterior=posterior,
            step_index=t,
            candidates=list(candidates),
            budget=budget,
            lam=lam,
        )

        # (3) Specialist channel updates
        if specialist_invoker is not None:
            # Production: invoke only what EIG selected.
            for cand in selected:
                hard, valid_alt = specialist_invoker(cand.name, t)
                num_specialist_calls += 1
                spec_lik = make_specialist_likelihood(
                    step_index=t,
                    num_steps=num_steps,
                    hard_conflict_strength=hard,
                    valid_alternative_strength=valid_alt,
                    sensitivity=cand.sensitivity,
                    source=cand.name,
                )
                posterior.bayes_update(spec_lik.values)
        else:
            # Offline replay: consume all emissions already in the bundle.
            for tool_name, (hard, valid_alt) in bundle.specialist_emissions.items():
                cand = next(
                    (c for c in candidates if c.name == tool_name),
                    SpecialistCandidate(name=tool_name, sensitivity=0.75),
                )
                spec_lik = make_specialist_likelihood(
                    step_index=t,
                    num_steps=num_steps,
                    hard_conflict_strength=hard,
                    valid_alternative_strength=valid_alt,
                    sensitivity=cand.sensitivity,
                    source=cand.name,
                )
                posterior.bayes_update(spec_lik.values)

        # (4) Review channel (optional)
        if bundle.review_mistake_prob is not None:
            rev_lik = make_review_likelihood(
                step_index=t,
                num_steps=num_steps,
                review_mistake_prob=bundle.review_mistake_prob,
            )
            posterior.bayes_update(rev_lik.values)

        # Contradiction-strength bookkeeping.
        # Combine FOUR channels into ONE effective per-step contradiction signal:
        #   - q_stage1     : stage1 soft-reference inconsistency (numeric)
        #   - q_stage2     : label-based probability that this step is broken
        #   - q_specialist : specialist's hard-conflict strength
        #   - valid_alt    : specialist's valid-alternative strength (suppressor)
        # Valid-alternative evidence multiplicatively SUPPRESSES contradiction
        # signals from the other channels (a step that looks broken but is in
        # fact a valid alternative should not push the posterior onto tau = step).
        q_stage1 = float(bundle.stage1_inconsistency)
        q_stage2 = stage2_lik.meta.get("q", 0.0) if stage2_lik.meta else 0.0
        q_specialist_hard = 0.0
        max_valid_alt = 0.0
        for emission in bundle.specialist_emissions.values():
            hard, valid_alt = emission
            if hard > q_specialist_hard:
                q_specialist_hard = hard
            if valid_alt > max_valid_alt:
                max_valid_alt = valid_alt
        suppression = 1.0 - float(max_valid_alt)
        contradiction_strengths[t] = max(
            float(q_stage1) * suppression,
            float(q_stage2) * suppression,
            float(q_specialist_hard) * suppression,
        )

        posterior.mark_observed(t)

        # (5) Phi-invariant projection.
        # The legacy Gibbs-refine step has been folded into this single
        # invariant projection -- both are projections onto the same support,
        # and applying them sequentially double-counts the label channel.
        posterior.apply_phi_invariant(
            contradiction_strength_at=contradiction_strengths,
        )

        # (6) Snapshot + observability-gated conformal stop
        snapshot = list(posterior.probs)
        argmax_idx = posterior.argmax_index()
        observable_commit = (
            (argmax_idx == -1 and posterior.observed_up_to == num_steps)
            or (argmax_idx != -1 and argmax_idx < posterior.observed_up_to)
        )
        threshold_commit = schedule.should_commit(
            step_index=t,
            max_posterior_mass=posterior.max_mass(),
        )
        step_committed = observable_commit and threshold_commit
        step_traces.append(
            StepTrace(
                step_index=t,
                posterior_snapshot=snapshot,
                selected_specialists=[cand.name for cand in selected],
                eig_scores=dict(eig_scores),
                contradiction_strength=contradiction_strengths[t],
                committed=step_committed,
            )
        )

        if step_committed and early_stop:
            committed = True
            break

    if not committed and step_traces and step_traces[-1].committed:
        committed = True

    argmax_idx = posterior.argmax_index()
    pred_first_mistake_index = None if argmax_idx == -1 else argmax_idx

    return PrismResult(
        pred_first_mistake_index=pred_first_mistake_index,
        committed=committed,
        abstained=not committed,
        final_posterior=posterior,
        step_traces=step_traces,
        num_specialist_calls=num_specialist_calls,
    )
