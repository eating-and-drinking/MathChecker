"""Per-channel likelihood functions p(evidence | tau).

Each evidence channel (stage2 LLM logits, specialist tool output, review
output) is modeled as a noisy channel that, when conditioned on the true
first-mistake position tau, emits a stochastic observation. PRISM combines
these via Bayes' rule -- they are NOT decision-makers, they are sensors.

The likelihoods are intentionally simple and parametric. A single calibration
parameter (`sensitivity`, in [0, 1]) controls how peaked the channel is:
sensitivity = 1 corresponds to a hard, deterministic signal; sensitivity = 0
corresponds to an uninformative channel. Calibration in production should set
sensitivity per-channel by minimizing negative log-likelihood on a held-out
slice (see scripts/train_prism_router.py for the harness).

Why this form? Two reasons:

1. Closed-form EIG. With factorized likelihoods of this shape, the Expected
   Information Gain in eig.py admits a closed-form summation over a small
   number of "evidence outcomes" per channel.

2. Identifiability. Each channel surfaces a different attribute of the step
   (label class, specialist evidence type, review verdict). Sharing a single
   sensitivity scalar prevents over-fitting on small calibration sets.

Theoretical reference: this is the noisy-OR / noisy-channel decomposition
that appears in Krause & Guestrin (2005) for submodular sensor placement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


# ---- core Likelihood vector ----

@dataclass(slots=True, frozen=True)
class Likelihood:
    """A likelihood vector p(evidence | tau) of length T+1.

    Index k in [0, T) is the "first mistake at step k" hypothesis. Index T is
    the "no error" hypothesis.
    """

    values: tuple[float, ...]
    source: str = "unspecified"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:  # type: ignore[override]
        if any(v < 0.0 for v in self.values):
            raise ValueError("Likelihood values must be non-negative")

    @property
    def num_hypotheses(self) -> int:
        return len(self.values)

    def as_list(self) -> list[float]:
        return list(self.values)


# ---- stage2 label channel ----

# This index list matches PRINCIPLE_LABELS in core.constants and reflects how
# strongly each label points at "this step is the first mistake".
_LABEL_TO_MISTAKE_PROB = {
    "correct-and-aligned": 0.02,
    "reasonable-but-incomplete": 0.10,
    "nothing-extracted": 0.20,
    "contradiction-found": 0.90,
}


def _label_mistake_prob(label: str | None) -> float:
    if label is None:
        return 0.20
    return _LABEL_TO_MISTAKE_PROB.get(label.strip().lower(), 0.20)


def make_stage2_likelihood(
    *,
    step_index: int,
    num_steps: int,
    principle_labels: Sequence[str | None],
    sensitivity: float = 0.85,
) -> Likelihood:
    """Likelihood for the stage2 label observation at one step.

    The trick: a step looking like a hard contradiction (high "mistake prob")
    is consistent with tau = step_index AND with tau <= step_index (if some
    earlier step was already the first mistake, then this one being broken is
    par for the course). A clean-looking step is consistent with tau > step
    or tau = infty.

    We turn the per-principle labels into a single mistake probability
    `q = max_principle p_label_mistake(principle)` (since contradiction along
    any axis is sufficient to call the step broken), then build a sigmoidal
    likelihood profile.
    """
    if not 0.0 <= sensitivity <= 1.0:
        raise ValueError("sensitivity must be in [0, 1]")
    if not principle_labels:
        return Likelihood(values=tuple([1.0] * (num_steps + 1)), source="stage2:empty")

    q = max(_label_mistake_prob(label) for label in principle_labels)

    # Build likelihood vector.
    # tau == step_index    : strongly consistent with mistake observation
    # tau <  step_index    : already-broken trace; this step being broken is
    #                        weakly informative -- moderate likelihood for either outcome.
    # tau >  step_index    : trace should be clean here; observation of mistake
    #                        is unlikely.
    # tau == infty (no err): everything should be clean.
    values: list[float] = []
    s = sensitivity
    base = 1.0 - s  # uninformative floor

    for tau in range(num_steps):
        if tau == step_index:
            # Probability of seeing "mistake-evidence q" given tau = here.
            # Mix q (the observation) with sensitivity.
            p = base + s * q
        elif tau < step_index:
            # Already broken; this step's evidence is uninformative.
            p = base + s * 0.5
        else:  # tau > step_index
            # This step should be clean. Probability of mistake observation = 1-q.
            p = base + s * (1.0 - q)
        values.append(max(p, 1e-6))

    # tau = infty: everything should be clean.
    values.append(max(base + s * (1.0 - q), 1e-6))

    return Likelihood(
        values=tuple(values),
        source=f"stage2:step{step_index}",
        meta={"q": q, "sensitivity": sensitivity},
    )


# ---- specialist evidence channel ----

# Specialist tools emit a structured payload. We coarsen it to two scalars:
#   - hard_conflict_strength in [0, 1] : evidence that the step has a hard math conflict
#   - valid_alternative_strength in [0, 1]: evidence that the step is a valid alternative
# These are extracted by predictor adapters from the actual tool trace.


def make_specialist_likelihood(
    *,
    step_index: int,
    num_steps: int,
    hard_conflict_strength: float,
    valid_alternative_strength: float = 0.0,
    sensitivity: float = 0.75,
    source: str = "specialist",
) -> Likelihood:
    """Likelihood for a specialist tool emission at one step.

    Hard conflict evidence pushes posterior toward "tau = step_index" (or
    "tau <= step_index"). Valid-alternative evidence pushes posterior toward
    "tau > step_index" or "tau = infty" -- the step is fine, just different.
    """
    if not 0.0 <= sensitivity <= 1.0:
        raise ValueError("sensitivity must be in [0, 1]")
    h = max(0.0, min(1.0, float(hard_conflict_strength)))
    a = max(0.0, min(1.0, float(valid_alternative_strength)))

    # Combine into a "mistake here" probability.
    # Hard evidence dominates; valid-alternative evidence suppresses.
    q = max(0.0, min(1.0, h * (1.0 - a)))

    s = sensitivity
    base = 1.0 - s
    values: list[float] = []
    for tau in range(num_steps):
        if tau == step_index:
            p = base + s * q
        elif tau < step_index:
            p = base + s * 0.5
        else:
            p = base + s * (1.0 - q)
        values.append(max(p, 1e-6))
    values.append(max(base + s * (1.0 - q), 1e-6))
    return Likelihood(
        values=tuple(values),
        source=f"{source}:step{step_index}",
        meta={
            "hard_conflict": h,
            "valid_alternative": a,
            "q": q,
            "sensitivity": sensitivity,
        },
    )


# ---- stage1 consistency channel ----

def make_stage1_likelihood(
    *,
    step_index: int,
    num_steps: int,
    inconsistency_strength: float,
    sensitivity: float = 0.55,
) -> Likelihood:
    """Likelihood for the stage1 soft-reference consistency observation.

    Sensitivity is set lower than stage2 (0.85) and specialists (0.75) because
    the inconsistency extractor is a heuristic regex/eval combo, not a
    calibrated LLM judge. The channel's role is to add an independent signal
    when stage1 made a concrete numeric prediction the step contradicted;
    when stage1 didn't say anything specific, the strength is 0 and this
    likelihood is uniform (no posterior change).
    """
    if not 0.0 <= sensitivity <= 1.0:
        raise ValueError("sensitivity must be in [0, 1]")
    q = max(0.0, min(1.0, float(inconsistency_strength)))
    if q == 0.0:
        # No signal -> uniform likelihood.
        return Likelihood(
            values=tuple([1.0] * (num_steps + 1)),
            source=f"stage1:step{step_index}:no_signal",
            meta={"q": 0.0, "sensitivity": sensitivity},
        )
    s = sensitivity
    base = 1.0 - s
    values: list[float] = []
    for tau in range(num_steps):
        if tau == step_index:
            p = base + s * q
        elif tau < step_index:
            p = base + s * 0.5
        else:
            p = base + s * (1.0 - q)
        values.append(max(p, 1e-6))
    values.append(max(base + s * (1.0 - q), 1e-6))
    return Likelihood(
        values=tuple(values),
        source=f"stage1:step{step_index}",
        meta={"q": q, "sensitivity": sensitivity},
    )


# ---- review channel ----

def make_review_likelihood(
    *,
    step_index: int,
    num_steps: int,
    review_mistake_prob: float,
    sensitivity: float = 0.6,
) -> Likelihood:
    """Likelihood for a review verdict (legacy stage2_review style).

    Used inside Gibbs refine. Sensitivity is intentionally lower because the
    review channel is correlated with the stage2 channel (same LLM family,
    similar prompts), and we don't want to double-count.
    """
    q = max(0.0, min(1.0, float(review_mistake_prob)))
    s = sensitivity
    base = 1.0 - s
    values: list[float] = []
    for tau in range(num_steps):
        if tau == step_index:
            p = base + s * q
        elif tau < step_index:
            p = base + s * 0.5
        else:
            p = base + s * (1.0 - q)
        values.append(max(p, 1e-6))
    values.append(max(base + s * (1.0 - q), 1e-6))
    return Likelihood(
        values=tuple(values),
        source=f"review:step{step_index}",
        meta={"q": q, "sensitivity": sensitivity},
    )


# ---- contradiction-strength extraction from principle labels ----

def principle_labels_to_contradiction_strength(
    principle_labels,
) -> float:
    """Squash per-principle labels to a single contradiction strength in [0, 1]."""
    if not principle_labels:
        return 0.0
    return max(_label_mistake_prob(label) for label in principle_labels)


def principle_labels_to_logits(
    principle_labels,
    *,
    contradiction_label_index: int = 3,
    temperature: float = 1.0,
):
    """Produce a 4-class logit vector for a step's principle labels.

    Index order matches core.constants.PRINCIPLE_LABELS:
      0: correct-and-aligned
      1: reasonable-but-incomplete
      2: nothing-extracted
      3: contradiction-found
    """
    del contradiction_label_index
    q = principle_labels_to_contradiction_strength(principle_labels)
    p_contradiction = q
    p_reasonable = (1.0 - q) * 0.35
    p_correct = (1.0 - q) * 0.55
    p_nothing = (1.0 - q) * 0.10
    probs = [p_correct, p_reasonable, p_nothing, p_contradiction]
    eps = 1e-6
    return [math.log(p + eps) / max(temperature, 1e-3) for p in probs]
