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


# ---- calibration loading ----

# Default hand-coded values (used when no calibration file is present).
_DEFAULT_LABEL_MISTAKE_PROB = {
    "correct-and-aligned": 0.02,
    "reasonable-but-incomplete": 0.10,
    "nothing-extracted": 0.20,
    "contradiction-found": 0.90,
}
_DEFAULT_SENSITIVITIES = {"stage1": 0.55, "stage2": 0.85, "specialist": 0.75}

# Active values. Mutable: load_calibration() overwrites these.
_LABEL_TO_MISTAKE_PROB: dict[str, float] = dict(_DEFAULT_LABEL_MISTAKE_PROB)
_SENSITIVITIES: dict[str, float] = dict(_DEFAULT_SENSITIVITIES)
# Per-specialist isotonic calibration: tool_name -> {"x": [...], "y": [...]}.
_SPECIALIST_ISOTONIC: dict[str, dict] = {}
_CALIBRATION_META: dict = {"loaded": False, "version": None, "calibration_size": 0}


def load_calibration(path: "str | Path | None" = None) -> dict:
    """Load calibrated likelihood parameters from JSON.

    Looks at `path` if given, else at the default
    artifacts/prism_likelihoods.json relative to CWD. Falls back silently to
    hard-coded defaults if absent. Idempotent and safe to call multiple times.
    """
    import json as _json
    from pathlib import Path as _Path

    target = _Path(path) if path is not None else _Path("artifacts/prism_likelihoods.json")
    if not target.exists():
        _CALIBRATION_META.update({"loaded": False})
        return dict(_CALIBRATION_META)
    try:
        payload = _json.loads(target.read_text(encoding="utf-8"))
    except (OSError, _json.JSONDecodeError):
        return dict(_CALIBRATION_META)

    global _LABEL_TO_MISTAKE_PROB, _SENSITIVITIES, _SPECIALIST_ISOTONIC
    label_map = payload.get("label_mistake_prob") or {}
    sensitivities = payload.get("sensitivities") or {}
    isotonic = payload.get("specialist_isotonic") or {}

    if isinstance(label_map, dict):
        _LABEL_TO_MISTAKE_PROB = {
            k: float(v) for k, v in label_map.items() if isinstance(v, (int, float))
        }
    if isinstance(sensitivities, dict):
        _SENSITIVITIES = {
            k: float(v) for k, v in sensitivities.items() if isinstance(v, (int, float))
        }
    if isinstance(isotonic, dict):
        _SPECIALIST_ISOTONIC = {
            k: v for k, v in isotonic.items() if isinstance(v, dict)
        }

    _CALIBRATION_META.update(
        loaded=True,
        version=payload.get("version"),
        calibration_size=int(payload.get("calibration_size", 0)),
    )
    return dict(_CALIBRATION_META)


def reset_calibration_to_defaults() -> None:
    """Restore hand-coded defaults. Useful for tests that mutate calibration."""
    global _LABEL_TO_MISTAKE_PROB, _SENSITIVITIES, _SPECIALIST_ISOTONIC
    _LABEL_TO_MISTAKE_PROB = dict(_DEFAULT_LABEL_MISTAKE_PROB)
    _SENSITIVITIES = dict(_DEFAULT_SENSITIVITIES)
    _SPECIALIST_ISOTONIC = {}
    _CALIBRATION_META.update(loaded=False, version=None, calibration_size=0)


def default_sensitivity(channel: str) -> float:
    return float(_SENSITIVITIES.get(channel, _DEFAULT_SENSITIVITIES.get(channel, 0.7)))


def isotonic_predict(tool_name: str, x: float) -> float | None:
    """Apply learned isotonic mapping for a specialist; return None if unfit."""
    spec = _SPECIALIST_ISOTONIC.get(tool_name)
    if not spec:
        return None
    xs = spec.get("x") or []
    ys = spec.get("y") or []
    if not xs or not ys or len(xs) != len(ys):
        return None
    # Piecewise-constant: find the largest x_i <= x.
    out = float(ys[0])
    for xi, yi in zip(xs, ys):
        if x >= xi:
            out = float(yi)
        else:
            break
    return out


# ---- stage2 label channel ----

def _label_mistake_prob(label: str | None) -> float:
    if label is None:
        return 0.20
    return _LABEL_TO_MISTAKE_PROB.get(label.strip().lower(), 0.20)


# Auto-load on import (no-op if file absent).
load_calibration()


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


