"""PRISM: Posterior-driven Routing with Information-theoretic Stopping.

A unified Bayesian framework for step-level error detection in mathematical
reasoning traces. Replaces the legacy stage2_review + stage2_specialist_review
+ deterministic_fallback + learned_router stack with a single posterior over
the first-mistake index tau, plus three principled decision rules:

- Routing  = greedy submodular Expected Information Gain (EIG)
- Stopping = sequential conformal threshold
- Fusion   = vectorized Bayesian update over per-channel likelihoods

See PRISM_ALGORITHM.md for the full theoretical exposition.
"""
from __future__ import annotations

__all__ = [
    "PRISM_VERSION",
    "Posterior",
    "Likelihood",
    "make_stage2_likelihood",
    "make_specialist_likelihood",
    "expected_information_gain",
    "greedy_select",
    "ConformalSchedule",
    "calibrate_conformal",
    "prism_infer",
    "PrismResult",
    "PrismEvidence",
    "PrismPredictor",
]

PRISM_VERSION = "1.0.0"


def __getattr__(name: str):  # pragma: no cover - lazy import shim
    if name == "Posterior":
        from .posterior import Posterior
        return Posterior
    if name == "Likelihood":
        from .likelihoods import Likelihood
        return Likelihood
    if name == "make_stage2_likelihood":
        from .likelihoods import make_stage2_likelihood
        return make_stage2_likelihood
    if name == "make_specialist_likelihood":
        from .likelihoods import make_specialist_likelihood
        return make_specialist_likelihood
    if name == "expected_information_gain":
        from .eig import expected_information_gain
        return expected_information_gain
    if name == "greedy_select":
        from .eig import greedy_select
        return greedy_select
    if name == "ConformalSchedule":
        from .conformal import ConformalSchedule
        return ConformalSchedule
    if name == "calibrate_conformal":
        from .conformal import calibrate_conformal
        return calibrate_conformal
    if name == "prism_infer":
        from .infer import prism_infer
        return prism_infer
    if name == "PrismResult":
        from .infer import PrismResult
        return PrismResult
    if name == "PrismEvidence":
        from .infer import PrismEvidence
        return PrismEvidence
    if name == "PrismPredictor":
        from .predictor import PrismPredictor
        return PrismPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
