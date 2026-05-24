"""PRISM: Posterior-driven Routing with Information-theoretic Stopping."""
from __future__ import annotations

__all__ = [
    "PRISM_VERSION",
    "Posterior", "Likelihood", "make_stage2_likelihood", "make_specialist_likelihood",
    "expected_information_gain", "greedy_select",
    "ConformalSchedule", "calibrate_conformal",
    "EProcessSchedule", "default_eprocess",
    "TemperatureMixer", "fit_temperatures",
    "CalibrationStep", "CalibrationTrajectory", "calibration_nll",
    "prism_infer", "PrismResult", "PrismEvidence", "PrismPredictor",
]

PRISM_VERSION = "1.1.0"


def __getattr__(name: str):  # pragma: no cover
    _MAP = {
        "Posterior": ("posterior", "Posterior"),
        "Likelihood": ("likelihoods", "Likelihood"),
        "make_stage2_likelihood": ("likelihoods", "make_stage2_likelihood"),
        "make_specialist_likelihood": ("likelihoods", "make_specialist_likelihood"),
        "expected_information_gain": ("eig", "expected_information_gain"),
        "greedy_select": ("eig", "greedy_select"),
        "ConformalSchedule": ("conformal", "ConformalSchedule"),
        "calibrate_conformal": ("conformal", "calibrate_conformal"),
        "EProcessSchedule": ("eprocess", "EProcessSchedule"),
        "default_eprocess": ("eprocess", "default_eprocess"),
        "TemperatureMixer": ("joint_calibration", "TemperatureMixer"),
        "fit_temperatures": ("joint_calibration", "fit_temperatures"),
        "CalibrationStep": ("joint_calibration", "CalibrationStep"),
        "CalibrationTrajectory": ("joint_calibration", "CalibrationTrajectory"),
        "calibration_nll": ("joint_calibration", "calibration_nll"),
        "prism_infer": ("infer", "prism_infer"),
        "PrismResult": ("infer", "PrismResult"),
        "PrismEvidence": ("infer", "PrismEvidence"),
        "PrismPredictor": ("predictor", "PrismPredictor"),
    }
    if name in _MAP:
        mod_name, attr = _MAP[name]
        import importlib
        return getattr(importlib.import_module(f".{mod_name}", __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
