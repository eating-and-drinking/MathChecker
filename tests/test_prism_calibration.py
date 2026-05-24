from __future__ import annotations

import json
from pathlib import Path

import pytest

from mathchecker.prism import likelihoods as L


@pytest.fixture
def reset_calibration():
    L.reset_calibration_to_defaults()
    yield
    L.reset_calibration_to_defaults()


def test_load_calibration_missing_file_is_safe(reset_calibration, tmp_path):
    info = L.load_calibration(tmp_path / "does_not_exist.json")
    assert info["loaded"] is False


def test_load_calibration_overrides_label_probs(reset_calibration, tmp_path):
    target = tmp_path / "cal.json"
    target.write_text(json.dumps({
        "version": 1,
        "label_mistake_prob": {
            "correct-and-aligned": 0.01,
            "reasonable-but-incomplete": 0.05,
            "nothing-extracted": 0.30,
            "contradiction-found": 0.95,
        },
        "sensitivities": {"stage1": 0.40, "stage2": 0.95, "specialist": 0.70},
        "specialist_isotonic": {
            "alternative_route_verifier_tool": {"x": [0.0, 0.5, 0.9], "y": [0.05, 0.35, 0.85]},
        },
        "calibration_size": 42,
    }), encoding="utf-8")
    info = L.load_calibration(target)
    assert info["loaded"] is True
    assert info["calibration_size"] == 42
    # Label override is in effect.
    assert L._label_mistake_prob("contradiction-found") == 0.95
    # Sensitivities override is in effect.
    assert L.default_sensitivity("stage1") == 0.40
    assert L.default_sensitivity("stage2") == 0.95
    # Isotonic prediction monotone.
    assert L.isotonic_predict("alternative_route_verifier_tool", 0.1) == 0.05
    assert L.isotonic_predict("alternative_route_verifier_tool", 0.6) == 0.35
    assert L.isotonic_predict("alternative_route_verifier_tool", 1.0) == 0.85
    # Unknown specialist returns None.
    assert L.isotonic_predict("unknown_tool", 0.5) is None


def test_calibration_affects_stage2_likelihood(reset_calibration, tmp_path):
    """After loading a calibration that shifts contradiction-found's mistake-prob
    upward, the stage2 likelihood at the corresponding step should peak harder."""
    base_lik = L.make_stage2_likelihood(
        step_index=2,
        num_steps=4,
        principle_labels=["contradiction-found", "correct-and-aligned", "correct-and-aligned"],
    )

    # Load calibration that bumps contradiction-found to 0.99.
    target = tmp_path / "cal.json"
    target.write_text(json.dumps({
        "version": 1,
        "label_mistake_prob": {"contradiction-found": 0.99},
    }), encoding="utf-8")
    L.load_calibration(target)

    cal_lik = L.make_stage2_likelihood(
        step_index=2,
        num_steps=4,
        principle_labels=["contradiction-found", "correct-and-aligned", "correct-and-aligned"],
    )
    # The likelihood at tau = step_index should be at least as high under calibration
    # (because q went up).
    assert cal_lik.values[2] >= base_lik.values[2]


def test_calibration_script_label_fit_basic(reset_calibration, tmp_path):
    """Smoke-test the calibration script's label-fit on a tiny synthetic record."""
    from scripts.calibrate_likelihoods import _fit_label_mistake_prob  # type: ignore

    records = [
        {
            "completed": True,
            "gold_first_mistake_index": 1,
            "steps": [
                {
                    "step_index": 0,
                    "stage2_parse": {"principle_labels": {
                        "mathematical_concepts": "correct-and-aligned",
                        "key_analyses": "correct-and-aligned",
                        "calculations": "correct-and-aligned",
                    }},
                },
                {
                    "step_index": 1,
                    "stage2_parse": {"principle_labels": {
                        "mathematical_concepts": "contradiction-found",
                        "key_analyses": "contradiction-found",
                        "calculations": "contradiction-found",
                    }},
                },
            ],
        },
        {
            "completed": True,
            "gold_first_mistake_index": None,  # no error
            "steps": [
                {
                    "step_index": 0,
                    "stage2_parse": {"principle_labels": {
                        "mathematical_concepts": "correct-and-aligned",
                        "key_analyses": "correct-and-aligned",
                        "calculations": "correct-and-aligned",
                    }},
                },
            ],
        },
    ]
    probs = _fit_label_mistake_prob(records)
    # "correct-and-aligned" appeared 4 times, never on a first-mistake step.
    # With Laplace smoothing (k+1)/(n+5): (0+1)/(4+5) = 0.111.
    assert probs["correct-and-aligned"] < 0.15
    # "contradiction-found" appeared 3 times, ALL on the first-mistake step.
    # (3+1)/(3+5) = 0.5.
    assert probs["contradiction-found"] > 0.45
