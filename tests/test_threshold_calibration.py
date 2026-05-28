"""Sprint 2 β2: confidence threshold calibration."""
from __future__ import annotations

import pytest


def test_calibration_returns_documented_keys():
    from tools.calibrate_confidence import calibrate

    out = calibrate([(0.5, True), (0.5, False)])
    assert set(out.keys()) == {"low", "mid", "high", "n_samples"}


def test_calibration_monotonic():
    from tools.calibrate_confidence import calibrate

    samples = [
        (0.10, False), (0.12, False), (0.15, False),
        (0.22, False), (0.25, True), (0.28, False),
        (0.40, True), (0.42, True), (0.45, False),
        (0.60, True), (0.65, True), (0.70, True),
    ]
    t = calibrate(samples)
    assert t["low"] <= t["mid"] <= t["high"]


def test_calibration_uses_precision_targets():
    """Targets: precision-at-confidence >= 0.5 / 0.7 / 0.85."""
    from tools.calibrate_confidence import calibrate, _precision_at

    samples = (
        [(0.1 + i * 0.01, False) for i in range(20)] +
        [(0.30 + i * 0.005, i % 2 == 0) for i in range(20)] +
        [(0.50 + i * 0.005, True) for i in range(40)]
    )
    t = calibrate(samples)
    assert _precision_at(samples, t["low"]) >= 0.45
    assert _precision_at(samples, t["mid"]) >= 0.6
    assert _precision_at(samples, t["high"]) >= 0.8


def test_calibration_falls_back_to_defaults_on_empty_input():
    from tools.calibrate_confidence import calibrate

    t = calibrate([])
    assert t == {"low": 0.20, "mid": 0.35, "high": 0.55, "n_samples": 0}


def test_precision_at_returns_zero_when_no_samples_above():
    from tools.calibrate_confidence import _precision_at

    assert _precision_at([(0.1, True), (0.2, False)], threshold=1.0) == 0.0
