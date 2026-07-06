"""测试评估指标。"""
import numpy as np
import pytest

from src.evaluation.metrics import MetricsCalculator


def test_mape_zero_protection():
    calc = MetricsCalculator()
    y_true = np.array([0.0, 100.0, 200.0])
    y_pred = np.array([10.0, 110.0, 190.0])
    mape = calc.mape(y_true, y_pred)
    assert mape >= 0


def test_direction_accuracy():
    calc = MetricsCalculator()
    y_true = np.array([100.0, 110.0, 105.0, 120.0])
    y_pred = np.array([100.0, 112.0, 103.0, 121.0])
    acc = calc.direction_accuracy(y_true, y_pred)
    assert 0 <= acc <= 100


def test_spike_metrics():
    calc = MetricsCalculator()
    y_true = np.array([100.0] * 100 + [500.0])
    y_pred = np.array([100.0] * 100 + [500.0])
    metrics = calc.spike_capture_rate(y_true, y_pred, threshold_sigma=2.0)
    assert metrics["spike_capture_rate"] == 1.0
    assert metrics["spike_miss_rate"] == 0.0
