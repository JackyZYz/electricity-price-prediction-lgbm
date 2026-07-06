"""测试特征工程。"""
import numpy as np
import pandas as pd
import pytest

from src.features.feature_builder import FeatureBuilder
from src.features.feature_registry import FeatureRegistry, FeatureMeta
from src.utils.time_utils import TimeUtils


def make_test_df(n_days=10):
    timestamps = pd.date_range("2026-01-01", periods=n_days * 96, freq="15min")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "target": np.random.randn(len(timestamps)),
        "sys_load_pred": np.random.randn(len(timestamps)) + 1000,
        "wind_power_pred": np.random.randn(len(timestamps)).clip(-5, 200),
        "solar_power_pred": np.random.randn(len(timestamps)).clip(-5, 200),
        "power_import_plan": np.random.randn(len(timestamps)) + 300,
        "coal_gen_plan": np.random.randn(len(timestamps)) + 500,
        "gas_gen_plan": np.random.randn(len(timestamps)) + 100,
        "storage_plan": np.random.randn(len(timestamps)),
        "reserve_pos": np.random.randn(len(timestamps)) + 50,
        "reserve_neg": np.random.randn(len(timestamps)) + 50,
        "renewable_capacity": np.random.randn(len(timestamps)) + 200,
        "actual_sys_load": np.random.randn(len(timestamps)) + 1000,
        "actual_wind_solar": np.random.randn(len(timestamps)) + 200,
    })
    return df


def test_feature_builder_basic():
    df = make_test_df(20)
    builder = FeatureBuilder()
    features = builder.build_all(df)
    assert "timestamp" in features.columns
    assert "hour" in features.columns
    assert "period_type" in features.columns
    assert "user_price_lag_1d" in features.columns
    assert "bidding_space" in features.columns


def test_feature_registry():
    registry = FeatureRegistry()
    registry.register(FeatureMeta(name="sys_load_pred", category="direct", status="OK"))
    registry.register(FeatureMeta(name="bidding_space", category="constructed", status="OK"))
    assert registry.get_available_features() == ["sys_load_pred", "bidding_space"]


def test_time_utils_period():
    ts = pd.to_datetime(["2026-01-01 02:00", "2026-01-01 09:00", "2026-01-01 14:00"])
    periods = TimeUtils.classify_period(ts)
    assert periods[0] == 0  # valley
    assert periods[1] == 2  # peak
    assert periods[2] == 1  # flat
