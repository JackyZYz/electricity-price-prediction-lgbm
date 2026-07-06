"""测试流程编排。"""
import pandas as pd
import pytest

from src.data.adapter import DataAdapter
from src.data.reader import MockDataReader
from src.data.validator import DataValidator
from src.features.feature_builder import FeatureBuilder
from src.features.preprocessor import Preprocessor
from src.models.lgbm_model import LGBMModel


def make_mock_tables():
    reader = MockDataReader(n_days=50)
    sources = ["target_price", "sys_load_pred", "wind_power_pred", "solar_power_pred"]
    return {name: reader.read_table(name) for name in sources}


def test_data_adapter_build_panel():
    tables = make_mock_tables()
    adapter = DataAdapter()
    df = adapter.build_panel(
        {k: v for k, v in tables.items() if k != "target_price"},
        tables["target_price"],
    )
    assert "target" in df.columns
    assert "sys_load_pred" in df.columns
    assert df["target"].notna().any()


def test_data_validator():
    reader = MockDataReader(n_days=10)
    df = reader.read_table("target_price")
    validator = DataValidator()
    report = validator.validate("target_price", df)
    assert report.is_passed
    assert report.total_points == len(df)


def test_end_to_end_train():
    reader = MockDataReader(n_days=50)
    tables = {name: reader.read_table(name) for name in ["target_price", "sys_load_pred", "wind_power_pred", "solar_power_pred"]}
    adapter = DataAdapter()
    df = adapter.build_panel(
        {k: v for k, v in tables.items() if k != "target_price"},
        tables["target_price"],
    )
    df = df.rename(columns={"target": "target"})
    preprocessor = Preprocessor({"missing_strategy": "ffill", "normalize_method": "none"})
    df = preprocessor.handle_missing(df)
    builder = FeatureBuilder(lag_windows=[1], rolling_windows=[96])
    features = builder.build_all(df)
    features["target"] = df["target"].values
    features = features.dropna()

    feature_cols = [c for c in features.columns if c not in ["timestamp", "target"]]
    split = int(len(features) * 0.8)
    train, valid = features.iloc[:split], features.iloc[split:]
    model = LGBMModel({"lgbm_params": {"num_leaves": 15, "max_depth": 5}, "num_boost_round": 100})
    model.fit(
        train[feature_cols].values, train["target"].values,
        valid[feature_cols].values, valid["target"].values,
        feature_names=feature_cols,
    )
    preds = model.predict(valid[feature_cols].values)
    assert len(preds) == len(valid)
