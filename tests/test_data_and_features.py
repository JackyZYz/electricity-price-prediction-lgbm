"""单元测试：数据读取与特征工程。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.data.reader import DatasetCSVReader
from src.data.validator import DataValidator
from src.data.adapter import DataAdapter
from src.features.feature_builder import FeatureBuilder
from src.features.preprocessor import Preprocessor


def test_reader():
    root = Path(__file__).parent.parent / "Dataset"
    sources = {
        "target_price": "用户侧日前出清发布/用户侧日前出清发布_统一结算点电价最终结果.csv",
        "sys_load_pred": "短期系统负荷预测/短期系统负荷预测信息_出清发布电力.csv",
    }
    reader = DatasetCSVReader(root, sources)
    df = reader.read_target()
    assert "timestamp" in df.columns
    assert "value" in df.columns
    assert len(df) > 0
    print("test_reader passed")


def test_validator():
    root = Path(__file__).parent.parent / "Dataset"
    sources = {"target_price": "用户侧日前出清发布/用户侧日前出清发布_统一结算点电价最终结果.csv"}
    reader = DatasetCSVReader(root, sources)
    df = reader.read_target()
    validator = DataValidator()
    report = validator.validate("target_price", df)
    assert report.total_points == len(df)
    print("test_validator passed")


def test_adapter_and_features():
    root = Path(__file__).parent.parent / "Dataset"
    sources = {
        "target_price": "用户侧日前出清发布/用户侧日前出清发布_统一结算点电价最终结果.csv",
        "sys_load_pred": "短期系统负荷预测/短期系统负荷预测信息_出清发布电力.csv",
        "wind_power_pred": "统调风电功率预测/统调风电功率预测_风力_地区汇总_出清发布电力.csv",
        "solar_power_pred": "统调光电功率预测/统调光电功率预测_太阳能_地区汇总_出清发布电力.csv",
    }
    reader = DatasetCSVReader(root, sources)
    target = reader.read_target()
    tables = {name: reader.read_table(name) for name in sources if name != "target_price"}
    adapter = DataAdapter()
    df = adapter.build_panel(tables, target)
    preprocessor = Preprocessor({"missing_strategy": "ffill", "normalize_method": "none"})
    df = preprocessor.handle_missing(df, solar_wind_cols=["wind_power_pred", "solar_power_pred"])
    builder = FeatureBuilder(lag_windows=[1], rolling_windows=[96])
    features = builder.build_all(df)
    assert len(features) > 0
    assert "hour" in features.columns
    assert "target" in df.columns
    print("test_adapter_and_features passed")


if __name__ == "__main__":
    test_reader()
    test_validator()
    test_adapter_and_features()
    print("All tests passed.")
