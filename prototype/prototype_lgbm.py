"""
原型脚本：纯 LightGBM 电价预测快速验证。
可直接运行：python prototype/prototype_lgbm.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from src.data.reader import DatasetCSVReader
from src.data.adapter import DataAdapter
from src.features.feature_builder import FeatureBuilder
from src.features.preprocessor import Preprocessor
from src.evaluation.metrics import MetricsCalculator


DATASET_ROOT = Path(__file__).parent.parent / "Dataset"
TEST_DAYS = 30
POINTS_PER_DAY = 96

SOURCES = {
    "target_price": "用户侧日前出清发布/用户侧日前出清发布_统一结算点电价最终结果.csv",
    "sys_load_pred": "短期系统负荷预测/短期系统负荷预测信息_出清发布电力.csv",
    "wind_power_pred": "统调风电功率预测/统调风电功率预测_风力_地区汇总_出清发布电力.csv",
    "solar_power_pred": "统调光电功率预测/统调光电功率预测_太阳能_地区汇总_出清发布电力.csv",
    "power_import_plan": "受电计划/受电计划_华东_出清发布电力.csv",
    "coal_gen_plan": "煤电发电计划/煤电发电计划_地区汇总_终发布电力.csv",
    "gas_gen_plan": "燃机固定出力总值/燃机固定出力总值_地区汇总_出清发布电力.csv",
}


def main():
    reader = DatasetCSVReader(DATASET_ROOT, SOURCES)
    target = reader.read_target()
    tables = {name: reader.read_table(name) for name in SOURCES if name != "target_price"}

    adapter = DataAdapter(fill_00_with_24=True, solar_wind_night_fill=0.0)
    df = adapter.build_panel(tables, target)
    df = df.rename(columns={"target": "target"})

    preprocessor = Preprocessor({"missing_strategy": "ffill", "normalize_method": "none"})
    df = preprocessor.handle_missing(df, solar_wind_cols=["wind_power_pred", "solar_power_pred"])

    builder = FeatureBuilder(lag_windows=[1, 7], rolling_windows=[96, 672])
    features = builder.build_all(df)
    features["target"] = df["target"].values
    features = features.dropna()

    feature_cols = [c for c in features.columns if c not in ["timestamp", "target"]]

    test_size = TEST_DAYS * POINTS_PER_DAY
    train = features.iloc[:-test_size]
    test = features.iloc[-test_size:]

    X_train, y_train = train[feature_cols].values, train["target"].values
    X_test, y_test = test[feature_cols].values, test["target"].values

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 10,
        "min_data_in_leaf": 20,
        "verbosity": -1,
    }
    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    valid_set = lgb.Dataset(X_test, label=y_test, feature_name=feature_cols, reference=train_set)
    model = lgb.train(
        params, train_set, num_boost_round=2000, valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    )
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    metrics = MetricsCalculator().compute_all(y_test, y_pred, periods=test["period_type"].values)
    print("评估指标:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # 保存结果图
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sample = test.iloc[-3*POINTS_PER_DAY:].copy()
    sample["pred"] = y_pred[-3*POINTS_PER_DAY:]
    axes[0, 0].plot(sample["timestamp"], sample["target"], label="Actual")
    axes[0, 0].plot(sample["timestamp"], sample["pred"], label="Predicted", alpha=0.8)
    axes[0, 0].set_title("Prediction vs Actual (Last 3 Days)")
    axes[0, 0].legend()
    axes[0, 1].scatter(y_test, y_pred, alpha=0.3)
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    axes[0, 1].set_title("Scatter Plot")
    imp = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importance(importance_type="gain")})
    imp = imp.sort_values("importance", ascending=True).tail(10)
    axes[1, 0].barh(imp["feature"], imp["importance"])
    axes[1, 0].set_title("Top 10 Feature Importance")
    axes[1, 1].hist(y_test - y_pred, bins=50, edgecolor="k")
    axes[1, 1].set_title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(reports_dir / "prototype_result.png", dpi=150)
    print(f"结果图已保存至 {reports_dir / 'prototype_result.png'}")


if __name__ == "__main__":
    main()
