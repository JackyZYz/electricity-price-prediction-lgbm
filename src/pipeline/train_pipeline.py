"""训练流程。"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.reader import DatasetCSVReader
from src.data.adapter import DataAdapter
from src.data.validator import DataValidator
from src.features.feature_builder import FeatureBuilder
from src.features.preprocessor import Preprocessor
from src.models.wavelet import WaveletDecomposer
from src.models.lgbm_trainer import LGBMTrainer
from src.evaluation.metrics import MetricsCalculator


class TrainPipeline:
    """完整训练流程"""

    def __init__(self, config_path: str = "config/default.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.reader = self._init_reader()
        self.validator = DataValidator()
        self.adapter = DataAdapter(
            fill_00_with_24=self.config["preprocessing"].get("fill_00_with_24", True),
            solar_wind_night_fill=self.config["preprocessing"].get("solar_wind_night_fill", 0.0),
        )
        self.preprocessor = Preprocessor(self.config["preprocessing"])
        self.feature_builder = FeatureBuilder(
            lag_windows=self.config["features"].get("lag_windows", [1, 7]),
            rolling_windows=self.config["features"].get("rolling_windows", [96, 672]),
        )
        self.wavelet = WaveletDecomposer(
            wavelet=self.config["model"].get("wavelet", "db4"),
            level=self.config["model"].get("decompose_level", 2),
        )
        self.metrics = MetricsCalculator()

    def _init_reader(self):
        cfg = self.config["data"]
        return DatasetCSVReader(cfg["dataset_root"], cfg["sources"])

    def load_data(self) -> pd.DataFrame:
        """加载并适配数据"""
        sources = self.config["data"]["sources"]
        # 读取目标变量
        target_df = self.reader.read_target()
        # 读取特征表
        feature_tables = {}
        for name in ["sys_load_pred", "wind_power_pred", "solar_power_pred",
                     "power_import_plan", "coal_gen_plan", "gas_gen_plan",
                     "storage_plan", "reserve_pos", "reserve_neg", "renewable_capacity"]:
            if name in sources:
                feature_tables[name] = self.reader.read_table(name)
        # 读取实际值（用于滞后特征）
        actual_tables = {}
        for name in ["actual_sys_load", "actual_wind", "actual_solar"]:
            if name in sources:
                actual_tables[name] = self.reader.read_table(name)
        # 合并实际风光
        if "actual_wind" in actual_tables and "actual_solar" in actual_tables:
            combined = actual_tables["actual_wind"][["timestamp", "value"]].copy()
            combined = combined.merge(
                actual_tables["actual_solar"][["timestamp", "value"]].rename(columns={"value": "solar"}),
                on="timestamp", how="outer"
            )
            combined["value"] = combined["value"].fillna(0) + combined["solar"].fillna(0)
            actual_tables["actual_wind_solar"] = combined[["timestamp", "value", "field_name"]]
            del actual_tables["actual_wind"]
            del actual_tables["actual_solar"]
        # 适配
        all_tables = {**feature_tables}
        if "actual_sys_load" in actual_tables:
            all_tables["actual_sys_load"] = actual_tables["actual_sys_load"]
        if "actual_wind_solar" in actual_tables:
            all_tables["actual_wind_solar"] = actual_tables["actual_wind_solar"]
        df = self.adapter.build_panel(all_tables, target_df)
        df = df.rename(columns={"target": "target"})
        return df

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建特征"""
        df = self.preprocessor.handle_missing(df, solar_wind_cols=["wind_power_pred", "solar_power_pred"])
        features = self.feature_builder.build_all(df)
        features["target"] = df["target"].values
        return features

    def time_series_split(self, df: pd.DataFrame, test_days: int = 30):
        """时序分割"""
        points_per_day = 96
        test_size = test_days * points_per_day
        train_df = df.iloc[:-test_size].copy()
        test_df = df.iloc[-test_size:].copy()
        return train_df, test_df

    def train(self, df: pd.DataFrame, test_days: int = 30):
        """训练 WT-LGBM 模型"""
        features = self.build_features(df)
        # 删除包含缺失值的行
        features = features.dropna()
        train_df, test_df = self.time_series_split(features, test_days)

        exclude_cols = ["timestamp", "target"]
        feature_cols = [c for c in train_df.columns if c not in exclude_cols]

        X_train = train_df[feature_cols].values
        y_train = train_df["target"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["target"].values

        # 小波分解
        components = self.wavelet.decompose(y_train)

        results = {}
        # 低频模型
        trainer_low = LGBMTrainer(self.config["model"]["lgbm_params"], "low")
        results["low"] = trainer_low.train(
            X_train, components["low"], X_test, components["low"],
            feature_cols, early_stopping_rounds=50
        )
        # 高频模型
        for key in components:
            if key.startswith("high"):
                trainer = LGBMTrainer(self.config["model"]["lgbm_params"], key)
                results[key] = trainer.train(
                    X_train, components[key], X_test, components[key],
                    feature_cols, early_stopping_rounds=50
                )

        # 简单评估：直接预测测试集（小波分量预测合成需要更复杂的对齐，这里先给出框架）
        pred_low = trainer_low.predict(X_test)
        pred = pred_low  # 简化：仅使用低频预测
        metrics = self.metrics.compute_all(y_test, pred, periods=test_df["period_type"].values if "period_type" in test_df else None)
        return {
            "models": results,
            "metrics": metrics,
            "feature_cols": feature_cols,
        }

    def run(self) -> dict:
        df = self.load_data()
        return self.train(df, self.config["model"]["train"]["test_days"])
