"""预测流程。"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from src.data.reader import DatasetCSVReader
from src.data.adapter import DataAdapter
from src.data.validator import DataValidator
from src.features.feature_builder import FeatureBuilder
from src.features.feature_registry import FeatureRegistry
from src.features.preprocessor import Preprocessor
from src.models.model_factory import ModelFactory
from src.utils.config import load_features_config
from src.utils.logger import setup_logger


class PredictPipeline:
    """每日预测流程"""

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
        self._init_feature_registry()
        self.feature_builder = FeatureBuilder(
            lag_windows=self.config["features"].get("lag_windows", [1, 7]),
            rolling_windows=self.config["features"].get("rolling_windows", [96, 672]),
            registry=self.registry,
            use_chinese_calendar=self.config.get("calendar", {}).get("use_chinese_calendar", True),
        )
        self.logger = setup_logger("predict_pipeline", self.config["output"].get("log_dir", "logs"))

    def _init_reader(self):
        cfg = self.config["data"]
        return DatasetCSVReader(cfg["dataset_root"], cfg["sources"])

    def _init_feature_registry(self):
        features_cfg_path = self.config.get("features", {}).get("config_path", "config/features.yaml")
        if Path(features_cfg_path).exists():
            features_cfg = load_features_config(features_cfg_path)
            self.registry = FeatureRegistry.from_config(features_cfg)
        else:
            self.registry = None

    def _load_model(self, model_path: Optional[str] = None):
        model_type = self.config["model"].get("type", "lgbm")
        if model_path is None:
            model_path = Path(self.config["output"].get("model_dir", "./models")) / f"{model_type}.pkl"
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        self.logger.info(f"加载模型: {model_path}")
        return ModelFactory.create(model_type, {"component_name": model_type}).load(str(model_path))

    def load_data(self, target_date: str, history_days: int = 30) -> pd.DataFrame:
        """
        加载目标日期的事前数据与历史事后数据。
        history_days 需要覆盖最大滞后窗口（默认 7 天 + 滚动窗口 7 天 = 14 天，留余量 30 天）。
        """
        sources = self.config["data"]["sources"]
        target_dt = pd.to_datetime(target_date)
        start_dt = target_dt - pd.Timedelta(days=history_days)
        start_date = start_dt.strftime("%Y-%m-%d")
        # 结束时间设为次日 00:00，以包含目标日期全天 96 个点
        end_dt = target_dt + pd.Timedelta(days=1)
        end_date = end_dt.strftime("%Y-%m-%d")

        # 读取目标变量历史（包含 target_date，但该日值为 NaN）
        target_df = self.reader.read_target(start_date=start_date, end_date=end_date)

        feature_tables = {}
        for name in ["sys_load_pred", "wind_power_pred", "solar_power_pred",
                     "power_import_plan", "coal_gen_plan", "gas_gen_plan",
                     "storage_plan", "reserve_pos", "reserve_neg", "renewable_capacity"]:
            if name in sources:
                feature_tables[name] = self.reader.read_table(name, start_date=start_date, end_date=end_date)

        actual_tables = {}
        for name in ["actual_sys_load", "actual_wind", "actual_solar"]:
            if name in sources:
                actual_tables[name] = self.reader.read_table(name, start_date=start_date, end_date=end_date)

        # 合并实际风光
        if "actual_wind" in actual_tables and "actual_solar" in actual_tables:
            combined = actual_tables["actual_wind"][["timestamp", "value"]].copy()
            combined = combined.merge(
                actual_tables["actual_solar"][["timestamp", "value"]].rename(columns={"value": "solar"}),
                on="timestamp", how="outer"
            )
            combined["value"] = combined["value"].fillna(0) + combined["solar"].fillna(0)
            combined["field_name"] = "actual_wind_solar"
            actual_tables["actual_wind_solar"] = combined[["timestamp", "value", "field_name"]]
            del actual_tables["actual_wind"]
            del actual_tables["actual_solar"]

        all_tables = {**feature_tables}
        if "actual_sys_load" in actual_tables:
            all_tables["actual_sys_load"] = actual_tables["actual_sys_load"]
        if "actual_wind_solar" in actual_tables:
            all_tables["actual_wind_solar"] = actual_tables["actual_wind_solar"]

        # 预测时不按 target 对齐，保留 target_date 行
        df = self.adapter.build_panel(all_tables, target_df, align_to_target=False)
        df = df.rename(columns={"target": "target"})
        return df

    def validate_features(self, pred_df: pd.DataFrame, expected_cols: list[str]) -> None:
        """校验预测特征维度与训练时一致。"""
        missing = [c for c in expected_cols if c not in pred_df.columns]
        if missing:
            raise ValueError(f"预测时缺少特征: {missing}")

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建特征"""
        df = self.preprocessor.handle_missing(df, solar_wind_cols=["wind_power_pred", "solar_power_pred"])
        features = self.feature_builder.build_all(df)
        features["target"] = df["target"].values
        return features

    def run(self, target_date: str, model_path: Optional[str] = None) -> pd.DataFrame:
        """预测目标日期的 96 点电价。"""
        model = self._load_model(model_path)
        df = self.load_data(target_date)
        features = self.build_features(df)

        # 筛选目标日期行
        target_dt = pd.to_datetime(target_date)
        mask = pd.to_datetime(features["timestamp"]).dt.date == target_dt.date()
        pred_df = features.loc[mask].copy()

        if pred_df.empty:
            raise ValueError(f"目标日期 {target_date} 无可用数据")

        feature_cols = [c for c in pred_df.columns if c not in ["timestamp", "target"]]
        self.validate_features(pred_df, feature_cols)
        X_pred = pred_df[feature_cols].values
        y_pred = model.predict(X_pred)

        result = pd.DataFrame({
            "timestamp": pd.to_datetime(pred_df["timestamp"]),
            "predicted_price": y_pred,
        })

        # 保存预测结果
        pred_dir = Path(self.config["output"].get("prediction_dir", "./data/predictions"))
        pred_dir.mkdir(parents=True, exist_ok=True)
        save_path = pred_dir / f"{target_date}_predictions.csv"
        result.to_csv(save_path, index=False)
        self.logger.info(f"预测结果已保存至 {save_path}")
        return result
