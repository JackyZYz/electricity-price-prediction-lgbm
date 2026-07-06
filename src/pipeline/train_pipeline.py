"""训练流程。"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.reader import DatasetCSVReader
from src.data.adapter import DataAdapter
from src.data.validator import DataValidator
from src.features.feature_builder import FeatureBuilder
from src.features.feature_registry import FeatureRegistry
from src.features.feature_store import FeatureStore
from src.features.preprocessor import Preprocessor
from src.models.model_factory import ModelFactory
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.report import ReportGenerator
from src.utils.config import load_features_config
from src.utils.logger import setup_logger


class TrainPipeline:
    """完整训练流程"""

    def __init__(self, config_path: str = "config/default.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.config_path = config_path
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
        self.feature_store = FeatureStore(
            store_dir=self.config["output"].get("feature_store_dir", "./data/features")
        )
        self.metrics = MetricsCalculator()
        self.report_generator = ReportGenerator(
            report_dir=self.config["output"].get("report_dir", "./reports")
        )
        self.logger = setup_logger("train_pipeline", self.config["output"].get("log_dir", "logs"))

    def _init_reader(self):
        cfg = self.config["data"]
        return DatasetCSVReader(cfg["dataset_root"], cfg["sources"])

    def _init_feature_registry(self):
        """加载特征注册中心。"""
        features_cfg_path = self.config.get("features", {}).get("config_path", "config/features.yaml")
        if Path(features_cfg_path).exists():
            features_cfg = load_features_config(features_cfg_path)
            self.registry = FeatureRegistry.from_config(features_cfg)
        else:
            self.registry = None

    def _read_tables(self):
        """读取目标变量与特征表。"""
        sources = self.config["data"]["sources"]
        target_df = self.reader.read_target()

        feature_tables = {}
        for name in ["sys_load_pred", "wind_power_pred", "solar_power_pred",
                     "power_import_plan", "coal_gen_plan", "gas_gen_plan",
                     "storage_plan", "reserve_pos", "reserve_neg", "renewable_capacity"]:
            if name in sources:
                feature_tables[name] = self.reader.read_table(name)

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
            combined["field_name"] = "actual_wind_solar"
            actual_tables["actual_wind_solar"] = combined[["timestamp", "value", "field_name"]]
            del actual_tables["actual_wind"]
            del actual_tables["actual_solar"]

        all_tables = {**feature_tables}
        if "actual_sys_load" in actual_tables:
            all_tables["actual_sys_load"] = actual_tables["actual_sys_load"]
        if "actual_wind_solar" in actual_tables:
            all_tables["actual_wind_solar"] = actual_tables["actual_wind_solar"]

        return target_df, all_tables

    def load_data(self) -> pd.DataFrame:
        """加载、校验并适配数据"""
        target_df, all_tables = self._read_tables()

        # 数据校验
        target_report = self.validator.validate("target_price", target_df)
        self.logger.info(f"目标变量校验: {target_report}")
        if not target_report.is_passed:
            self.logger.warning(f"目标变量未通过校验: {target_report.warnings}")

        for name, df in all_tables.items():
            report = self.validator.validate(name, df)
            if not report.is_passed:
                self.logger.warning(f"{name} 未通过校验: {report.warnings}")

        df = self.adapter.build_panel(all_tables, target_df)
        df = df.rename(columns={"target": "target"})
        return df

    def validate_data(self, df: pd.DataFrame) -> None:
        """基础校验：检查目标变量缺失和核心特征覆盖。"""
        if df["target"].isna().all():
            raise ValueError("目标变量全部缺失")
        missing_features = [c for c in self.feature_builder.DIRECT_FEATURES if c not in df.columns]
        if missing_features:
            self.logger.warning(f"缺少以下直接特征: {missing_features}")

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
        if test_size >= len(df):
            raise ValueError(f"测试集大小 {test_size} 超过总样本数 {len(df)}")
        train_df = df.iloc[:-test_size].copy()
        test_df = df.iloc[-test_size:].copy()
        return train_df, test_df

    def _build_model_config(self) -> dict:
        """从全局配置中提取模型相关配置。"""
        model_cfg = self.config["model"]
        train_cfg = model_cfg.get("train", {})
        return {
            "component_name": model_cfg.get("type", "lgbm"),
            "wavelet": model_cfg.get("wavelet", "db4"),
            "decompose_level": model_cfg.get("decompose_level", 2),
            "lgbm_params": model_cfg.get("lgbm_params", {}),
            "xgb_params": model_cfg.get("xgb_params", {}),
            "arima_params": model_cfg.get("arima_params", {}),
            "early_stopping_rounds": train_cfg.get("early_stopping_rounds", 50),
            "num_boost_round": train_cfg.get("num_boost_round", 2000),
        }

    def train(self, df: pd.DataFrame, save_report: bool = True) -> dict:
        """训练模型"""
        self.validate_data(df)
        features = self.build_features(df)
        features = features.dropna()
        self.logger.info(f"特征矩阵形状: {features.shape}")

        # 保存特征矩阵
        feature_cols = [c for c in features.columns if c not in ["timestamp", "target"]]
        version = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        self.feature_store.save(
            features, feature_names=feature_cols, target_col="target", version=version,
            metadata={"config": self.config_path}
        )
        self.logger.info(f"特征矩阵已保存至 feature store, version={version}")

        test_days = self.config["model"].get("train", {}).get("test_days", 30)
        train_df, test_df = self.time_series_split(features, test_days)

        X_train = train_df[feature_cols].values
        y_train = train_df["target"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["target"].values

        model_type = self.config["model"].get("type", "lgbm")
        model_config = self._build_model_config()
        model = ModelFactory.create(model_type, model_config)

        self.logger.info(f"开始训练模型: {model_type}")
        fit_result = model.fit(X_train, y_train, X_test, y_test, feature_names=feature_cols)

        y_pred = model.predict(X_test)
        periods = test_df["period_type"].values if "period_type" in test_df else None
        metrics = self.metrics.compute_all(y_test, y_pred, periods=periods)

        # 保存模型
        output_dir = Path(self.config["output"].get("model_dir", "./models"))
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{model_type}.pkl"
        model.save(str(model_path))
        self.logger.info(f"模型已保存至 {model_path}")

        # 特征重要性
        importance = model.get_feature_importance()

        # 生成报告
        if save_report and len(y_test) > 0:
            test_date = pd.to_datetime(test_df["timestamp"].iloc[-1]).strftime("%Y-%m-%d")
            report = self.report_generator.generate_daily(
                date=test_date,
                y_true=y_test,
                y_pred=y_pred,
                timestamps=test_df["timestamp"].values,
                importance=importance,
                save_plot=True,
            )
            self.logger.info(f"训练报告已生成: {report.get('plot_path')}")

        return {
            "model": model,
            "model_type": model_type,
            "model_path": str(model_path),
            "feature_cols": feature_cols,
            "fit_result": fit_result,
            "metrics": metrics,
            "importance": importance,
            "y_test": y_test,
            "y_pred": y_pred,
            "test_timestamps": test_df["timestamp"].values,
            "feature_version": version,
        }

    def run(self) -> dict:
        df = self.load_data()
        return self.train(df)
