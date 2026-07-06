"""特征工程模块。"""
from typing import Optional

import numpy as np
import pandas as pd

from src.features.feature_registry import FeatureRegistry
from src.utils.time_utils import TimeUtils


class FeatureBuilder:
    """特征构建器：将原始数据转换为模型可用特征矩阵。"""

    DIRECT_FEATURES = [
        "sys_load_pred",
        "wind_power_pred",
        "solar_power_pred",
        "power_import_plan",
        "coal_gen_plan",
        "gas_gen_plan",
        "storage_plan",
        "reserve_pos",
        "reserve_neg",
        "renewable_capacity",
    ]

    def __init__(
        self,
        lag_windows=None,
        rolling_windows=None,
        use_price_lags=True,
        use_actual_lags=True,
        use_date_features=True,
        registry: Optional[FeatureRegistry] = None,
        use_chinese_calendar: bool = True,
    ):
        self.lag_windows = lag_windows or [1, 7]
        self.rolling_windows = rolling_windows or [96, 672]
        self.use_price_lags = use_price_lags
        self.use_actual_lags = use_actual_lags
        self.use_date_features = use_date_features
        self.registry = registry
        self.use_chinese_calendar = use_chinese_calendar

    def build_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建全部特征，入口方法"""
        features = df[["timestamp"]].copy()
        features = pd.concat([features, self._build_direct_features(df)], axis=1)
        features = pd.concat([features, self._build_constructed_features(df)], axis=1)
        if self.use_price_lags and "target" in df.columns:
            features = pd.concat([features, self._build_price_lags(df)], axis=1)
        if self.use_actual_lags:
            features = pd.concat([features, self._build_actual_lags(df)], axis=1)
        if self.use_date_features:
            features = pd.concat([features, self._build_date_features(df)], axis=1)
        return features

    def _select_direct_features(self) -> list[str]:
        """根据注册中心或默认列表选择直接特征。"""
        if self.registry is None:
            return self.DIRECT_FEATURES
        available = set(self.registry.get_available_features())
        return [c for c in self.DIRECT_FEATURES if c in available]

    def _build_direct_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取可直接使用的原始特征"""
        cols = [c for c in self._select_direct_features() if c in df.columns]
        return df[cols].copy()

    def _build_constructed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算构造特征"""
        constructed = pd.DataFrame(index=df.index)
        available = set(self.registry.get_available_features()) if self.registry else None

        def _enabled(name: str) -> bool:
            return available is None or name in available

        # 仅当所需列存在时才计算
        required = ["sys_load_pred", "power_import_plan", "coal_gen_plan",
                    "gas_gen_plan", "storage_plan", "wind_power_pred", "solar_power_pred"]
        if all(c in df.columns for c in required):
            if _enabled("bidding_space"):
                constructed["bidding_space"] = (
                    df["sys_load_pred"] + df["power_import_plan"]
                    - df["coal_gen_plan"] - df["gas_gen_plan"] - df["storage_plan"]
                    - df["wind_power_pred"] - df["solar_power_pred"]
                )
            if _enabled("net_load"):
                constructed["net_load"] = (
                    df["sys_load_pred"] - df["wind_power_pred"] - df["solar_power_pred"]
                )
            if _enabled("import_ratio"):
                constructed["import_ratio"] = df["power_import_plan"] / df["sys_load_pred"]
            if _enabled("renewable_penetration"):
                constructed["renewable_penetration"] = (
                    df["wind_power_pred"] + df["solar_power_pred"]
                ) / df["sys_load_pred"]
            if _enabled("thermal_ratio"):
                constructed["thermal_ratio"] = (
                    df["coal_gen_plan"] + df["gas_gen_plan"]
                ) / df["sys_load_pred"]
        if "reserve_pos" in df.columns and "sys_load_pred" in df.columns and _enabled("reserve_margin"):
            constructed["reserve_margin"] = df["reserve_pos"] / df["sys_load_pred"]
        if "reserve_neg" in df.columns and "sys_load_pred" in df.columns and _enabled("negative_reserve_margin"):
            constructed["negative_reserve_margin"] = df["reserve_neg"] / df["sys_load_pred"]
        if all(c in df.columns for c in ["reserve_pos", "reserve_neg", "sys_load_pred"]) and _enabled("total_reserve_margin"):
            constructed["total_reserve_margin"] = (df["reserve_pos"] + df["reserve_neg"]) / df["sys_load_pred"]
        if "actual_sys_load_lag_1d" in df.columns and "sys_load_pred" in df.columns and _enabled("load_forecast_error"):
            constructed["load_forecast_error"] = df["actual_sys_load_lag_1d"] - df["sys_load_pred"]

        # 尖峰相关特征
        if all(c in df.columns for c in ["wind_power_pred", "solar_power_pred"]) and _enabled("renewable_volatility"):
            # 新能源出力波动性：当前时段相比上一时段的变化幅度
            renewable = df["wind_power_pred"] + df["solar_power_pred"]
            constructed["renewable_volatility"] = renewable.diff().abs()
        if "sys_load_pred" in df.columns and _enabled("load_ramp"):
            # 负荷爬坡率
            constructed["load_ramp"] = df["sys_load_pred"].diff().abs()
        if all(c in df.columns for c in ["sys_load_pred", "wind_power_pred", "solar_power_pred"]) and _enabled("net_load_ramp"):
            # 净负荷爬坡率
            net_load = df["sys_load_pred"] - df["wind_power_pred"] - df["solar_power_pred"]
            constructed["net_load_ramp"] = net_load.diff().abs()
        if "sys_load_pred" in df.columns and _enabled("load_ratio_to_max"):
            # 负荷率（当前负荷 / 历史最大负荷）
            constructed["load_ratio_to_max"] = df["sys_load_pred"] / df["sys_load_pred"].rolling(window=672, min_periods=1).max()
        return constructed

    def _build_price_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建电价滞后特征"""
        lags = pd.DataFrame(index=df.index)
        points_per_day = TimeUtils.get_points_per_day("15min")
        for d in self.lag_windows:
            lags[f"user_price_lag_{d}d"] = df["target"].shift(d * points_per_day)
        for w in self.rolling_windows:
            lags[f"user_price_ma_{w}"] = df["target"].shift(1).rolling(window=w, min_periods=1).mean()
            lags[f"user_price_std_{w}"] = df["target"].shift(1).rolling(window=w, min_periods=1).std()
            lags[f"user_price_max_{w}"] = df["target"].shift(1).rolling(window=w, min_periods=1).max()
            lags[f"user_price_min_{w}"] = df["target"].shift(1).rolling(window=w, min_periods=1).min()
        return lags

    def _build_actual_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建实际值滞后特征"""
        lags = pd.DataFrame(index=df.index)
        points_per_day = TimeUtils.get_points_per_day("15min")
        if "actual_sys_load" in df.columns:
            lags["actual_sys_load_lag_1d"] = df["actual_sys_load"].shift(points_per_day)
        if "actual_wind_solar" in df.columns:
            lags["actual_wind_solar_lag_1d"] = df["actual_wind_solar"].shift(points_per_day)
        return lags

    def _build_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """日期特征编码"""
        timestamps = pd.to_datetime(df["timestamp"])
        features = TimeUtils.get_date_features(timestamps, self.use_chinese_calendar)
        features.index = df.index
        return features
