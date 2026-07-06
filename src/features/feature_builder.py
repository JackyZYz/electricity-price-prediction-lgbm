"""特征工程模块。"""
import numpy as np
import pandas as pd


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

    def __init__(self, lag_windows=None, rolling_windows=None, use_price_lags=True, use_actual_lags=True, use_date_features=True):
        self.lag_windows = lag_windows or [1, 7]
        self.rolling_windows = rolling_windows or [96, 672]
        self.use_price_lags = use_price_lags
        self.use_actual_lags = use_actual_lags
        self.use_date_features = use_date_features

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

    def _build_direct_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取可直接使用的原始特征"""
        cols = [c for c in self.DIRECT_FEATURES if c in df.columns]
        return df[cols].copy()

    def _build_constructed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算构造特征"""
        constructed = pd.DataFrame(index=df.index)
        # 仅当所需列存在时才计算
        required = ["sys_load_pred", "power_import_plan", "coal_gen_plan",
                    "gas_gen_plan", "storage_plan", "wind_power_pred", "solar_power_pred"]
        if all(c in df.columns for c in required):
            constructed["bidding_space"] = (
                df["sys_load_pred"] + df["power_import_plan"]
                - df["coal_gen_plan"] - df["gas_gen_plan"] - df["storage_plan"]
                - df["wind_power_pred"] - df["solar_power_pred"]
            )
            constructed["net_load"] = (
                df["sys_load_pred"] - df["wind_power_pred"] - df["solar_power_pred"]
            )
            constructed["import_ratio"] = df["power_import_plan"] / df["sys_load_pred"]
            constructed["renewable_penetration"] = (
                df["wind_power_pred"] + df["solar_power_pred"]
            ) / df["sys_load_pred"]
            constructed["thermal_ratio"] = (
                df["coal_gen_plan"] + df["gas_gen_plan"]
            ) / df["sys_load_pred"]
        if "reserve_pos" in df.columns and "sys_load_pred" in df.columns:
            constructed["reserve_margin"] = df["reserve_pos"] / df["sys_load_pred"]
        if "actual_sys_load_lag_1d" in df.columns and "sys_load_pred" in df.columns:
            constructed["load_forecast_error"] = df["actual_sys_load_lag_1d"] - df["sys_load_pred"]
        return constructed

    def _build_price_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建电价滞后特征"""
        lags = pd.DataFrame(index=df.index)
        points_per_day = 96
        for d in self.lag_windows:
            lags[f"user_price_lag_{d}d"] = df["target"].shift(d * points_per_day)
        for w in self.rolling_windows:
            lags[f"user_price_ma_{w}"] = df["target"].shift(1).rolling(window=w, min_periods=1).mean()
            lags[f"user_price_std_{w}"] = df["target"].shift(1).rolling(window=w, min_periods=1).std()
        return lags

    def _build_actual_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建实际值滞后特征"""
        lags = pd.DataFrame(index=df.index)
        points_per_day = 96
        if "actual_sys_load" in df.columns:
            lags["actual_sys_load_lag_1d"] = df["actual_sys_load"].shift(points_per_day)
        if "actual_wind_solar" in df.columns:
            lags["actual_wind_solar_lag_1d"] = df["actual_wind_solar"].shift(points_per_day)
        return lags

    def _build_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """日期特征编码"""
        timestamps = pd.to_datetime(df["timestamp"])
        features = pd.DataFrame(index=df.index)
        features["hour"] = timestamps.dt.hour
        features["minute"] = timestamps.dt.minute
        features["weekday"] = timestamps.dt.weekday
        features["month"] = timestamps.dt.month
        features["day_of_year"] = timestamps.dt.dayofyear
        features["is_weekend"] = (timestamps.dt.weekday >= 5).astype(int)
        hour = timestamps.dt.hour
        features["is_peak"] = (((hour >= 8) & (hour < 11)) | ((hour >= 17) & (hour < 21))).astype(int)
        features["hour_sin"] = np.sin(2 * np.pi * timestamps.dt.hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * timestamps.dt.hour / 24)
        # 峰平谷分类
        is_valley = (hour >= 0) & (hour < 6)
        is_peak = features["is_peak"].astype(bool)
        features["period_type"] = np.where(is_valley, 0, np.where(is_peak, 2, 1))
        return features
