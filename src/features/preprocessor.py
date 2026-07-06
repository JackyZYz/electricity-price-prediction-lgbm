"""预处理器：缺失值、异常值、归一化。"""
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Preprocessor:
    """数据预处理器"""

    def __init__(self, config: dict):
        self.missing_strategy = config.get("missing_strategy", "ffill")
        self.outlier_method = config.get("outlier_method", "iqr")
        self.outlier_threshold = config.get("outlier_threshold", 3.0)
        self.normalize_method = config.get("normalize_method", "standard")
        self.scaler = None
        self.target_scaler = None

    def handle_missing(self, df: pd.DataFrame, solar_wind_cols: Optional[list] = None) -> pd.DataFrame:
        """缺失值处理"""
        df = df.copy()
        # 风光夜间空值填 0
        if solar_wind_cols:
            for col in solar_wind_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
        if self.missing_strategy == "ffill":
            df = df.ffill().bfill()
        elif self.missing_strategy == "linear":
            df = df.interpolate(method="linear").ffill().bfill()
        elif self.missing_strategy == "mean":
            df = df.fillna(df.mean(numeric_only=True))
        return df

    def handle_outliers(self, df: pd.DataFrame, exclude_cols: Optional[list] = None) -> pd.DataFrame:
        """
        异常值处理：对非目标/非电价列使用 IQR clip。
        电价尖峰保留，因为属于真实现货行情。
        """
        df = df.copy()
        exclude = set(exclude_cols or ["target"])
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in exclude:
                continue
            values = df[col].dropna()
            if values.empty:
                continue
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - self.outlier_threshold * iqr
            upper = q3 + self.outlier_threshold * iqr
            df[col] = df[col].clip(lower, upper)
        return df

    def normalize(self, df: pd.DataFrame, feature_cols: list, fit: bool = True) -> pd.DataFrame:
        """归一化"""
        df = df.copy()
        if self.normalize_method == "none":
            return df
        cols = [c for c in feature_cols if c in df.columns]
        if fit:
            if self.normalize_method == "standard":
                self.scaler = StandardScaler()
            elif self.normalize_method == "minmax":
                self.scaler = MinMaxScaler()
            df[cols] = self.scaler.fit_transform(df[cols])
        else:
            if self.scaler is None:
                raise RuntimeError("Scaler 未训练，无法做 transform")
            df[cols] = self.scaler.transform(df[cols])
        return df
