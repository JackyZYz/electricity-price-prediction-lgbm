"""数据适配器：多表对齐、缺失值处理、宽表合并。"""
from typing import Dict, Optional

import pandas as pd
import numpy as np


class DataAdapter:
    """将多个长表数据合并为按时间戳对齐的宽表。"""

    def __init__(
        self,
        fill_00_with_24: bool = True,
        solar_wind_night_fill: Optional[float] = 0.0,
    ):
        self.fill_00_with_24 = fill_00_with_24
        self.solar_wind_night_fill = solar_wind_night_fill

    def merge_tables(self, tables: Dict[str, pd.DataFrame], target_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        将多个长表按 timestamp 合并为宽表。
        tables: {field_name: DataFrame[timestamp, value, field_name]}
        target_df: 目标变量长表，可选
        """
        merged = None
        for name, df in tables.items():
            df = df[["timestamp", "value"]].copy()
            df.rename(columns={"value": name}, inplace=True)
            if merged is None:
                merged = df
            else:
                merged = pd.merge(merged, df, on="timestamp", how="outer")
        if target_df is not None:
            target = target_df[["timestamp", "value"]].copy()
            target.rename(columns={"value": "target"}, inplace=True)
            merged = pd.merge(merged, target, on="timestamp", how="outer")
        merged = merged.sort_values("timestamp").reset_index(drop=True)
        return merged

    def handle_missing(self, df: pd.DataFrame, strategy: str = "ffill", solar_wind_cols: Optional[list] = None) -> pd.DataFrame:
        """缺失值处理"""
        df = df.copy()
        # 先对 00:00 回填已经在 reader 中完成
        # 风光夜间空值按物理意义填 0
        if solar_wind_cols and self.solar_wind_night_fill is not None:
            for col in solar_wind_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(self.solar_wind_night_fill)
        # 按策略填充
        if strategy == "ffill":
            df = df.ffill().bfill()
        elif strategy == "linear":
            df = df.interpolate(method="linear").ffill().bfill()
        elif strategy == "mean":
            df = df.fillna(df.mean())
        return df

    def align_to_target(self, df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
        """以目标变量存在的时间戳为基准，丢弃目标缺失的样本"""
        return df[df[target_col].notna()].copy()

    def build_panel(self, tables: Dict[str, pd.DataFrame], target_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """完整适配流程：合并 → 缺失处理 → 对齐目标"""
        solar_wind_cols = kwargs.get("solar_wind_cols", ["wind_power_pred", "solar_power_pred"])
        missing_strategy = kwargs.get("missing_strategy", "ffill")
        df = self.merge_tables(tables, target_df)
        df = self.handle_missing(df, strategy=missing_strategy, solar_wind_cols=solar_wind_cols)
        df = self.align_to_target(df)
        return df
