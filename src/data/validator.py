"""数据校验器。"""
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .schema import Granularity


@dataclass
class ValidationReport:
    """数据质量校验报告"""
    table_name: str
    total_rows: int
    total_points: int
    missing_rate: float
    outlier_count: int
    time_continuity: dict
    duplicate_dates: int
    granularity: Granularity
    is_passed: bool
    warnings: list = field(default_factory=list)


class DataValidator:
    """数据质量校验器"""

    def __init__(self, expected_points_per_day: int = 96):
        self.expected_points_per_day = expected_points_per_day

    def check_completeness(self, df: pd.DataFrame) -> float:
        """检查缺失率（仅 value 列）"""
        return df["value"].isna().mean()

    def check_outliers(self, df: pd.DataFrame, method: str = "iqr", threshold: float = 3.0) -> int:
        """IQR 异常值检测"""
        values = df["value"].dropna()
        if values.empty:
            return 0
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return int(((values < lower) | (values > upper)).sum())

    def check_time_continuity(self, df: pd.DataFrame) -> dict:
        """检查时间连续性"""
        ts = pd.to_datetime(df["timestamp"]).sort_values()
        if len(ts) < 2:
            return {"missing_periods": 0, "expected_periods": len(ts)}
        freq = pd.Timedelta(minutes=15)
        full_range = pd.date_range(ts.min(), ts.max(), freq=freq)
        missing = len(full_range) - len(ts)
        return {"missing_periods": missing, "expected_periods": len(full_range)}

    def check_duplicate_dates(self, df: pd.DataFrame) -> int:
        """检查日期重复（按日期维度）"""
        dates = pd.to_datetime(df["timestamp"]).dt.date
        return int(dates.duplicated().sum())

    def check_alignment(self, df_target: pd.DataFrame, df_feature: pd.DataFrame) -> dict:
        """检查特征表与目标表的时间戳对齐情况"""
        target_ts = set(pd.to_datetime(df_target["timestamp"]))
        feature_ts = set(pd.to_datetime(df_feature["timestamp"]))
        common = target_ts & feature_ts
        only_in_target = target_ts - feature_ts
        only_in_feature = feature_ts - target_ts
        return {
            "common": len(common),
            "only_in_target": len(only_in_target),
            "only_in_feature": len(only_in_feature),
        }

    def validate(self, table_name: str, df: pd.DataFrame, missing_threshold: float = 0.5) -> ValidationReport:
        """执行全部校验，返回报告"""
        missing_rate = self.check_completeness(df)
        outliers = self.check_outliers(df)
        continuity = self.check_time_continuity(df)
        duplicates = self.check_duplicate_dates(df)
        warnings = []
        if missing_rate > missing_threshold:
            warnings.append(f"缺失率 {missing_rate:.2%} 超过阈值 {missing_threshold:.2%}")
        if continuity["missing_periods"] > 0:
            warnings.append(f"缺失 {continuity['missing_periods']} 个时点")
        if duplicates > 0:
            warnings.append(f"存在 {duplicates} 个重复日期")
        is_passed = len(warnings) == 0
        return ValidationReport(
            table_name=table_name,
            total_rows=len(df) // self.expected_points_per_day if self.expected_points_per_day else len(df),
            total_points=len(df),
            missing_rate=missing_rate,
            outlier_count=outliers,
            time_continuity=continuity,
            duplicate_dates=duplicates,
            granularity=Granularity.MIN15,
            is_passed=is_passed,
            warnings=warnings,
        )
