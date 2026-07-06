"""时间处理工具。"""
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


class TimeUtils:
    """时序数据处理工具类。"""

    POINTS_PER_DAY = 96

    @staticmethod
    def classify_period(timestamps: pd.DatetimeIndex) -> np.ndarray:
        """峰/平/谷时段分类：0→谷, 1→平, 2→峰。"""
        hour = timestamps.hour
        is_peak = ((hour >= 8) & (hour < 11)) | ((hour >= 17) & (hour < 21))
        is_valley = (hour >= 0) & (hour < 6)
        return np.where(is_valley, 0, np.where(is_peak, 2, 1))

    @staticmethod
    def is_holiday(timestamps: pd.DatetimeIndex, use_chinese_calendar: bool = True) -> np.ndarray:
        """判断是否为中国法定节假日。"""
        if use_chinese_calendar:
            try:
                import chinese_calendar
                return np.array([int(chinese_calendar.is_holiday(ts)) for ts in timestamps])
            except ImportError:
                pass
        # fallback：仅周末
        return (timestamps.weekday >= 5).astype(int)

    @staticmethod
    def get_date_features(timestamps, use_chinese_calendar: bool = True) -> pd.DataFrame:
        """生成日期/时间特征。"""
        timestamps = pd.to_datetime(timestamps)
        if not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.DatetimeIndex(timestamps)
        features = pd.DataFrame(index=timestamps)
        features["hour"] = timestamps.hour
        features["minute"] = timestamps.minute
        features["weekday"] = timestamps.weekday
        features["month"] = timestamps.month
        features["day_of_year"] = timestamps.dayofyear
        features["is_weekend"] = (timestamps.weekday >= 5).astype(int)
        features["is_holiday"] = TimeUtils.is_holiday(timestamps, use_chinese_calendar)
        hour = timestamps.hour
        features["is_peak"] = (((hour >= 8) & (hour < 11)) | ((hour >= 17) & (hour < 21))).astype(int)
        features["hour_sin"] = np.sin(2 * np.pi * timestamps.hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * timestamps.hour / 24)
        features["period_type"] = TimeUtils.classify_period(timestamps)
        return features

    @staticmethod
    def align_to_granularity(timestamps: pd.DatetimeIndex, freq: str = "15min") -> pd.DatetimeIndex:
        """将时间戳对齐到指定粒度。"""
        return pd.to_datetime(timestamps).floor(freq)

    @staticmethod
    def get_points_per_day(freq: str = "15min") -> int:
        """根据频率返回每日点数。"""
        mapping = {"15min": 96, "1h": 24, "30min": 48, "5min": 288}
        return mapping.get(freq, 96)

    @staticmethod
    def timestamp_to_point_index(timestamp: datetime, freq: str = "15min") -> int:
        """将时间戳转换为当日时段序号。"""
        start_of_day = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        minutes = int((timestamp - start_of_day).total_seconds() // 60)
        if freq == "15min":
            return minutes // 15
        elif freq == "1h":
            return minutes // 60
        else:
            raise ValueError(f"Unsupported freq: {freq}")
