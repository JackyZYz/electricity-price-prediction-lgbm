"""数据 Schema 定义。"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import numpy as np


class Granularity(Enum):
    """数据时间粒度"""
    MIN15 = 15    # 15分钟/点，每天96点
    HOUR1 = 60    # 1小时/点，每天24点


class DataSource(Enum):
    """数据来源类型"""
    PRE_DAY_AHEAD = "pre_day_ahead"    # 事前预测/计划类
    POST_ACTUAL = "post_actual"        # 事后实际值
    POST_CLEARING = "post_clearing"    # 事后出清结果


@dataclass
class TimeSeriesRecord:
    """单条时序记录的基础结构"""
    timestamp: datetime          # 时间戳
    point_index: int             # 时段序号 (0-95 或 0-23)
    value: float                 # 数值
    source: DataSource           # 数据来源
    table_name: str              # 来源表名
    field_name: str              # 字段名


@dataclass
class FeatureMatrix:
    """特征矩阵 — 所有上下游模块的标准数据结构"""
    data: np.ndarray             # shape: (n_samples, n_features)
    timestamps: np.ndarray       # shape: (n_samples,) 每个样本的时间戳
    feature_names: list          # 特征名列表，长度 = n_features
    target: Optional[np.ndarray] # shape: (n_samples,) 训练时有，预测时为None
    target_name: str = "user_price"


@dataclass
class PredictionOutput:
    """预测结果标准结构"""
    timestamps: np.ndarray       # 预测时段
    predicted: np.ndarray        # 预测电价
    actual: Optional[np.ndarray] # 实际值（评估时有）
    low_freq: Optional[np.ndarray]   # 低频分量预测值
    high_freq: Optional[np.ndarray]  # 高频分量预测值
    confidence_lower: Optional[np.ndarray]  # 置信下界
    confidence_upper: Optional[np.ndarray]  # 置信上界
