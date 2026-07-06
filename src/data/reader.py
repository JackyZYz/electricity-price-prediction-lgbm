"""数据读取器：支持 Dataset/ 目录 CSV 宽表读取。"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd


class BaseDataReader(ABC):
    """数据读取器抽象基类"""

    @abstractmethod
    def read_table(self, table_name: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """读取单张数据表，返回长表 DataFrame[timestamp, value, field_name]"""
        pass

    @abstractmethod
    def list_available_tables(self) -> list:
        """列出所有可用表"""
        pass

    @abstractmethod
    def get_date_range(self, table_name: str) -> tuple:
        """获取某张表的数据日期范围"""
        pass


class DatasetCSVReader(BaseDataReader):
    """Dataset/ 目录 CSV 读取器：处理宽表格式（96 时点）。"""

    META_COLS = [
        "ID", "父ID", "数据类型", "数据地区", "数据所属菜单",
        "数据来源", "数据描述", "日期", "更新时间",
    ]

    def __init__(self, dataset_root: str, data_sources: dict):
        self.dataset_root = Path(dataset_root)
        self.data_sources = data_sources

    def read_table(
        self,
        table_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fill_00_with_24: bool = True,
    ) -> pd.DataFrame:
        """
        读取单张宽表并转换为长表。
        返回 DataFrame 列：timestamp, value, field_name
        """
        if table_name not in self.data_sources:
            raise KeyError(f"表 '{table_name}' 未在 data_sources 中配置")
        path = self.dataset_root / self.data_sources[table_name]
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在: {path}")

        df = pd.read_csv(path, parse_dates=["日期"])
        time_cols = [c for c in df.columns if c not in self.META_COLS]

        # 00:00 缺失用 24:00 回填（约定）
        if fill_00_with_24 and "00:00" in df.columns and "24:00" in df.columns:
            df["00:00"] = df["00:00"].fillna(df["24:00"])

        # melt 成长表
        df_long = df.melt(
            id_vars=["日期"], value_vars=time_cols,
            var_name="time_str", value_name="value",
        )
        # 处理 24:00 为次日 00:00
        df_long["base_date"] = df_long["日期"].dt.strftime("%Y-%m-%d")
        next_day_mask = df_long["time_str"] == "24:00"
        df_long.loc[next_day_mask, "time_str"] = "00:00"
        df_long.loc[next_day_mask, "base_date"] = (
            pd.to_datetime(df_long.loc[next_day_mask, "base_date"]) + pd.Timedelta(days=1)
        ).dt.strftime("%Y-%m-%d")
        df_long["timestamp"] = pd.to_datetime(df_long["base_date"] + " " + df_long["time_str"])
        df_long = df_long[["timestamp", "value"]].sort_values("timestamp").reset_index(drop=True)
        df_long["field_name"] = table_name

        if start_date is not None:
            df_long = df_long[df_long["timestamp"] >= pd.to_datetime(start_date)]
        if end_date is not None:
            df_long = df_long[df_long["timestamp"] <= pd.to_datetime(end_date)]
        return df_long

    def read_target(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """读取目标变量（统一结算点电价最终结果）。"""
        return self.read_table("target_price", start_date, end_date)

    def list_available_tables(self) -> list:
        return list(self.data_sources.keys())

    def get_date_range(self, table_name: str) -> tuple:
        df = self.read_table(table_name)
        return df["timestamp"].min(), df["timestamp"].max()


class MockDataReader(BaseDataReader):
    """Mock 读取器：用于无数据环境或单元测试。"""

    def __init__(self, n_days: int = 200, freq: str = "15min", seed: int = 42):
        self.n_days = n_days
        self.freq = freq
        self.seed = seed

    def read_table(self, table_name: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        import numpy as np
        rng = np.random.default_rng(self.seed)
        timestamps = pd.date_range("2026-01-01", periods=self.n_days * 96, freq=self.freq)
        values = rng.normal(300, 50, size=len(timestamps))
        df = pd.DataFrame({"timestamp": timestamps, "value": values, "field_name": table_name})
        if start_date:
            df = df[df["timestamp"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["timestamp"] <= pd.to_datetime(end_date)]
        return df

    def list_available_tables(self) -> list:
        return ["target_price", "sys_load_pred", "wind_power_pred", "solar_power_pred"]

    def get_date_range(self, table_name: str) -> tuple:
        return pd.to_datetime("2026-01-01"), pd.to_datetime("2026-01-01") + pd.Timedelta(days=self.n_days)
