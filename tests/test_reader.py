"""测试数据读取器。"""
import pandas as pd
import pytest

from src.data.reader import MockDataReader


def test_mock_reader_read_table():
    reader = MockDataReader(n_days=10)
    df = reader.read_table("target_price")
    assert len(df) == 10 * 96
    assert set(df.columns) == {"timestamp", "value", "field_name"}
    assert df["field_name"].iloc[0] == "target_price"


def test_mock_reader_get_date_range():
    reader = MockDataReader(n_days=10)
    start, end = reader.get_date_range("target_price")
    assert start == pd.to_datetime("2026-01-01")
    assert end == pd.to_datetime("2026-01-01") + pd.Timedelta(days=10)


def test_mock_reader_date_filter():
    reader = MockDataReader(n_days=10)
    df = reader.read_table("target_price", start_date="2026-01-02", end_date="2026-01-03")
    assert len(df) == 2 * 96
