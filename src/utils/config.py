"""工具函数。"""
import yaml
from pathlib import Path


def load_config(path: str = "config/default.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
