"""模型抽象基类。"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """所有预测模型的抽象基类。"""

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None,
    ) -> dict:
        """训练模型。"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测。"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """保存模型。"""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseModel":
        """加载模型。"""
        pass

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """获取特征重要性（可选）。"""
        return None
