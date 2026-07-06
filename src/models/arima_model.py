"""ARIMA 基准模型实现。"""
from typing import Optional

import numpy as np
import pandas as pd

from src.models.base import BaseModel


class ARIMAModel(BaseModel):
    """简单 ARIMA 时序基准模型。"""

    def __init__(self, config: dict):
        self.config = config
        self.component_name = config.get("component_name", "arima")
        self.model = None
        self.feature_names = None
        self.order = config.get("order", (2, 1, 2))
        self.seasonal_order = config.get("seasonal_order", (1, 1, 1, 96))

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None,
    ) -> dict:
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            raise ImportError("statsmodels is not installed. Run: pip install statsmodels")

        self.feature_names = feature_names or []
        # ARIMA 仅依赖目标历史序列，忽略外生特征
        self.model = SARIMAX(
            y_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        return {
            "aic": self.model.aic,
            "bic": self.model.bic,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ARIMA model not trained")
        steps = len(X)
        return self.model.forecast(steps=steps)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({
            "model": self.model,
            "config": self.config,
            "feature_names": self.feature_names,
        }, path)

    @classmethod
    def load(cls, path: str) -> "ARIMAModel":
        import joblib
        data = joblib.load(path)
        model = cls(data["config"])
        model.model = data["model"]
        model.feature_names = data["feature_names"]
        return model

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        return None
