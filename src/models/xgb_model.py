"""XGBoost 模型实现（可选基准）。"""
from typing import Optional

import numpy as np
import pandas as pd

from src.models.base import BaseModel


class XGBModel(BaseModel):
    """XGBoost 回归模型（可选基准）。"""

    def __init__(self, config: dict):
        self.config = config
        self.component_name = config.get("component_name", "xgb")
        self.model = None
        self.feature_names = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None,
    ) -> dict:
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost is not installed. Run: pip install xgboost")

        if X_valid is None or y_valid is None:
            split_idx = int(len(X_train) * 0.8)
            X_valid, y_valid = X_train[split_idx:], y_train[split_idx:]
            X_train, y_train = X_train[:split_idx], y_train[:split_idx]

        self.feature_names = feature_names or []
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=self.feature_names)

        params = self.config.get("xgb_params", self._default_params())
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config.get("num_boost_round", 2000),
            evals=[(dvalid, "valid")],
            early_stopping_rounds=self.config.get("early_stopping_rounds", 50),
            verbose_eval=False,
        )
        return {
            "best_iteration": self.model.best_iteration,
            "best_score": self.model.best_score,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("XGB model not trained")
        import xgboost as xgb
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.model.predict(dtest, iteration_range=(0, self.model.best_iteration + 1))

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({
            "model": self.model,
            "config": self.config,
            "feature_names": self.feature_names,
        }, path)

    @classmethod
    def load(cls, path: str) -> "XGBModel":
        import joblib
        data = joblib.load(path)
        model = cls(data["config"])
        model.model = data["model"]
        model.feature_names = data["feature_names"]
        return model

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if self.model is None or not self.feature_names:
            return None
        importance = self.model.get_score(importance_type="gain")
        df = pd.DataFrame([
            {"feature": k, "importance": v} for k, v in importance.items()
        ]).sort_values("importance", ascending=False)
        return df

    @staticmethod
    def _default_params() -> dict:
        return {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "learning_rate": 0.05,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "lambda": 1.0,
            "alpha": 0.1,
        }
