"""纯 LightGBM 模型实现。"""
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseModel
from .lgbm_trainer import LGBMTrainer


class LGBMModel(BaseModel):
    """纯 LightGBM 回归模型。"""

    def __init__(self, config: dict):
        self.config = config
        self.component_name = config.get("component_name", "lgbm")
        self.trainer = LGBMTrainer(config.get("lgbm_params", {}), self.component_name)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None,
    ) -> dict:
        if X_valid is None or y_valid is None:
            # 默认从训练集末尾切出 20% 作为验证集（时序）
            split_idx = int(len(X_train) * 0.8)
            X_valid, y_valid = X_train[split_idx:], y_train[split_idx:]
            X_train, y_train = X_train[:split_idx], y_train[:split_idx]
        return self.trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            feature_names=feature_names or [],
            early_stopping_rounds=self.config.get("early_stopping_rounds", 50),
            num_boost_round=self.config.get("num_boost_round", 2000),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.trainer.predict(X)

    def save(self, path: str) -> None:
        self.trainer.save(path)

    @classmethod
    def load(cls, path: str) -> "LGBMModel":
        trainer = LGBMTrainer.load(path)
        # 用 trainer 中的参数重建配置
        config = {
            "component_name": trainer.component_name,
            "lgbm_params": trainer.params,
        }
        model = cls(config)
        model.trainer = trainer
        return model

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        return self.trainer.get_feature_importance()
