"""WT-LGBM 模型实现（基于 SWT 保持分量长度对齐）。"""
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseModel
from .lgbm_trainer import LGBMTrainer
from .wavelet import WaveletDecomposer


class WTLGBMModel(BaseModel):
    """
    小波变换 + LightGBM 模型。
    使用平稳小波变换（SWT）将目标分解为长度相等的低频/高频分量，
    对每个分量单独训练 LGBM，预测时求和得到最终电价。
    """

    def __init__(self, config: dict):
        self.config = config
        self.wavelet = WaveletDecomposer(
            wavelet=config.get("wavelet", "db4"),
            level=config.get("decompose_level", 2),
        )
        self.lgbm_params = config.get("lgbm_params", {})
        self.early_stopping_rounds = config.get("early_stopping_rounds", 50)
        self.num_boost_round = config.get("num_boost_round", 2000)
        self.trainers: dict[str, LGBMTrainer] = {}
        self.feature_names: Optional[list] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None,
    ) -> dict:
        if X_valid is None or y_valid is None:
            split_idx = int(len(X_train) * 0.8)
            X_valid, y_valid = X_train[split_idx:], y_train[split_idx:]
            X_train, y_train = X_train[:split_idx], y_train[:split_idx]

        self.feature_names = feature_names or []
        train_components = self.wavelet.stationary_decompose(y_train)
        valid_components = self.wavelet.stationary_decompose(y_valid)

        results = {}
        for key in ["low"] + [f"high_{i}" for i in range(1, self.wavelet.level + 1)]:
            trainer = LGBMTrainer(self.lgbm_params, key)
            results[key] = trainer.train(
                X_train=X_train,
                y_train=train_components[key],
                X_valid=X_valid,
                y_valid=valid_components[key],
                feature_names=self.feature_names,
                early_stopping_rounds=self.early_stopping_rounds,
                num_boost_round=self.num_boost_round,
            )
            self.trainers[key] = trainer
        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trainers:
            raise RuntimeError("WT-LGBM 模型未训练")
        components = {}
        for key, trainer in self.trainers.items():
            components[key] = trainer.predict(X)
        return self.wavelet.stationary_reconstruct(components)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({
            "trainers": self.trainers,
            "config": self.config,
            "feature_names": self.feature_names,
        }, path)

    @classmethod
    def load(cls, path: str) -> "WTLGBMModel":
        import joblib
        data = joblib.load(path)
        model = cls(data["config"])
        model.trainers = data["trainers"]
        model.feature_names = data["feature_names"]
        return model

    def get_feature_importance(self, aggregate: str = "mean") -> Optional[pd.DataFrame]:
        """汇总各分量特征重要性。"""
        if not self.trainers:
            return None
        importances = []
        for trainer in self.trainers.values():
            imp = trainer.get_feature_importance()
            imp = imp.rename(columns={"importance": trainer.component_name})
            importances.append(imp)
        merged = importances[0]
        for imp in importances[1:]:
            merged = merged.merge(imp, on="feature", how="outer")
        imp_cols = [c for c in merged.columns if c != "feature"]
        merged[imp_cols] = merged[imp_cols].fillna(0)
        if aggregate == "mean":
            merged["importance"] = merged[imp_cols].mean(axis=1)
        elif aggregate == "sum":
            merged["importance"] = merged[imp_cols].sum(axis=1)
        return merged[["feature", "importance"]].sort_values("importance", ascending=False)
