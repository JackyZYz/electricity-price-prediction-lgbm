"""LightGBM 训练器。"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd


class LGBMTrainer:
    """LightGBM 训练器"""

    def __init__(self, params: dict, component_name: str):
        self.params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": params.get("learning_rate", 0.05),
            "num_leaves": params.get("num_leaves", 63),
            "max_depth": params.get("max_depth", 12),
            "min_data_in_leaf": params.get("min_data_in_leaf", 20),
            "feature_fraction": params.get("feature_fraction", 0.8),
            "bagging_fraction": params.get("bagging_fraction", 0.8),
            "bagging_freq": params.get("bagging_freq", 5),
            "lambda_l1": params.get("lambda_l1", 0.01),
            "lambda_l2": params.get("lambda_l2", 0.01),
            "verbosity": -1,
        }
        self.component_name = component_name
        self.model = None
        self.feature_names = None

    def _compute_weights(self, y: np.ndarray, method: str = "none", spike_sigma: float = 2.0,
                         high_price_percentile: float = 80.0) -> np.ndarray:
        """根据目标值计算样本权重，用于提升高价/尖峰拟合。"""
        if method == "none" or method is None:
            return np.ones_like(y, dtype=float)
        if method == "price_level":
            threshold = np.percentile(y, high_price_percentile)
            weights = np.where(y >= threshold, 2.0, 1.0)
            return weights
        if method == "spike":
            threshold = np.mean(y) + spike_sigma * np.std(y)
            weights = np.where(y >= threshold, 3.0, 1.0)
            return weights
        if method == "quantile":
            # 按电价分位数连续加权，高价样本权重更大
            ranks = np.argsort(np.argsort(y))
            quantiles = (ranks + 1) / len(y)
            weights = 1.0 + 2.0 * quantiles
            return weights
        if method == "combined":
            spike_threshold = np.mean(y) + spike_sigma * np.std(y)
            level_threshold = np.percentile(y, high_price_percentile)
            weights = np.where(y >= spike_threshold, 4.0,
                               np.where(y >= level_threshold, 2.0, 1.0))
            return weights
        return np.ones_like(y, dtype=float)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        feature_names: list,
        early_stopping_rounds: int = 50,
        num_boost_round: int = 5000,
        weight_method: str = "none",
        weight_spike_sigma: float = 2.0,
        weight_high_price_percentile: float = 80.0,
    ) -> dict:
        self.feature_names = feature_names
        train_weights = self._compute_weights(
            y_train, method=weight_method,
            spike_sigma=weight_spike_sigma,
            high_price_percentile=weight_high_price_percentile
        )
        train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights, feature_name=feature_names)
        valid_data = lgb.Dataset(X_valid, label=y_valid, feature_name=feature_names, reference=train_data)

        callbacks = [
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = lgb.train(
                params=self.params,
                train_set=train_data,
                valid_sets=[valid_data],
                valid_names=["valid"],
                num_boost_round=num_boost_round,
                callbacks=callbacks,
            )
        return {
            "best_iteration": self.model.best_iteration,
            "best_score": self.model.best_score["valid"]["rmse"],
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"Component '{self.component_name}' not trained")
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        importance = self.model.feature_importance(importance_type=importance_type)
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)

    def save(self, path: str):
        import joblib
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "params": self.params,
            "component_name": self.component_name,
        }, path)

    @classmethod
    def load(cls, path: str):
        import joblib
        data = joblib.load(path)
        trainer = cls(data["params"], data["component_name"])
        trainer.model = data["model"]
        trainer.feature_names = data["feature_names"]
        return trainer
