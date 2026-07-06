"""测试模型层。"""
import numpy as np
import pytest

from src.models.lgbm_model import LGBMModel
from src.models.model_factory import ModelFactory


def make_data(n=500):
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(n, 5))
    y = X[:, 0] * 2 + X[:, 1] * -1 + rng.normal(0, 0.1, size=n)
    return X, y


def test_lgbm_model_fit_predict():
    X, y = make_data()
    split = int(len(X) * 0.8)
    X_train, X_valid = X[:split], X[split:]
    y_train, y_valid = y[:split], y[split:]

    model = LGBMModel({"lgbm_params": {"num_leaves": 15, "max_depth": 5}, "num_boost_round": 100})
    model.fit(X_train, y_train, X_valid, y_valid, feature_names=["f0", "f1", "f2", "f3", "f4"])
    preds = model.predict(X_valid)
    assert len(preds) == len(X_valid)
    assert preds.shape == y_valid.shape


def test_model_factory_lgbm():
    model = ModelFactory.create("lgbm", {"lgbm_params": {}})
    assert isinstance(model, LGBMModel)


def test_model_factory_unknown():
    with pytest.raises(ValueError):
        ModelFactory.create("unknown", {})
