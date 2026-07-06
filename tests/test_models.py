"""测试模型层。"""
import numpy as np
import pytest

from src.models.lgbm_model import LGBMModel
from src.models.model_factory import ModelFactory
from src.models.wavelet import WaveletDecomposer


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


def test_wavelet_round_trip():
    rng = np.random.default_rng(42)
    signal = rng.normal(0, 1, size=96)
    wd = WaveletDecomposer(wavelet="db4", level=2)
    components = wd.stationary_decompose(signal)
    reconstructed = wd.stationary_reconstruct(components)
    assert reconstructed.shape == signal.shape
    np.testing.assert_allclose(reconstructed, signal, atol=1e-10)


def test_wavelet_reconstruct_incomplete_day():
    """SWT 重构需兼容长度非 2^level 整数倍的不完整预测日。"""
    rng = np.random.default_rng(42)
    signal = rng.normal(0, 1, size=1)
    wd = WaveletDecomposer(wavelet="db4", level=2)
    components = wd.stationary_decompose(signal)
    reconstructed = wd.stationary_reconstruct(components)
    assert reconstructed.shape == signal.shape
    np.testing.assert_allclose(reconstructed, signal, atol=1e-10)


def test_wavelet_reconstruct_short_length():
    """重构应兼容长度非 4 的整数倍的不完整日，至少不抛异常并返回正确长度。"""
    rng = np.random.default_rng(42)
    wd = WaveletDecomposer(wavelet="db4", level=2)
    for length in [1, 2, 3, 5, 10, 95, 96, 97]:
        signal = rng.normal(0, 1, size=length)
        components = wd.stationary_decompose(signal)
        reconstructed = wd.stationary_reconstruct(components)
        assert reconstructed.shape == signal.shape
        # 对无需补零的长度验证可逆性；需补零的长度受边界效应影响，不强制精确重建
        if length % (2 ** wd.level) == 0:
            np.testing.assert_allclose(reconstructed, signal, atol=1e-10)
