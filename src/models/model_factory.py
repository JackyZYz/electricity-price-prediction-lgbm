"""模型工厂。"""
from .base import BaseModel
from .lgbm_model import LGBMModel


class ModelFactory:
    """根据配置创建模型实例。"""

    _registry = {
        "lgbm": LGBMModel,
    }

    @classmethod
    def create(cls, model_type: str, config: dict) -> BaseModel:
        model_type = model_type.lower()
        if model_type == "wtlgbm":
            from .wtlgbm_model import WTLGBMModel
            return WTLGBMModel(config)
        if model_type == "xgb":
            from .xgb_model import XGBModel
            return XGBModel(config)
        if model_type == "arima":
            from .arima_model import ARIMAModel
            return ARIMAModel(config)
        if model_type not in cls._registry:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(cls._registry.keys()) + ['wtlgbm', 'xgb', 'arima']}")
        return cls._registry[model_type](config)

    @classmethod
    def register(cls, model_type: str, model_class: type):
        if not issubclass(model_class, BaseModel):
            raise TypeError("Model class must inherit from BaseModel")
        cls._registry[model_type.lower()] = model_class
