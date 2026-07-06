"""配置加载与校验。"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class DataConfig:
    granularity: str = "15min"
    reader_type: str = "dataset_csv"
    dataset_root: str = "./Dataset"
    sources: dict = field(default_factory=dict)


@dataclass
class PreprocessingConfig:
    missing_strategy: str = "ffill"
    outlier_method: str = "iqr"
    outlier_threshold: float = 3.0
    normalize_method: str = "none"
    fill_00_with_24: bool = True
    solar_wind_night_fill: float = 0.0


@dataclass
class FeatureConfig:
    lag_windows: list = field(default_factory=lambda: [1, 7])
    rolling_windows: list = field(default_factory=lambda: [96, 672])
    use_price_lags: bool = True
    use_actual_lags: bool = True
    use_date_features: bool = True


@dataclass
class ModelTrainConfig:
    test_days: int = 30
    early_stopping_rounds: int = 50
    num_boost_round: int = 5000
    cross_validation: bool = True
    cv_folds: int = 5


@dataclass
class ModelPredictConfig:
    horizon: int = 96


@dataclass
class ModelConfig:
    type: str = "lgbm"
    wavelet: str = "db4"
    decompose_level: int = 2
    window_size: int = 30
    train: ModelTrainConfig = field(default_factory=ModelTrainConfig)
    predict: ModelPredictConfig = field(default_factory=ModelPredictConfig)
    lgbm_params: dict = field(default_factory=dict)


@dataclass
class OptunaConfig:
    enabled: bool = False
    n_trials: int = 50
    cv_folds: int = 3


@dataclass
class EvaluationConfig:
    spike_threshold_sigma: float = 2.0
    alert_thresholds: dict = field(default_factory=lambda: {"mape": 20, "direction_accuracy": 55})


@dataclass
class OutputConfig:
    prediction_dir: str = "./data/predictions/"
    report_dir: str = "./reports/"
    model_dir: str = "./models/"
    log_dir: str = "./logs/"
    log_level: str = "INFO"
    output_methods: list = field(default_factory=lambda: ["csv"])


@dataclass
class ScheduleConfig:
    run_time: str = "08:00"
    auto_trigger: bool = False


@dataclass
class CalendarConfig:
    use_chinese_calendar: bool = True
    custom_holidays: list = field(default_factory=list)


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    calendar: CalendarConfig = field(default_factory=CalendarConfig)


def _build_dataclass(cls, data: dict) -> Any:
    """将字典递归转换为 dataclass。"""
    if not isinstance(data, dict):
        return data
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for key, value in data.items():
        if key in field_types and hasattr(field_types[key], "__dataclass_fields__"):
            kwargs[key] = _build_dataclass(field_types[key], value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str = "config/default.yaml") -> dict:
    """加载 YAML 配置为字典。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_app_config(path: str = "config/default.yaml") -> AppConfig:
    """加载 YAML 配置为类型安全的 AppConfig。"""
    raw = load_config(path)
    return _build_dataclass(AppConfig, raw)


def load_features_config(path: str = "config/features.yaml") -> dict:
    """加载特征配置文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
