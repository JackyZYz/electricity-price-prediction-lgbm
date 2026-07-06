"""Setup script for electricity price prediction package."""
from setuptools import setup, find_packages

setup(
    name="electricity-price-prediction",
    version="0.1.0",
    description="基于 LightGBM 的用户侧日前出清电价预测",
    author="",
    packages=find_packages(exclude=["tests", "tests.*", "prototype", "prototype.*"]),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "lightgbm>=4.0.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.2.0",
        "PyWavelets>=1.4.0",
        "optuna>=3.0.0",
        "joblib>=1.2.0",
        "loguru>=0.7.0",
        "openpyxl>=3.0.0",
        "pyarrow>=12.0.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "baseline": ["xgboost>=2.0.0", "statsmodels>=0.14.0"],
        "calendar": ["chinese-calendar>=1.9.0"],
        "dev": ["pytest>=7.0.0", "black>=23.0.0", "isort>=5.12.0"],
    },
    entry_points={
        "console_scripts": [
            "epp-train=src.cli.main:main",
        ],
    },
)
