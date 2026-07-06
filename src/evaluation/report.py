"""报告生成器。"""
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.metrics import MetricsCalculator
from src.utils.time_utils import TimeUtils


class ReportGenerator:
    """报告生成器：日报 / 周报 / 月报。"""

    def __init__(self, report_dir: str = "./reports"):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate_daily(
        self,
        date: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: np.ndarray,
        importance: Optional[pd.DataFrame] = None,
        save_plot: bool = True,
    ) -> dict:
        """生成每日预测精度简报。"""
        calc = MetricsCalculator()
        periods = None
        if len(timestamps) > 0:
            periods = TimeUtils.classify_period(pd.to_datetime(timestamps))
        metrics = calc.compute_all(y_true, y_pred, periods=periods)

        report = {
            "date": date,
            "metrics": metrics,
            "n_samples": len(y_true),
        }

        if save_plot:
            save_path = self.report_dir / "daily" / f"{date}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.plot_prediction_vs_actual(y_true, y_pred, timestamps, str(save_path))
            report["plot_path"] = str(save_path)

            if importance is not None:
                imp_path = self.report_dir / "daily" / f"{date}_importance.png"
                self.plot_feature_importance(importance, top_n=10, save_path=str(imp_path))
                report["importance_plot_path"] = str(imp_path)

        return report

    def generate_monthly(
        self,
        month: str,
        history: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> str:
        """
        月度模型评估报告。
        history 需包含列：date, y_true, y_pred。
        """
        required = {"date", "y_true", "y_pred"}
        if not required.issubset(history.columns):
            raise ValueError(f"history must contain columns: {required}")

        calc = MetricsCalculator()
        history["mape"] = history.apply(
            lambda row: calc.mape(np.array([row["y_true"]]), np.array([row["y_pred"]])),
            axis=1,
        )
        history["mae"] = history.apply(
            lambda row: calc.mae(np.array([row["y_true"]]), np.array([row["y_pred"]])),
            axis=1,
        )

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        history.plot(x="date", y="mape", ax=axes[0, 0], title="MAPE Trend")
        history.plot(x="date", y="mae", ax=axes[0, 1], title="MAE Trend")
        axes[1, 0].scatter(history["y_true"], history["y_pred"], alpha=0.5)
        axes[1, 0].plot(
            [history["y_true"].min(), history["y_true"].max()],
            [history["y_true"].min(), history["y_true"].max()],
            "r--",
        )
        axes[1, 0].set_title("Predicted vs Actual")
        axes[1, 0].set_xlabel("Actual")
        axes[1, 0].set_ylabel("Predicted")
        axes[1, 1].hist(history["y_true"] - history["y_pred"], bins=30, edgecolor="k")
        axes[1, 1].set_title("Residual Distribution")
        plt.tight_layout()

        if save_path is None:
            save_path = self.report_dir / "monthly" / f"{month}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return str(save_path)

    def plot_prediction_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: np.ndarray,
        save_path: str,
    ) -> None:
        """预测 vs 实际对比图（日曲线）。"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        ts = pd.to_datetime(timestamps)
        axes[0].plot(ts, y_true, label="Actual", linewidth=1.5)
        axes[0].plot(ts, y_pred, label="Predicted", linewidth=1.5, alpha=0.8)
        axes[0].set_title("Prediction vs Actual")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(ts, y_true - y_pred, color="gray", linewidth=1)
        axes[1].axhline(0, color="red", linestyle="--")
        axes[1].set_title("Residual (Actual - Predicted)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 10,
        save_path: str = None,
    ) -> None:
        """特征重要性柱状图。"""
        df = importance_df.copy()
        if "importance" not in df.columns:
            raise ValueError("importance_df must contain 'importance' column")
        df = df.sort_values("importance", ascending=True).tail(top_n)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(df["feature"], df["importance"])
        ax.set_title(f"Top {top_n} Feature Importance")
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
            plt.close(fig)

    def plot_residual_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str,
    ) -> None:
        """残差分析图。"""
        residuals = y_true - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(residuals, bins=50, edgecolor="k")
        axes[0].set_title("Residual Distribution")
        axes[0].set_xlabel("Residual")

        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title("Q-Q Plot")
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
