"""模型运行监控与告警。"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import MetricsCalculator


class ModelMonitor:
    """模型运行监控器。"""

    def __init__(self, alert_thresholds: Optional[dict] = None):
        self.thresholds = alert_thresholds or {"mape": 20, "direction_accuracy": 55}
        self.metrics_history: list[dict] = []

    def check_prediction_quality(self, y_true: np.ndarray, y_pred: np.ndarray) -> list[str]:
        """检查预测质量是否低于阈值，触发告警。"""
        calc = MetricsCalculator()
        metrics = calc.compute_all(y_true, y_pred)
        alerts = []
        if metrics["MAPE"] > self.thresholds.get("mape", 20):
            alerts.append(f"MAPE {metrics['MAPE']:.1f}% exceeds threshold {self.thresholds['mape']}%")
        if metrics["direction_accuracy"] < self.thresholds.get("direction_accuracy", 55):
            alerts.append(
                f"Direction Accuracy {metrics['direction_accuracy']:.1f}% below threshold {self.thresholds['direction_accuracy']}%"
            )
        return alerts

    def record(self, date: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """记录一次预测评估结果。"""
        calc = MetricsCalculator()
        metrics = calc.compute_all(y_true, y_pred)
        record = {"date": date, **metrics}
        self.metrics_history.append(record)
        return record

    def detect_data_drift(
        self,
        recent_features: np.ndarray,
        reference_features: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        使用 PSI (Population Stability Index) 检测数据漂移。
        对每个特征计算 PSI，返回最大 PSI。
        """
        eps = 1e-8
        psi_values = []
        for i in range(reference_features.shape[1]):
            ref = reference_features[:, i]
            rec = recent_features[:, i]
            # 基于参考分布分箱
            bins = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
            bins[-1] += eps  # 包含最大值
            ref_dist, _ = np.histogram(ref, bins=bins)
            rec_dist, _ = np.histogram(rec, bins=bins)
            ref_dist = ref_dist / (ref_dist.sum() + eps)
            rec_dist = rec_dist / (rec_dist.sum() + eps)
            psi = np.sum((ref_dist - rec_dist) * np.log((ref_dist + eps) / (rec_dist + eps)))
            psi_values.append(psi)
        return float(np.max(psi_values))

    def should_retrain(
        self,
        recent_days: int = 5,
        mape_threshold: Optional[float] = None,
        direction_threshold: Optional[float] = None,
        psi: Optional[float] = None,
        psi_threshold: float = 0.25,
    ) -> tuple[bool, list[str]]:
        """
        判断是否需要触发重训练。
        返回 (should_retrain, reasons)。
        """
        reasons = []
        if len(self.metrics_history) < recent_days:
            return False, reasons

        recent = self.metrics_history[-recent_days:]
        mape_threshold = mape_threshold or self.thresholds.get("mape", 20)
        direction_threshold = direction_threshold or self.thresholds.get("direction_accuracy", 55)

        # 连续 N 天 MAPE 超过阈值
        if all(r["MAPE"] > mape_threshold for r in recent):
            reasons.append(f"MAPE exceeded {mape_threshold}% for {recent_days} consecutive days")

        # 方向准确率连续下降
        direction_accs = [r["direction_accuracy"] for r in self.metrics_history[-recent_days - 1:]]
        if len(direction_accs) >= recent_days + 1 and all(
            direction_accs[i] > direction_accs[i + 1] for i in range(recent_days)
        ):
            reasons.append(f"Direction accuracy declined for {recent_days} consecutive days")

        # 方向准确率连续低于阈值
        if all(r["direction_accuracy"] < direction_threshold for r in recent):
            reasons.append(f"Direction accuracy below {direction_threshold}% for {recent_days} consecutive days")

        # PSI 漂移
        if psi is not None and psi > psi_threshold:
            reasons.append(f"PSI {psi:.2f} exceeds threshold {psi_threshold}")

        return len(reasons) > 0, reasons

    def save_history(self, path: str) -> None:
        """保存评估历史到 CSV。"""
        if not self.metrics_history:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(path, index=False)

    def load_history(self, path: str) -> None:
        """从 CSV 加载评估历史。"""
        if not Path(path).exists():
            return
        df = pd.read_csv(path)
        self.metrics_history = df.to_dict("records")
