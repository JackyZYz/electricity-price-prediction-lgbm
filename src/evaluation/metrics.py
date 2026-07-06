"""评估指标。"""
import numpy as np


class MetricsCalculator:
    """评估指标计算器"""

    @staticmethod
    def mae(y_true, y_pred) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def rmse(y_true, y_pred) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mape(y_true, y_pred, epsilon: float = 1e-8) -> float:
        mask = np.abs(y_true) > epsilon
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + epsilon))) * 100)

    @staticmethod
    def smape(y_true, y_pred) -> float:
        return float(np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100)

    @staticmethod
    def r2(y_true, y_pred) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1 - ss_res / ss_tot)

    @staticmethod
    def direction_accuracy(y_true, y_pred) -> float:
        actual_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        return float(np.mean(actual_dir == pred_dir) * 100)

    @staticmethod
    def spike_capture_rate(y_true, y_pred, threshold_sigma: float = 2.0) -> dict:
        threshold = np.mean(y_true) + threshold_sigma * np.std(y_true)
        true_spikes = y_true > threshold
        pred_spikes = y_pred > threshold
        tp = np.sum(true_spikes & pred_spikes)
        fp = np.sum(~true_spikes & pred_spikes)
        fn = np.sum(true_spikes & ~pred_spikes)
        return {
            "spike_capture_rate": tp / (tp + fn) if (tp + fn) > 0 else 1.0,
            "spike_false_alarm": fp / (fp + tp) if (fp + tp) > 0 else 0.0,
            "spike_miss_rate": fn / (tp + fn) if (tp + fn) > 0 else 0.0,
        }

    @staticmethod
    def period_mape(y_true, y_pred, periods) -> dict:
        results = {}
        for period_name in np.unique(periods):
            mask = periods == period_name
            results[str(period_name)] = MetricsCalculator.mape(y_true[mask], y_pred[mask])
        return results

    def compute_all(self, y_true, y_pred, periods=None) -> dict:
        result = {
            "MAE": self.mae(y_true, y_pred),
            "RMSE": self.rmse(y_true, y_pred),
            "MAPE": self.mape(y_true, y_pred),
            "sMAPE": self.smape(y_true, y_pred),
            "R2": self.r2(y_true, y_pred),
            "direction_accuracy": self.direction_accuracy(y_true, y_pred),
            "spike_metrics": self.spike_capture_rate(y_true, y_pred),
        }
        if periods is not None:
            result["period_mape"] = self.period_mape(y_true, y_pred, periods)
        return result
