"""每日自动运行器。"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from src.data.reader import DatasetCSVReader
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.monitor import ModelMonitor
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.utils.logger import setup_logger


class DailyRunner:
    """每日自动运行器。"""

    def __init__(self, config_path: str = "config/default.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.config_path = config_path
        self.predict_pipeline = PredictPipeline(config_path)
        self.metrics = MetricsCalculator()
        self.monitor = ModelMonitor(
            alert_thresholds=self.config["evaluation"].get("alert_thresholds", {"mape": 20, "direction_accuracy": 55})
        )
        self.logger = setup_logger("daily_runner", self.config["output"].get("log_dir", "logs"))
        self.history_path = Path(self.config["output"].get("report_dir", "./reports")) / "metrics_history.csv"
        self.monitor.load_history(str(self.history_path))

    def predict(self, target_date: str, model_path: Optional[str] = None) -> pd.DataFrame:
        """预测当日电价。"""
        self.logger.info(f"开始预测 {target_date}")
        result = self.predict_pipeline.run(target_date, model_path=model_path)
        self.logger.info(f"预测完成: {len(result)} 个点")
        return result

    def evaluate(self, target_date: str) -> Optional[dict]:
        """
        对比预测值与次日实际出清价。
        target_date 为预测日期，实际值需从 target_date 的次日或已公布日期读取。
        """
        pred_path = Path(self.config["output"].get("prediction_dir", "./data/predictions")) / f"{target_date}_predictions.csv"
        if not pred_path.exists():
            self.logger.warning(f"预测结果不存在: {pred_path}")
            return None

        pred_df = pd.read_csv(pred_path, parse_dates=["timestamp"])

        # 尝试读取实际值（假设实际值已公布）
        reader = DatasetCSVReader(
            self.config["data"]["dataset_root"],
            self.config["data"]["sources"],
        )
        target_df = reader.read_target()
        actual = target_df.set_index("timestamp")["value"]
        pred_df["actual_price"] = pred_df["timestamp"].map(actual)

        valid = pred_df.dropna(subset=["actual_price"])
        if valid.empty:
            self.logger.info(f"{target_date} 实际值尚未公布，跳过评估")
            return None

        y_true = valid["actual_price"].values
        y_pred = valid["predicted_price"].values
        periods = valid["timestamp"].dt.hour.map(lambda h: 0 if h < 6 else (2 if 8 <= h < 11 or 17 <= h < 21 else 1)).values
        metrics = self.metrics.compute_all(y_true, y_pred, periods=periods)

        record = self.monitor.record(target_date, y_true, y_pred)
        alerts = self.monitor.check_prediction_quality(y_true, y_pred)
        if alerts:
            for alert in alerts:
                self.logger.warning(f"告警: {alert}")

        self.monitor.save_history(str(self.history_path))
        return {"metrics": metrics, "record": record, "alerts": alerts, "n_evaluated": len(valid)}

    def should_retrain(self, recent_features: Optional[np.ndarray] = None,
                       reference_features: Optional[np.ndarray] = None) -> tuple[bool, list[str]]:
        """判断是否需要重训练。"""
        psi = None
        if recent_features is not None and reference_features is not None:
            psi = self.monitor.detect_data_drift(recent_features, reference_features)
        return self.monitor.should_retrain(psi=psi)

    def run(
        self,
        target_date: Optional[str] = None,
        model_path: Optional[str] = None,
        auto_retrain: bool = False,
    ) -> dict:
        """
        每日运行流程：
        1. 预测当日电价
        2. 若实际值已公布，评估并更新监控指标
        3. 必要时触发重训练
        """
        if target_date is None:
            target_date = (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        result = self.predict(target_date, model_path=model_path)
        eval_result = self.evaluate(target_date)

        retrain = False
        reasons = []
        if auto_retrain and eval_result is not None:
            retrain, reasons = self.should_retrain()
            if retrain:
                self.logger.warning(f"触发重训练: {reasons}")
                trainer = TrainPipeline(self.config_path)
                trainer.run()

        return {
            "target_date": target_date,
            "prediction": result,
            "evaluation": eval_result,
            "retrain_triggered": retrain,
            "retrain_reasons": reasons,
        }
