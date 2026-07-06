"""每日自动运行器。"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from src.data.reader import DatasetCSVReader
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.monitor import ModelMonitor
from src.evaluation.report import ReportGenerator
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.utils.logger import setup_logger
from src.utils.time_utils import TimeUtils


class DailyRunner:
    """每日自动运行器。"""

    def __init__(self, config_path: str = "config/default.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.config_path = config_path
        self.predict_pipeline = PredictPipeline(config_path)
        self.metrics = MetricsCalculator()
        self.report_generator = ReportGenerator(
            report_dir=self.config["output"].get("report_dir", "./reports")
        )
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
        periods = TimeUtils.classify_period(pd.to_datetime(valid["timestamp"]))
        metrics = self.metrics.compute_all(y_true, y_pred, periods=periods)

        record = self.monitor.record(target_date, y_true, y_pred)
        alerts = self.monitor.check_prediction_quality(y_true, y_pred)
        if alerts:
            for alert in alerts:
                self.logger.warning(f"告警: {alert}")

        self.monitor.save_history(str(self.history_path))
        return {"metrics": metrics, "record": record, "alerts": alerts, "n_evaluated": len(valid)}

    def evaluate_range(
        self,
        start_date: str,
        end_date: str,
        model_path: Optional[str] = None,
    ) -> dict:
        """
        对一段时间内的每一天进行预测、评估、独立存储结果并画图，
        最终汇总所有天的结果计算综合评价指标。
        """
        dates = pd.date_range(start=start_date, end=end_date, freq="D").strftime("%Y-%m-%d").tolist()
        self.logger.info(f"开始批量预测与评估: {start_date} ~ {end_date}, 共 {len(dates)} 天")

        report_dir = Path(self.config["output"].get("report_dir", "./reports"))
        pred_dir = Path(self.config["output"].get("prediction_dir", "./data/predictions"))
        report_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

        reader = DatasetCSVReader(
            self.config["data"]["dataset_root"],
            self.config["data"]["sources"],
        )
        target_df = reader.read_target()
        actual_series = target_df.set_index("timestamp")["value"]

        per_day_results = []
        all_y_true, all_y_pred, all_timestamps = [], [], []

        for date in dates:
            self.logger.info(f"处理 {date}")
            # 1. 预测并保存
            pred_df = self.predict_pipeline.run(date, model_path=model_path)
            pred_path = pred_dir / f"{date}_predictions.csv"
            pred_df.to_csv(pred_path, index=False)

            # 2. 对齐实际值
            pred_df["actual_price"] = pred_df["timestamp"].map(actual_series)
            valid = pred_df.dropna(subset=["actual_price"])
            if valid.empty:
                self.logger.warning(f"{date} 实际值尚未公布，仅保存预测")
                per_day_results.append({"date": date, "evaluated": False, "n_samples": len(pred_df)})
                continue

            y_true = valid["actual_price"].values
            y_pred = valid["predicted_price"].values
            timestamps = valid["timestamp"].values
            periods = TimeUtils.classify_period(pd.to_datetime(timestamps))
            metrics = self.metrics.compute_all(y_true, y_pred, periods=periods)

            # 3. 独立画图
            plot_path = report_dir / "daily" / f"{date}.png"
            self.report_generator.plot_daily(y_true, y_pred, timestamps, str(plot_path), date=date)

            per_day_results.append({
                "date": date,
                "evaluated": True,
                "n_samples": len(valid),
                "metrics": metrics,
                "prediction_path": str(pred_path),
                "plot_path": str(plot_path),
            })

            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
            all_timestamps.append(timestamps)

        # 4. 汇总综合评价指标
        aggregate = None
        if all_y_true:
            y_true_all = np.concatenate(all_y_true)
            y_pred_all = np.concatenate(all_y_pred)
            timestamps_all = np.concatenate(all_timestamps)
            periods_all = TimeUtils.classify_period(pd.to_datetime(timestamps_all))
            aggregate = self.metrics.compute_all(y_true_all, y_pred_all, periods=periods_all)

            # 汇总图
            summary_plot_path = report_dir / "daily" / f"summary_{start_date}_{end_date}.png"
            self.report_generator.plot_daily(
                y_true_all, y_pred_all, timestamps_all, str(summary_plot_path),
                date=f"{start_date} ~ {end_date} 汇总"
            )

        # 5. 保存每日指标 CSV / JSON
        summary_csv = report_dir / "daily_metrics_summary.csv"
        summary_json = report_dir / "daily_metrics_summary.json"
        self._save_daily_summary(per_day_results, summary_csv, summary_json, aggregate)

        return {
            "dates": dates,
            "per_day_results": per_day_results,
            "aggregate_metrics": aggregate,
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "n_evaluated": sum(1 for r in per_day_results if r.get("evaluated", False)),
        }

    @staticmethod
    def _save_daily_summary(per_day_results: list, csv_path: Path, json_path: Path, aggregate: Optional[dict]) -> None:
        """将每日指标保存为 CSV 和 JSON，并写入汇总指标。"""
        rows = []
        for r in per_day_results:
            row = {"date": r["date"], "evaluated": r.get("evaluated", False), "n_samples": r.get("n_samples", 0)}
            if r.get("evaluated") and "metrics" in r:
                m = r["metrics"]
                row.update({
                    "MAE": m["MAE"],
                    "RMSE": m["RMSE"],
                    "MAPE": m["MAPE"],
                    "sMAPE": m["sMAPE"],
                    "R2": m["R2"],
                    "direction_accuracy": m["direction_accuracy"],
                    "spike_capture_rate": m["spike_metrics"]["spike_capture_rate"],
                    "spike_false_alarm": m["spike_metrics"]["spike_false_alarm"],
                    "spike_miss_rate": m["spike_metrics"]["spike_miss_rate"],
                })
                for period, val in m.get("period_mape", {}).items():
                    row[f"period_mape_{period}"] = val
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        summary = {
            "per_day": rows,
            "aggregate_metrics": aggregate,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

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
