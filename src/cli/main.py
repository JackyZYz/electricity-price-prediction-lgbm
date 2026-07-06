"""命令行入口。"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.train_pipeline import TrainPipeline
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.daily_runner import DailyRunner


def _print_metrics(metrics):
    for k, v in metrics.items():
        if k == "spike_metrics":
            print(f"  {k}:")
            for sk, sv in v.items():
                print(f"    {sk}: {sv}")
        elif k == "period_mape":
            print(f"  {k}:")
            for pk, pv in v.items():
                print(f"    {pk}: {pv:.2f}%")
        else:
            print(f"  {k}: {v}")


def cmd_train(args):
    pipeline = TrainPipeline(config_path=args.config)
    result = pipeline.run()
    print("训练完成")
    print(f"模型类型: {result['model_type']}")
    print(f"模型路径: {result['model_path']}")
    print("评估指标:")
    _print_metrics(result["metrics"])
    if result["importance"] is not None:
        print("\nTOP10 特征重要性:")
        print(result["importance"].head(10).to_string(index=False))


def cmd_predict(args):
    pipeline = PredictPipeline(config_path=args.config)
    result = pipeline.run(args.date, model_path=args.model)
    print(f"预测完成: {args.date}")
    print(result.head())


def cmd_evaluate(args):
    runner = DailyRunner(config_path=args.config)
    result = runner.evaluate(args.date)
    if result is None:
        print(f"{args.date} 的实际值尚未公布，无法评估")
        return
    print(f"评估完成: {args.date}")
    print("评估指标:")
    _print_metrics(result["metrics"])
    if result["alerts"]:
        print("告警:")
        for alert in result["alerts"]:
            print(f"  - {alert}")


def cmd_retrain(args):
    print("开始重训练...")
    pipeline = TrainPipeline(config_path=args.config)
    result = pipeline.run()
    print("重训练完成")
    print(f"模型路径: {result['model_path']}")
    print("评估指标:")
    _print_metrics(result["metrics"])


def cmd_daily(args):
    runner = DailyRunner(config_path=args.config)
    result = runner.run(
        target_date=args.date,
        model_path=args.model,
        auto_retrain=args.auto_retrain,
    )
    print(f"日报运行完成: {result['target_date']}")
    print(f"预测点数: {len(result['prediction'])}")
    if result["evaluation"]:
        print("评估指标:")
        _print_metrics(result["evaluation"]["metrics"])
    if result["retrain_triggered"]:
        print(f"已触发重训练: {result['retrain_reasons']}")


def main():
    parser = argparse.ArgumentParser(description="用户侧日前出清电价预测")
    parser.add_argument("--config", default="config/default.yaml", help="配置文件路径")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.set_defaults(func=cmd_train)

    predict_parser = subparsers.add_parser("predict", help="预测指定日期")
    predict_parser.add_argument("--date", required=True, help="目标日期，格式 YYYY-MM-DD")
    predict_parser.add_argument("--model", default=None, help="模型文件路径（可选）")
    predict_parser.set_defaults(func=cmd_predict)

    evaluate_parser = subparsers.add_parser("evaluate", help="评估指定日期的预测结果")
    evaluate_parser.add_argument("--date", required=True, help="目标日期，格式 YYYY-MM-DD")
    evaluate_parser.set_defaults(func=cmd_evaluate)

    retrain_parser = subparsers.add_parser("retrain", help="手动重训练模型")
    retrain_parser.set_defaults(func=cmd_retrain)

    daily_parser = subparsers.add_parser("daily", help="运行日报")
    daily_parser.add_argument("--date", default=None, help="目标日期，默认明天")
    daily_parser.add_argument("--model", default=None, help="模型文件路径（可选）")
    daily_parser.add_argument("--auto-retrain", action="store_true", help="是否自动触发重训练")
    daily_parser.set_defaults(func=cmd_daily)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
