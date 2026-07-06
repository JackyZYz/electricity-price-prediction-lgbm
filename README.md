# 用户侧日前出清电价预测

基于 LightGBM 的江苏省用户侧日前出清电价预测项目。

## 数据说明

实际数据集位于 `Dataset/` 目录：
- 48 个 CSV 文件，13 个数据类别
- 时间范围：2026-01-01 ~ 2026-07-01
- 时间粒度：15 分钟，每日 96 时点
- 目标变量：`Dataset/用户侧日前出清发布/用户侧日前出清发布_统一结算点电价最终结果.csv`

## 项目结构

```
.
├── config/                # 配置文件
├── src/                   # 源代码
│   ├── data/              # 数据读取、校验、适配
│   ├── features/          # 特征工程
│   ├── models/            # 小波分解、LGBM、预测器
│   ├── evaluation/        # 评估指标与报告
│   ├── pipeline/          # 训练/预测流程
│   └── utils/             # 工具函数
├── prototype/             # 原型验证 Notebook/脚本
├── data/                  # 运行时生成的数据（gitignore）
├── models/                # 模型保存目录（gitignore）
├── reports/               # 报告输出（gitignore）
├── tests/                 # 单元测试
└── scripts/               # 运维脚本
```

## 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 运行原型验证：
   ```bash
   jupyter notebook prototype/prototype_lgbm.ipynb
   ```

3. 运行完整训练流程：
   ```bash
   python -m src.pipeline.train_pipeline
   ```

## 主要文档

- `需求文档_用户侧日前出清电价预测.md`
- `原型架构设计_纯LGBM快速验证.md`
- `详细设计方案_用户侧日前出清电价预测.md`
- `数据集更新影响分析.md`
