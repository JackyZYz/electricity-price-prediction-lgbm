# 原型架构设计：纯 LightGBM 电价预测快速验证

**版本**：v1.0
**日期**：2025-07-02
**定位**：在完整 WT-LGBM 方案之前，用最简架构快速跑通"数据→特征→训练→评估"闭环

---

## 一、原型目标

一句话：**用纯 LightGBM（不做小波分解）在 Mock 数据上跑出基准 MAPE，验证整体思路可行。**

| 对比维度 | 完整方案 (WT-LGBM) | 原型方案 (纯 LGBM) |
|----------|-------------------|---------------------|
| 模型复杂度 | 小波分解 + 多分量 LGBM | 单模型 LGBM |
| 特征数量 | ~30 维（4类） | ~15 维（精简核心） |
| 代码量 | 20+ 文件，模块化 | **3 个文件** |
| 运行方式 | 定时调度 / CLI | **单个 Jupyter Notebook 或单脚本** |
| 调参 | Optuna 自动搜索 | 手动设参 / 简单 GridSearch |
| 目标 | MAPE < 10%，上线部署 | **MAPE < 15%，验证特征有效性** |
| 依赖数据 | 真实数据 | **内置 Mock 数据生成器** |

---

## 二、原型架构

### 2.1 极简流程

```mermaid
flowchart LR
    gen["① 生成 Mock 数据<br/>(内置生成器)"] --> feat["② 特征工程<br/>(原始 + 构造 + 滞后 + 日期)"]
    feat --> split["③ 时序分割<br/>(最近30天为测试集)"]
    split --> train["④ 训练 LightGBM<br/>(默认参数，单模型)"]
    train --> eval["⑤ 评估<br/>(MAE/RMSE/MAPE/方向准确率)"]
    eval --> vis["⑥ 可视化<br/>(预测vs实际 · 特征重要性)"]
```

### 2.2 文件规划

```
prototype/
├── prototype_lgbm.ipynb          # ★ 主文件：Jupyter Notebook 一站式脚本
└── prototype_lgbm.py             #    备选：纯 Python 脚本版本（内容一致）

# 只需一个文件！或者拆成三个小文件：
# prototype/
# ├── mock_data.py                # Mock 数据生成
# ├── train.py                    # 训练 + 评估
# └── predict.py                  # 推理示例
```

---

## 三、模块拆解

### 3.1 Mock 数据生成器

**设计原则**：用最少的代码生成合理、可验证的模拟数据。

```python
def generate_mock_data(
    start_date: str = "2024-01-01",
    end_date: str = "2025-06-30",
    freq: str = "1h",          # "1h" = 24点/天, "15min" = 96点/天
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成包含以下列的 DataFrame：
    - datetime:        时间戳
    - sys_load_pred:   系统负荷预测 (MW)     → 双峰日曲线 + 周末效应 + 季节
    - wind_power_pred: 风电功率预测 (MW)     → 随机游走 + 季节
    - solar_power_pred:光伏功率预测 (MW)     → 日周期 + 季节
    - power_import:    受电计划 (MW)         → 稳定基值 + 小幅波动
    - coal_gen:        煤电出力 (MW)         → 与负荷正相关
    - gas_gen:         燃机出力 (MW)         → 峰段抬升
    - storage_plan:    储能计划 (MW)         → 峰谷套利模式
    - user_price:      用户侧日前出清电价     → ★ 目标变量

    电价生成逻辑：
        price = base_price
              + load_factor       × 负荷偏离均值的比例
              - renewable_factor  × 新能源渗透率
              + peak_multiplier   × 是否峰段(08-11, 17-21)
              + season_factor     × 季节效应
              + noise             × 随机噪声(5%)
    """
```

**电价生成公式**（用意：确保特征和目标有可学习的因果关系）：

```
price = 350                                    # 基础电价（元/MWh）
      + 150 × (load - load_mean) / load_std    # 负荷越高越贵
      - 100 × renewable_penetration            # 新能源越多越便宜
      + 80  × is_peak_period                  # 峰段溢价
      + 50  × sin(2π × (month-1)/12)          # 夏季贵、春秋便宜
      + ε    (ε ~ N(0, 25))                   # 随机噪声
```

### 3.2 精简特征集

原型只取最核心、能直接从 Mock 数据生成的特征（~15 维）：

```python
FEATURE_CONFIG = {
    # ---- 直接特征（6个） ----
    "sys_load_pred":       "系统负荷预测",
    "wind_power_pred":     "风电功率预测",
    "solar_power_pred":    "光伏功率预测",
    "power_import":        "受电计划",
    "coal_gen":            "煤电出力",
    "gas_gen":             "燃机出力",

    # ---- 构造特征（4个） ----
    "net_load":            "净负荷 = 负荷 - 风光",
    "renewable_penetration":"新能源渗透率",
    "import_ratio":        "外来电占比",
    "thermal_ratio":       "火电占比 = (煤+燃)/负荷",

    # ---- 滞后特征（3个） ----
    "price_lag_1d":        "昨日同时段电价",
    "price_lag_7d":        "上周同时段电价",
    "price_ma_24h":        "过去24h均价",

    # ---- 日期特征（4个） ----
    "hour":                "0-23",
    "weekday":             "0-6",
    "is_weekend":          "0/1",
    "month":               "1-12",
}
```

### 3.3 模型训练

```python
def train_lgbm(X_train, y_train, X_test, y_test) -> dict:
    """
    单模型训练，不做小波分解

    参数（原型默认值，不做调优）：
        learning_rate = 0.05
        num_leaves = 63
        max_depth = 10
        min_data_in_leaf = 20
        early_stopping_rounds = 50
        num_boost_round = 2000
    """
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 10,
        "min_data_in_leaf": 20,
        "verbosity": -1,
    }
    # 时序分割：最近30天为测试集
    # 训练 + early stopping
    # 返回：model, predictions, feature_importance
```

### 3.4 评估输出

```python
def evaluate(y_true, y_pred) -> dict:
    """单函数输出全部核心指标"""
    return {
        "MAE":  ...,   # 元/MWh
        "RMSE": ...,
        "MAPE": ...,   # %
        "R2":   ...,
        "方向准确率": ...,  # %
    }
```

---

## 四、Notebook 结构（推荐交付形式）

```
# 原型：纯LightGBM用户侧日前出清电价预测
# ==========================================

## Cell 1: 导入 & 配置
  - import lightgbm, pandas, numpy, matplotlib
  - 全局参数: FREQ="1h", TEST_DAYS=30

## Cell 2: Mock 数据生成
  - generate_mock_data()
  - 输出: df.head(), df.describe(), 价格分布直方图

## Cell 3: 特征工程
  - 构造特征 (net_load, renewable_penetration, ...)
  - 滞后特征 (shift操作)
  - 日期特征 (hour, weekday, ...)
  - 输出: FeatureMatrix 形状, 特征相关性热力图

## Cell 4: 时序分割
  - 训练集: 开始 ~ (结束 - 30天)
  - 测试集: 最后30天
  - 不打乱顺序

## Cell 5: 训练 LightGBM
  - train_lgbm()
  - 输出: best_iteration, best_score

## Cell 6: 评估
  - evaluate(y_test, y_pred)
  - 输出: 指标表

## Cell 7: 可视化
  - 子图1: 预测 vs 实际（测试集最后7天逐时曲线）
  - 子图2: 散点图 (y_test vs y_pred)
  - 子图3: 特征重要性 TOP10 柱状图
  - 子图4: 残差分布直方图

## Cell 8: (可选) 简单对比
  - 去掉滞后特征 → 评估 → 对比精度下降
  - 去掉构造特征 → 评估 → 对比精度下降
  - 验证每类特征的贡献
```

---

## 五、原型验收标准

在 Mock 数据上跑通后，满足以下条件即可进入真实数据阶段：

| 检查项 | 标准 | 验证方式 |
|--------|------|----------|
| 全流程跑通 | 无报错，输出评估指标 | Cell 1-7 顺序执行 |
| 模型收敛 | best_iteration < num_boost_round | 检查 early stopping 是否触发 |
| 特征合理 | 负荷/峰段/新能源的特征重要性排前 | 检查 Cell 7 的特征重要性图 |
| 方向准确率 | > 60%（Mock 数据是干净的） | 检查 Cell 6 输出 |
| MAPE | < 10%（Mock 数据是干净的） | 检查 Cell 6 输出 |
| 预测曲线合理 | 预测曲线能跟踪实际曲线的走势 | 肉眼检查 Cell 7 曲线 |

> **注意**：Mock 数据是理想化的（噪声可控、因果关系明确），所以在 Mock 上 MAPE 应该很容易 < 10%。如果在 Mock 上都做不到，说明特征设计或模型参数有问题，需要排查后再进入真实数据。

---

## 六、从原型到完整方案的升级路径

原型验证通过后，按以下顺序逐步加复杂度：

```
原型 (1文件)                → 验证 LightGBM 可行
  │
  ├─ 接入真实数据            → 替换 MockDataReader
  │
  ├─ 对比基准模型            → 加 ARIMA / XGBoost 对比
  │
  ├─ 特征扩充                → 30维完整特征集
  │
  ├─ 超参数优化              → Optuna 自动调参
  │
  ├─ 加入小波分解            → WT-LGBM（完整方案核心）
  │
  └─ 工程化部署              → 模块化代码 + 定时调度
```

---

## 七、开发排期

| 任务 | 预计耗时 | 产出 |
|------|----------|------|
| Mock 数据生成器 | 0.5天 | `generate_mock_data()` |
| 特征工程函数 | 0.5天 | 4 类 15 维特征 |
| 训练+评估+可视化 | 0.5天 | Notebook 跑通 |
| 消融实验（去掉某类特征看效果） | 0.5天 | 特征贡献分析 |

**总计：2 天**即可完成原型全流程并得出初步结论。

---

*文档结束。*
