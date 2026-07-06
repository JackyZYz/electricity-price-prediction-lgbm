# 原型架构设计：纯 LightGBM 电价预测快速验证

**版本**：v2.0
**日期**：2026-07-06
**定位**：基于 `Dataset/` 真实数据，用最简架构快速跑通"数据→特征→训练→评估"闭环

---

## 一、原型目标

一句话：**用纯 LightGBM（不做小波分解）在真实 CSV 数据上跑出基准 MAPE，验证特征有效性。**

| 对比维度 | 完整方案 (WT-LGBM) | 原型方案 (纯 LGBM) |
|----------|-------------------|---------------------|
| 模型复杂度 | 小波分解 + 多分量 LGBM | 单模型 LGBM |
| 特征数量 | ~28 维（4类） | ~15 维（精简核心） |
| 代码量 | 20+ 文件，模块化 | **3 个文件** |
| 运行方式 | 定时调度 / CLI | **单个 Jupyter Notebook 或单脚本** |
| 调参 | Optuna 自动搜索 | 手动设参 / 简单 GridSearch |
| 目标 | MAPE < 10%，上线部署 | **MAPE < 15%，验证真实数据上特征有效** |
| 依赖数据 | 真实数据（Dataset/） | **真实数据（Dataset/）** |

---

## 二、原型架构

### 2.1 极简流程

```mermaid
flowchart LR
    read["① 读取 Dataset/ 真实 CSV<br/>(宽表转长表 + 时间对齐)"] --> feat["② 特征工程<br/>(原始 + 构造 + 滞后 + 日期)"]
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

### 3.1 数据读取

原型不再使用 Mock 数据，改为直接读取 `Dataset/` 下真实 CSV。数据读取函数需完成：

1. 找到目标文件（如 `Dataset/短期系统负荷预测/短期系统负荷预测信息_出清发布电力.csv`）
2. 读取宽表，剥离前 9 列元数据
3. 将 `日期` + `00:00`~`24:00` 从宽表转为长表（melt）
4. 处理 `00:00` 列缺失：用 `24:00` 回填或删除
5. 合并所有特征到同一时间索引

```python
def load_target_price(dataset_root: str) -> pd.DataFrame:
    """读取目标变量：统一结算点电价最终结果"""
    pass

def load_feature_table(dataset_root: str, relative_path: str, value_name: str) -> pd.DataFrame:
    """读取单个特征宽表，返回长表 DataFrame[datetime, value_name]"""
    pass
```

### 3.2 精简特征集

原型只取最核心、能直接从真实数据生成的特征（~15 维）：

```python
FEATURE_CONFIG = {
    # ---- 直接特征（6个） ----
    "sys_load_pred":       "短期系统负荷预测",
    "wind_power_pred":     "统调风电功率预测（地区汇总）",
    "solar_power_pred":    "统调光电功率预测（地区汇总）",
    "power_import":        "受电计划（华东）",
    "coal_gen":            "煤电发电计划（地区汇总）",
    "gas_gen":             "燃机固定出力总值（地区汇总）",

    # ---- 构造特征（4个） ----
    "net_load":            "净负荷 = 负荷 - 风光",
    "renewable_penetration":"新能源渗透率",
    "import_ratio":        "外来电占比",
    "thermal_ratio":       "火电占比 = (煤+燃)/负荷",

    # ---- 滞后特征（2个） ----
    "price_lag_1d":        "昨日同时段电价",
    "price_lag_7d":        "上周同时段电价",

    # ---- 日期特征（4个） ----
    "hour":                "0-23",
    "weekday":             "0-6",
    "is_weekend":          "0/1",
    "month":               "1-12",
}
```

> 储能计划、正负备用等特征在原型中暂不纳入，以先跑通主流程；后续完整方案中逐步加入。

### 3.3 时序分割

```python
def time_series_split(df: pd.DataFrame, test_days: int = 30):
    """
    按日期顺序切分训练集/测试集，不打乱。
    由于数据为 15 分钟粒度（每日 96 点），test_days=30 对应 2880 条样本。
    """
    pass
```

### 3.4 模型训练

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
  - 全局参数: DATASET_ROOT="../Dataset", FREQ="15min", TEST_DAYS=30

## Cell 2: 读取真实数据
  - load_target_price()
  - load_feature_table() for each feature
  - 输出: 目标变量 describe(), 价格分布直方图

## Cell 3: 特征工程
  - 构造特征 (net_load, renewable_penetration, ...)
  - 滞后特征 (shift操作，96/672 点)
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
  - 子图1: 预测 vs 实际（测试集最后3天逐15分钟曲线）
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

在真实数据上跑通后，满足以下条件即可进入完整方案阶段：

| 检查项 | 标准 | 验证方式 |
|--------|------|----------|
| 全流程跑通 | 无报错，输出评估指标 | Cell 1-7 顺序执行 |
| 数据对齐正确 | 训练/测试样本数合理，无未来信息泄露 | 检查特征矩阵时间索引 |
| 模型收敛 | best_iteration < num_boost_round | 检查 early stopping 是否触发 |
| 特征合理 | 负荷/峰段/新能源的特征重要性排前 | 检查 Cell 7 的特征重要性图 |
| 方向准确率 | > 60% | 检查 Cell 6 输出 |
| MAPE | < 15% | 检查 Cell 6 输出 |
| 预测曲线合理 | 预测曲线能跟踪实际曲线的走势 | 肉眼检查 Cell 7 曲线 |

> **注意**：真实数据存在噪声、缺失、尖峰，MAPE < 15% 是合理的原型目标。如果达不到，需优先检查特征对齐、缺失处理、数据泄露。

---

## 六、从原型到完整方案的升级路径

原型验证通过后，按以下顺序逐步加复杂度：

```
原型 (1文件)                → 验证 LightGBM 在真实数据上可行
  │
  ├─ 特征扩充                → 28维完整特征集（储能、备用、分区、更多滞后）
  │
  ├─ 对比基准模型            → 加 ARIMA / XGBoost 对比
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
