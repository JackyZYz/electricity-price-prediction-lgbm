#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
原型：纯 LightGBM 用户侧日前出清电价预测
=========================================
定位：在完整 WT-LGBM 方案之前，用最简架构快速验证思路可行性。
运行：python prototype_lgbm.py
依赖：pip install lightgbm pandas numpy matplotlib seaborn scikit-learn
"""

import warnings
from datetime import datetime, timedelta

import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')  # 无 GUI 环境强制用文件后端，避免 Tcl/Tk 报错
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ============================================================
# 全局配置（修改这里的参数来调整原型行为）
# ============================================================
CONFIG = {
    "start_date": "2024-01-01",
    "end_date": "2025-06-30",
    "freq": "1h",             # "1h" = 24点/天, "15min" = 96点/天
    "test_days": 30,
    "seed": 42,
    "output_dir": "./prototype_output/",
}

np.random.seed(CONFIG["seed"])

POINTS_PER_DAY = 24 if CONFIG["freq"] == "1h" else 96

print("=" * 60)
print("  纯 LightGBM 用户侧日前出清电价预测 - 原型验证")
print("=" * 60)
print(f"  数据范围: {CONFIG['start_date']} ~ {CONFIG['end_date']}")
print(f"  时间粒度: {CONFIG['freq']} ({POINTS_PER_DAY}点/天)")
print(f"  测试集:   最近 {CONFIG['test_days']} 天")
print("=" * 60)

# ============================================================
# Cell 1: Mock 数据生成（约 100 行）
# ============================================================
print("\n[1/7] 生成 Mock 数据...")


def generate_mock_data(start_date, end_date, freq="1h", seed=42):
    """
    生成带因果关系的模拟电力市场数据。

    电价生成逻辑：
        price = 350                                           # 基础电价
              + 150 * (load - load_mean) / load_std           # 负荷因子
              - 100 * renewable_penetration                   # 新能源因子
              + 80  * is_peak                                # 峰段溢价
              + 50  * sin(2π * (month-1)/12)                  # 季节因子
              + noise                                         # N(0, 25)

    这确保了特征和目标是可学习的——LGBM 应该能发现这些规律。
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start_date, end_date, freq=freq, inclusive="left")
    n = len(timestamps)
    hours = timestamps.hour.astype(float)
    months = timestamps.month.astype(float)
    weekdays = timestamps.weekday.astype(float)

    # --- 辅助函数 ---
    def smooth(series, window=None):
        """给序列加一点平滑，避免太突兀"""
        if window is None:
            window = POINTS_PER_DAY // 4
        return pd.Series(series).rolling(window, center=True, min_periods=1).mean().values

    # --- 系统负荷（MW）：双峰日曲线 + 周末效应 + 季节 ---
    # 日曲线：凌晨低 → 上午爬升 → 午后微降 → 晚高峰 → 夜间回落
    hour_rad = 2 * np.pi * hours / 24
    daily_shape = (
        0.65
        + 0.15 * np.sin(hour_rad - np.pi / 3)
        + 0.20 * np.sin(2 * hour_rad - np.pi / 4)
    )
    load_base = 50000  # MW
    # 周末负荷降低 10%
    weekend_factor = np.where(weekdays >= 5, 0.90, 1.00)
    # 季节性：夏季和冬季更高
    season_factor = 1.0 + 0.08 * np.sin(2 * np.pi * (months - 1) / 12 - np.pi / 2)
    load = load_base * daily_shape * weekend_factor * season_factor
    load += rng.normal(0, load_base * 0.03, n)  # 3% 噪声
    load = smooth(load)

    # --- 风电功率（MW）：随机游走 + 季节 + 日间略高 ---
    wind_base = 8000
    wind_season = 1.0 + 0.3 * np.sin(2 * np.pi * (months - 1) / 12)  # 冬季风大
    wind_daily = 1.0 + 0.05 * np.sin(hour_rad)  # 日间略高
    wind_random = np.cumsum(rng.normal(0, 0.03, n))  # 随机游走
    wind_random = (wind_random - wind_random.mean()) / wind_random.std() * 0.15
    wind = wind_base * wind_season * wind_daily * (1 + wind_random)
    wind = np.clip(wind, 0, None)
    wind = smooth(wind)

    # --- 光伏功率（MW）：强日周期，夜间归零 ---
    solar_base = 6000
    # 白天：sin 曲线形状，夜间为 0
    solar_envelope = np.maximum(0, np.sin(np.pi * (hours - 6) / 12))
    solar_envelope = np.where((hours >= 6) & (hours <= 18), solar_envelope, 0)
    solar_season = 1.0 + 0.4 * np.sin(2 * np.pi * (months - 5) / 12)  # 夏季高
    cloud_factor = 1.0 + rng.normal(0, 0.2, n)  # 云层随机
    solar = solar_base * solar_envelope * solar_season * cloud_factor
    solar = np.clip(solar, 0, None)
    solar = smooth(solar)

    # --- 受电计划（MW）：稳定基值 + 小幅波动 ---
    import_power = 10000 + rng.normal(0, 500, n)
    import_power = smooth(import_power)

    # --- 煤电出力（MW）：与负荷正相关 ---
    coal = 20000 + 0.25 * (load - load_base) + rng.normal(0, 800, n)
    coal = smooth(coal)
    coal = np.clip(coal, 5000, None)

    # --- 燃机出力（MW）：峰段抬升 ---
    gas_base = 5000
    peak_mask = ((hours >= 8) & (hours <= 11)) | ((hours >= 17) & (hours <= 21))
    gas = gas_base + peak_mask.astype(float) * 3000 + rng.normal(0, 400, n)
    gas = smooth(gas)
    gas = np.clip(gas, 0, None)

    # --- 储能计划（MW）：谷段充电(负)、峰段放电(正) ---
    valley_mask = (hours >= 0) & (hours <= 6)
    storage = np.where(valley_mask, -1500.0, np.where(peak_mask, 1500.0, 200.0))
    storage += rng.normal(0, 300, n)
    storage = smooth(storage)

    # --- ★ 用户侧日前出清电价（目标变量）---
    load_std = np.std(load)
    renewable_penetration = (wind + solar) / (load + 1)
    base_price = 350
    load_factor = 150 * (load - load.mean()) / load_std
    renewable_factor = -100 * renewable_penetration
    peak_factor = 80 * peak_mask.astype(float)
    season_factor_price = 50 * np.sin(2 * np.pi * (months - 1) / 12)
    noise = rng.normal(0, 25, n)
    user_price = (
        base_price + load_factor + renewable_factor + peak_factor + season_factor_price + noise
    )
    user_price = np.clip(user_price, 50, 800)  # 合理价格区间

    # --- 组装 DataFrame ---
    df = pd.DataFrame(
        {
            "datetime": timestamps,
            "sys_load_pred": load,
            "wind_power_pred": wind,
            "solar_power_pred": solar,
            "power_import": import_power,
            "coal_gen": coal,
            "gas_gen": gas,
            "storage_plan": storage,
            "user_price": user_price,
        }
    )
    df.set_index("datetime", inplace=True)
    return df


df = generate_mock_data(
    CONFIG["start_date"], CONFIG["end_date"], CONFIG["freq"], CONFIG["seed"]
)

print(f"  ✓ 生成数据: {len(df)} 行, {len(df.columns)} 列")
print(f"  ✓ 日期范围: {df.index[0]} ~ {df.index[-1]}")
print(f"  ✓ 目标变量均值: {df['user_price'].mean():.1f} 元/MWh")
print(f"  ✓ 目标变量标准差: {df['user_price'].std():.1f} 元/MWh")
print(f"  ✓ 价格范围: [{df['user_price'].min():.0f}, {df['user_price'].max():.0f}] 元/MWh")

# ============================================================
# Cell 2: 特征工程（约 80 行）
# ============================================================
print("\n[2/7] 特征工程...")


def build_features(df):
    """构建精简的 15 维特征集"""
    f = pd.DataFrame(index=df.index)

    # --- 2a. 直接特征（6个）---
    f["sys_load_pred"] = df["sys_load_pred"]
    f["wind_power_pred"] = df["wind_power_pred"]
    f["solar_power_pred"] = df["solar_power_pred"]
    f["power_import"] = df["power_import"]
    f["coal_gen"] = df["coal_gen"]
    f["gas_gen"] = df["gas_gen"]

    # --- 2b. 构造特征（4个）---
    f["net_load"] = df["sys_load_pred"] - df["wind_power_pred"] - df["solar_power_pred"]
    f["renewable_penetration"] = (df["wind_power_pred"] + df["solar_power_pred"]) / (
        df["sys_load_pred"] + 1
    )
    f["import_ratio"] = df["power_import"] / (df["sys_load_pred"] + 1)
    f["thermal_ratio"] = (df["coal_gen"] + df["gas_gen"]) / (df["sys_load_pred"] + 1)

    # --- 2c. 滞后特征（3个）------
    price = df["user_price"]
    points_per_day = POINTS_PER_DAY
    f["price_lag_1d"] = price.shift(points_per_day)
    f["price_lag_7d"] = price.shift(7 * points_per_day)
    f["price_ma_24h"] = price.shift(1).rolling(points_per_day, min_periods=1).mean()

    # --- 2d. 日期特征（4个）---
    f["hour"] = df.index.hour
    f["weekday"] = df.index.weekday
    f["is_weekend"] = (df.index.weekday >= 5).astype(int)
    f["month"] = df.index.month

    # 收集特征名和目标
    feature_names = list(f.columns)
    target = df["user_price"].values

    # 删除因 shift 产生的 NaN 行
    mask = ~f.isna().any(axis=1)
    f = f[mask]
    target = target[mask]

    print(f"  ✓ 特征维度: {f.shape[1]}")
    print(f"  ✓ 有效样本: {f.shape[0]} (去除 NaN 后)")
    print(f"  ✓ 特征列表: {feature_names}")
    return f.values, target, f.index, feature_names


X, y, timestamps, feature_names = build_features(df)

# ============================================================
# Cell 3: 时序分割（约 15 行）
# ============================================================
print("\n[3/7] 时序分割...")

test_points = CONFIG["test_days"] * POINTS_PER_DAY
split_idx = len(X) - test_points

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
t_train, t_test = timestamps[:split_idx], timestamps[split_idx:]

print(f"  ✓ 训练集: {len(X_train)} 样本 ({t_train[0]} ~ {t_train[-1]})")
print(f"  ✓ 测试集: {len(X_test)} 样本 ({t_test[0]} ~ {t_test[-1]})")

# ============================================================
# Cell 4: 训练 LightGBM（约 40 行）
# ============================================================
print("\n[4/7] 训练 LightGBM...")

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 10,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.01,
    "lambda_l2": 0.01,
    "verbosity": -1,
    "seed": CONFIG["seed"],
}

train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
valid_data = lgb.Dataset(
    X_test, label=y_test, reference=train_data, feature_name=feature_names
)

callbacks = [
    lgb.early_stopping(50, verbose=False),
    lgb.log_evaluation(period=200),
]

model = lgb.train(
    params=params,
    train_set=train_data,
    valid_sets=[valid_data],
    valid_names=["valid"],
    num_boost_round=2000,
    callbacks=callbacks,
)

y_pred = model.predict(X_test)
print(f"  ✓ Best iteration: {model.best_iteration}")
print(f"  ✓ Best valid RMSE: {model.best_score['valid']['rmse']:.2f}")

# ============================================================
# Cell 5: 评估（约 35 行）
# ============================================================
print("\n[5/7] 评估...")


def evaluate(y_true, y_pred):
    """计算全部核心指标"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE（对接近零的值做保护）
    mask = np.abs(y_true) > 1
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    r2 = r2_score(y_true, y_pred)
    # 方向准确率
    actual_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    dir_acc = np.mean(actual_dir == pred_dir) * 100
    return {
        "MAE (元/MWh)": round(mae, 2),
        "RMSE (元/MWh)": round(rmse, 2),
        "MAPE (%)": round(mape, 2),
        "R²": round(r2, 4),
        "方向准确率 (%)": round(dir_acc, 2),
    }


metrics = evaluate(y_test, y_pred)
for k, v in metrics.items():
    print(f"  {k}: {v}")

# 验收判断
mape_ok = metrics["MAPE (%)"] < 15
dir_ok = metrics["方向准确率 (%)"] > 60
print(f"\n  验收: MAPE < 15% {'✓' if mape_ok else '✗'} | 方向准确率 > 60% {'✓' if dir_ok else '✗'}")
if mape_ok and dir_ok:
    print("  ✅ 原型验证通过！可以进入真实数据阶段。")
else:
    print("  ⚠️ 指标未达标，请检查特征设计或模型参数。")

# ============================================================
# Cell 6: 特征重要性
# ============================================================
print("\n[6/7] 特征重要性...")

importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": model.feature_importance(importance_type="gain"),
    }
).sort_values("importance", ascending=False)

print(importance_df.to_string(index=False))

# ============================================================
# Cell 7: 可视化（4张子图）
# ============================================================
print("\n[7/7] 生成可视化...")
import os

output_dir = CONFIG["output_dir"]
os.makedirs(output_dir, exist_ok=True)

# --- 设置中文字体（Windows 用 SimHei 或微软雅黑）---
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    f"纯 LightGBM 电价预测原型验证 | MAPE={metrics['MAPE (%)']:.1f}% | "
    f"方向准确率={metrics['方向准确率 (%)']:.1f}%",
    fontsize=15,
    fontweight="bold",
)

# --- 子图1: 预测 vs 实际（测试集最后 7 天）---
ax1 = axes[0, 0]
plot_points = 7 * POINTS_PER_DAY
ax1.plot(
    range(plot_points),
    y_test[-plot_points:],
    label="实际电价",
    linewidth=1.2,
    alpha=0.85,
)
ax1.plot(
    range(plot_points),
    y_pred[-plot_points:],
    label="预测电价",
    linewidth=1.2,
    alpha=0.85,
    linestyle="--",
)
ax1.set_title("测试集最后 7 天：预测 vs 实际", fontsize=12)
ax1.set_xlabel("时段序号")
ax1.set_ylabel("电价 (元/MWh)")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)

# --- 子图2: 散点图 ---
ax2 = axes[0, 1]
ax2.scatter(y_test, y_pred, alpha=0.3, s=8, edgecolors="none")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax2.plot(lims, lims, "r--", linewidth=1, label="y=x")
ax2.set_title("预测值 vs 实际值", fontsize=12)
ax2.set_xlabel("实际电价 (元/MWh)")
ax2.set_ylabel("预测电价 (元/MWh)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- 子图3: 特征重要性 TOP10 ---
ax3 = axes[1, 0]
top10 = importance_df.head(10)
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top10)))
ax3.barh(range(len(top10)), top10["importance"].values, color=colors[::-1])
ax3.set_yticks(range(len(top10)))
ax3.set_yticklabels(top10["feature"].values[::-1])
ax3.set_title("特征重要性 TOP10 (gain)", fontsize=12)
ax3.set_xlabel("Importance (gain)")
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis="x")

# --- 子图4: 残差分布 ---
ax4 = axes[1, 1]
residuals = y_test - y_pred
ax4.hist(residuals, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
ax4.axvline(0, color="red", linestyle="--", linewidth=1.2)
ax4.set_title(f"残差分布 (μ={residuals.mean():.1f}, σ={residuals.std():.1f})", fontsize=12)
ax4.set_xlabel("残差 (元/MWh)")
ax4.set_ylabel("频次")
ax4.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
save_path = os.path.join(output_dir, "prototype_results.png")
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"  ✓ 图表已保存到: {save_path}")
plt.close()

print("\n" + "=" * 60)
print("  原型验证完成！")
print(f"  输出目录: {os.path.abspath(output_dir)}")
print("=" * 60)
