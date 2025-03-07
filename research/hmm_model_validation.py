# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
# %reset -f

# %%
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from probabilistic_trading.models.hmm import HMMConfig
from probabilistic_trading.models.hmm import HMMModel
from probabilistic_trading.models.hmm import TrainingConfig


# %%
# Load data
bars_ = pl.read_parquet(
    "../data/binance/futures_processed/ADA_USDT_USDT-5m-futures-processed.parquet"
)
print(bars_.describe())
bars = bars_.slice(0, 90000)
# print(bars.tail())
# print(bars.describe())
# %%
n_states = 3
n_splits = 10
n_features = 6
tscv = TimeSeriesSplit(n_splits=n_splits)
scaler = StandardScaler()
pca = PCA(n_components=n_features)
# %%
# 計算基礎特徵 (Batch operations together)
features_ = []
for i in range(1, 72, 14):
    features_.extend(
        [
            (
                (
                    ((pl.col("close") + pl.col("high") + pl.col("low")) / 3) * pl.col("volume")
                ).rolling_sum(i)
                / pl.col("volume").rolling_sum(i)
            )
            .fill_null(1e-8)
            .alias(f"vwap_{i}"),
            # (pl.col("close") - pl.col("close").shift(i))
            # .fill_null(1e-8)
            # .alias(f"close_momentum_{i}"),
            # ((pl.col("close") - pl.col("close").shift(1)) / pl.col("close").shift(i))
            # .fill_null(1e-8)
            # .alias(f"returns_{i}"),
            (pl.col("close").log() - pl.col("close").shift(i).log())
            .fill_null(1e-8)
            .alias(f"log_returns_{i}"),
            pl.col("volume").rolling_mean(i).fill_null(1e-8).alias(f"volume_{i}_ma"),
            (pl.col("volume") - pl.col("volume").shift(i))
            .fill_null(1e-8)
            .alias(f"volume_{i}_momentum"),
            ((pl.col("close") - pl.col("close").shift(1)) / pl.col("close").shift(i))
            .rolling_std(i)
            .fill_null(1e-8)
            .alias(f"volatility_{i}"),
        ]
    )

# Apply all feature calculations at once
bars = bars.with_columns(features_)
bars = bars.slice(120, bars.height)

# 選擇用於建模的特徵 (Faster filtering)
features = bars.select(
    [
        col
        for col in bars.columns
        if col not in ["timestamp", "open", "high", "low", "close", "volume"]
    ]
)

features.to_numpy()
np.isnan(features).any()

# %%
try:
    # 檢查無效值
    if np.isnan(features).any():
        print("Features contain NaN values")

    if np.isinf(features).any():
        print("Features contain infinite values")

    # 標準化 (使用scikit-learn)
    features_scaled = scaler.fit_transform(features)

    # 檢查標準化後的數據
    if np.isnan(features_scaled).any() or np.isinf(features_scaled).any():
        print("Scaling produced invalid values")

    # PCA降維 (仍使用sklearn)
    features_pca = pca.fit_transform(features_scaled)

    # 驗證PCA結果
    if np.isnan(features_pca).any() or np.isinf(features_pca).any():
        print("PCA produced invalid values")

    # 記錄解釋方差比
    explained_variance_ratio = np.array(pca.explained_variance_ratio_)
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    print(f"PCA explained variance ratio: {cumulative_variance_ratio[-1]:.4f}")


except Exception as e:
    print(f"Feature extraction failed: {e!s}")
    print("Stack trace:")

# %%
model = HMMModel(
    config=HMMConfig(
        n_states=n_states,  # 隱藏狀態數量
        emission_dim=n_features,  # 使用PCA降維後的特徵
        emission_type="gaussian",
    )
)

# %%
# 儲存每個測試集的結果
all_test_results = []
# 遍歷 TimeSeriesSplit 進行交叉驗證
for fold_idx, (train_index, test_index) in enumerate(tscv.split(features_pca)):
    print(f"\n===== Fold {fold_idx + 1} =====")
    X_train, X_test = features_pca[train_index], features_pca[test_index]

    # Reshape for HMM
    X_train = X_train.reshape(1, -1, n_features)

    # 訓練 HMM
    print("Training HMM model...")
    model.fit(X_train, training_config=TrainingConfig(method="em", num_epochs=50))
    print("HMM training completed.")

    # 預測測試集的狀態
    states = model.predict(X_test)
    probas = model.predict_proba(X_test)

    # 存儲狀態與機率
    test_states = np.array(states, dtype=np.int32).reshape(-1)
    test_state_proba = np.array(
        [probas[i, test_states[i]] for i in range(len(test_states))], dtype=np.float32
    )

    # 建立 DataFrame
    test_results = pl.DataFrame(
        {
            "timestamp": bars["timestamp"].to_numpy()[test_index],
            "states": test_states,
            "state_proba": test_state_proba,
            "close": bars["close"].to_numpy()[test_index],
            "log_returns_1": bars["log_returns_1"].to_numpy()[test_index],
            "volatility_15": bars["volatility_15"].to_numpy()[test_index],
            "volume_15_ma": bars["volume_15_ma"].to_numpy()[test_index],
            "vwap_15": bars["vwap_15"].to_numpy()[test_index],
        }
    )
    all_test_results.append(test_results)

    # **統計該時間區段不同狀態的分布**
    summary = test_results.group_by("states").agg(
        [
            pl.col("log_returns_1").cum_sum().exp().last().alias("long_return"),
            (pl.col("log_returns_1") * -1).cum_sum().exp().last().alias("short_return"),
            pl.col("log_returns_1").max().alias("max_log_returns"),
            pl.col("log_returns_1").mean().alias("mean_log_returns"),
            pl.col("log_returns_1").min().alias("min_log_returns"),
            pl.col("volatility_15").quantile(0.98, interpolation="higher").alias("max_volatility"),
            pl.col("volatility_15").mean().alias("mean_volatility"),
            pl.col("volume_15_ma").mean().alias("mean_volume_15_ma"),
            pl.col("state_proba").mean().alias("mean_proba"),
            pl.col("states").count().alias("bars_in_state"),
        ]
    )

    for row in summary.iter_rows(named=True):
        print(f"State {row['states']}:")
        print("long_return:", row["long_return"])
        print("short_return:", row["short_return"])
        print("max_log_returns:", row["max_log_returns"])
        print("mean_log_returns:", row["mean_log_returns"])
        print("min_log_returns:", row["min_log_returns"])
        print("max_volatility:", row["max_volatility"])
        print("mean_volatility:", row["mean_volatility"])
        print("mean_volume_15_ma:", row["mean_volume_15_ma"])
        print("Bars in this state:", row["bars_in_state"])
        print("-" * 40)

    # 輸出狀態統計
    unique_states, counts = np.unique(test_states, return_counts=True)
    for state, count in zip(unique_states, counts):
        print(f"State {state}: {count} bars")

# 合併所有測試結果
final_test_results = pl.concat(all_test_results)
# %%
# 繪製不同時間窗口的 HMM 預測狀態
fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# **測試集價格 vs 預測狀態**
for fold_idx, test_result in enumerate(all_test_results):
    ax[0].plot(test_result["timestamp"], test_result["states"], label=f"Test Window {fold_idx+1}")

ax[0].set_title("HMM Hidden States in Different Time Windows")
ax[0].set_ylabel("HMM States")
ax[0].legend()

# **狀態機率**
for fold_idx, test_result in enumerate(all_test_results):
    ax[1].plot(
        test_result["timestamp"], test_result["state_proba"], label=f"Test Window {fold_idx+1}"
    )

ax[1].set_title("HMM State Probability in Different Time Windows")
ax[1].set_ylabel("Probability")
ax[1].set_xlabel("Timestamp")
ax[1].legend()

plt.tight_layout()
plt.show()

# %%
# 設定顏色對應不同的 HMM 隱藏狀態
colors = ["#FFBBBB", "#BBFFBB", "#BBBBFF", "#FFBBFF", "#FFFFBB"]

# 設定畫布大小, 每個 fold 兩張圖(價格走勢 + 累積報酬)
fig, axes = plt.subplots(n_splits, 2, figsize=(12, 5 * n_splits))

for fold_idx, test_result in enumerate(all_test_results):
    ax_price = axes[fold_idx, 0]
    ax_return = axes[fold_idx, 1]

    mask = bars["timestamp"].is_in(test_result["timestamp"])

    ax_price.plot(bars.filter(mask)["timestamp"], bars.filter(mask)["close"], label="Price")
    ax_price.plot(bars.filter(mask)["timestamp"], bars.filter(mask)["vwap_15"], label="VWAP_15")

    # 標記 HMM 狀態變更
    for i in range(1, len(test_result)):
        if test_result["states"][i] != test_result["states"][i - 1]:  # 狀態變更點
            ax_price.axvspan(
                test_result["timestamp"][i - 1],
                test_result["timestamp"][i],
                color=colors[test_result["states"][i] % len(colors)],
                alpha=0.5,
            )

    ax_price.set_title(f"Fold {fold_idx+1}: Close Price with HMM States")
    ax_price.legend()

    # 計算策略報酬
    test_result = test_result.with_columns(
        (
            pl.col("log_returns_1")
            * pl.when(pl.col("states") == 1)
            # 波動度越大 => 杠桿越低
            .then(3 * (0.01 / (1e-8 + pl.col("volatility_15"))))
            .when(pl.col("states") == 0).then(-3 * (0.01 / (1e-8 + pl.col("volatility_15"))))
            .when(pl.col("states") == 2)
            .then(0)
        ).alias("strategy_return")
    )

    # **計算 HMM 策略 vs. Buy & Hold 累積報酬**
    hmm_strategy = test_result["strategy_return"].cum_sum().exp()
    buy_hold = test_result["log_returns_1"].cum_sum().exp()

    # 繪製累積報酬
    ax_return.plot(test_result["timestamp"], hmm_strategy, label="HMM-based strategy")
    ax_return.plot(test_result["timestamp"], buy_hold, label="Buy and Hold")

    ax_return.set_title(f"Fold {fold_idx+1}: Cumulative Returns Comparison")
    ax_return.legend()

plt.tight_layout()
plt.show()

# %%
