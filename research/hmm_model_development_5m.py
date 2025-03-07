# %%
# %reset -f

# %%
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from probabilistic_trading.models.hmm import HMMConfig
from probabilistic_trading.models.hmm import HMMModel
from probabilistic_trading.models.hmm import TrainingConfig


# %%
# Load data
bars_ = pl.read_parquet(
    "../data/binance/futures_processed/LTC_USDT_USDT-15m-futures-processed.parquet"
)
# print(bars_.tail())
bars = bars_.slice(0, 30000)
# print(bars.describe())

# %%
n_states = 3
n_features = 6
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
train_features_pca = features_pca[:5000]

# 首次訓練
print("Initial training of HMM model...")
model.fit(
    train_features_pca.reshape(1, -1, n_features),
    training_config=TrainingConfig(method="em", num_epochs=50),
)
print("Initial training completed")

# %%
sample_states = model.predict(train_features_pca)  # 取全部時間步的狀態
sample_probas = model.predict_proba(train_features_pca)  # [-1]  # 取最後一個時間步的狀態機率

# %%
sample_states = np.array(sample_states, dtype=np.int32)
sample_states = sample_states.reshape(-1)
sample_state_proba = np.array(
    [sample_probas[i, sample_states[i]] for i in range(len(sample_states))], dtype=np.float32
)
# 將狀態和機率添加到DataFrame
sample_prediction = pl.DataFrame(
    {
        "states": sample_states,
        "state_proba": sample_state_proba,
    }
)
train_bars = bars.slice(0, 5000)
train_bars = train_bars.hstack(sample_prediction)
# train_bars = bars.hstack(sample_prediction)

# %%
# print(train_bars.tail())
print(train_bars.columns)

# %%
# pl.DataFrame.group_by
summary = train_bars.group_by("states").agg(
    [
        pl.col("log_returns_1").cum_sum().exp().last().alias("long_returns"),
        (pl.col("log_returns_1") * -1).cum_sum().exp().last().alias("short_returns"),
        pl.col("log_returns_1").max().alias("max_log_returns"),
        pl.col("log_returns_1").mean().alias("mean_log_returns"),
        pl.col("log_returns_1").min().alias("min_log_returns"),
        pl.col("volatility_15").quantile(0.98, interpolation="higher").alias("max_volatility"),
        pl.col("volatility_15").mean().alias("mean_volatility"),
        pl.col("volume_15_momentum").mean().alias("mean_volume_15_momentum"),
        pl.col("volume_15_ma").mean().alias("mean_volume_15_ma"),
        pl.col("states").count().alias("bars_in_state"),
    ]
)

# 迭代輸出結果
for row in summary.iter_rows(named=True):
    print(f"State {row['states']}:")
    print("long_returns:", row["long_returns"])
    print("short_returns:", row["short_returns"])
    print("max_log_returns:", row["max_log_returns"])
    print("mean_log_returns:", row["mean_log_returns"])
    print("min_log_returns:", row["min_log_returns"])
    print("max_volatility:", row["max_volatility"])
    print("mean_volatility:", row["mean_volatility"])
    print("mean_volume_15_momentum:", row["mean_volume_15_momentum"])
    print("mean_volume_15_ma:", row["mean_volume_15_ma"])
    print("Bars in this state:", row["bars_in_state"])
    print("-" * 40)

# %%
# 繪圖
fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# 繪製價格走勢
ax[0].plot(train_bars["timestamp"], train_bars["close"], label="Price")
ax[0].plot(train_bars["timestamp"], train_bars["vwap_15"], label="Price")
ax[0].set_title("Close Price with HMM States")

# 定義顏色對應 (假設最多 5 種 state)
colors = ["#FFBBBB", "#BBFFBB", "#BBBBFF", "#FFBBFF", "#FFFFBB"]

# 使用 polars 遍歷 DataFrame 來標示狀態
for timestamp, state in zip(train_bars["timestamp"].to_list(), train_bars["states"].to_list()):
    ax[0].axvspan(timestamp, timestamp, color=colors[state % len(colors)], alpha=0.5)

# 繪製 HMM 狀態變化
ax[1].plot(train_bars["timestamp"], train_bars["states"], label="State")
ax[1].set_yticks(range(train_bars["states"].n_unique()))
ax[1].set_title("Hidden States")

plt.tight_layout()
plt.show()

# %%
# 計算策略報酬
train_bars = train_bars.with_columns(
    (
        pl.col("log_returns_1")
        * pl.when(pl.col("states") == 0)
        # 波動度越大 => 杠桿越低
        .then(3 * (0.01 / (1e-8 + pl.col("volatility_15"))))
        .when(pl.col("states") == 1)
        .then(3 * (0.01 / (1e-8 + pl.col("volatility_15"))))
        .when(pl.col("states") == 2)
        .then(-1)
        # .when(pl.col("states") == 3)
        # .then(2 * (0.01 / (1e-8 + pl.col("volatility_15"))))
        # .when(pl.col("states") == 4)
        # .then(2 * (0.01 / (1e-8 + pl.col("volatility_15"))))
        # .otherwise(
        #     -3 * (0.01 / (1e-8 + pl.col("volatility_15")))
        # )
    ).alias("strategy_return")
)

# 計算累積報酬
cumulative_strategy = train_bars["strategy_return"].cum_sum().exp()
cumulative_buy_and_hold = train_bars["log_returns_1"].cum_sum().exp()

# 繪圖
plt.figure(figsize=(12, 6))
plt.plot(train_bars["timestamp"], cumulative_strategy, label="HMM-based strategy")
plt.plot(train_bars["timestamp"], cumulative_buy_and_hold, label="Buy and Hold")
plt.title("Cumulative Returns Comparison")
plt.legend()
plt.show()


# %%
