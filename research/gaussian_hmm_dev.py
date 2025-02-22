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
import logging
from collections.abc import Sequence
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmm_plotting import plot_gaussian_hmm  # noqa
from hmm_plotting import plot_gaussian_hmm_data  # noqa
from hmm_plotting import plot_training_progress
from hmm_plotting import plot_validation_results
from jax import vmap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from probabilistic_trading.model.hmm.hmm_model import HMMConfig
from probabilistic_trading.model.hmm.hmm_model import HMMModel
from probabilistic_trading.model.hmm.hmm_model import TrainingConfig


# %%
ada_perp_ = pd.read_parquet(
    "../data/binance/futures_processed/ADA_USDT_USDT-15m-futures-processed.parquet"
)
print(ada_perp_.shape)
print(ada_perp_.tail())

# ada_perp = ada_perp_.to_numpy()
ada_perp = ada_perp_.iloc[: 288 * 7, 0:5].copy()
print(ada_perp.shape)
print(ada_perp.tail())


# %% [markdown]
# ### Helper functions
#


# %%
def plot_hmm_states_with_data(
    data: pd.DataFrame,
    states: Sequence[int],
    start_idx: int = 0,
    title: str = "Hidden States and Price Data",
) -> None:
    """
    繪製HMM隱藏狀態與價格數據的可視化圖表。
    使用背景顏色表示隱藏狀態, 並疊加價格數據的折線圖。

    Parameters
    ----------
    data : pd.DataFrame
        原始價格數據, 包含 open, high, low, close, volume 列
    states : Sequence[int]
        HMM預測的隱藏狀態序列
    start_idx : int
        資料的起始索引, 用於對齊狀態和數據
    title : str
        圖表標題
    """
    # 設置子圖
    fig, axs = plt.subplots(5, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(title, fontsize=14)

    # 定義顏色映射
    cmap = plt.cm.get_cmap("Set3")
    n_states = len(np.unique(states))
    colors = [cmap(i) for i in np.linspace(0, 1, n_states)]

    # 獲取時間戳
    timestamps = pd.to_datetime(data.index[start_idx:])

    # 繪製每個數據列的圖表
    columns = ["open", "high", "low", "close", "volume"]
    for i, col in enumerate(columns):
        # 繪製狀態背景
        for state in range(n_states):
            mask = states == state
            axs[i].fill_between(
                timestamps[mask],
                data[col].iloc[start_idx:][mask].min(),
                data[col].iloc[start_idx:][mask].max(),
                alpha=0.3,
                color=colors[state],
                label=f"State {state}" if i == 0 else "",
            )

        # 繪製數據線
        axs[i].plot(
            timestamps,
            data[col].iloc[start_idx:],
            color="black",
            linewidth=1,
            label=col.capitalize(),
        )

        axs[i].set_ylabel(col.capitalize())
        axs[i].grid(True, alpha=0.3)
        axs[i].legend()

    # 調整x軸標籤和布局
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# %%
# def calculate_state_statistics(
#     data: pd.DataFrame,
#     states: Sequence[int],
#     start_idx: int = 0
# ) -> pd.DataFrame:
#     """
#     計算每個隱藏狀態下的統計數據。

#     Parameters
#     ----------
#     data : pd.DataFrame
#         原始價格數據
#     states : Sequence[int]
#         HMM預測的隱藏狀態序列
#     start_idx : int
#         資料的起始索引

#     Returns
#     -------
#     pd.DataFrame
#         包含每個狀態統計數據的DataFrame
#     """
#     # 計算收益率
#     returns = data['close'].pct_change()

#     # 初始化統計結果字典
#     stats = {}

#     # 對每個狀態進行統計
#     for state in np.unique(states):
#         state_mask = states == state
#         state_returns = returns.iloc[start_idx:][state_mask]

#         # 計算各項統計數據
#         stats[f'State {state}'] = {
#             'Count': sum(state_mask),
#             'Return Mean (%)': state_returns.mean() * 100,
#             'Return Std (%)': state_returns.std() * 100,
#             'Return Max (%)': state_returns.max() * 100,
#             'Return Min (%)': state_returns.min() * 100,
#             'Winning Rate (%)': (state_returns > 0).mean() * 100,
#             'Avg Volume': data['volume'].iloc[start_idx:][state_mask].mean(),
#             'Avg Price Range (%)': ((data['high'].iloc[start_idx:][state_mask] /
#                                    data['low'].iloc[start_idx:][state_mask] - 1) * 100).mean()
#         }

#     # 轉換為DataFrame並格式化
#     stats_df = pd.DataFrame(stats).round(4)

#     return stats_df


# %%
def calculate_state_statistics(data, states, start_idx=0):
    # Prepare data
    sliced_data = data.iloc[start_idx:]
    aligned_states = np.array(states[-len(sliced_data) :])
    returns = sliced_data["close"].pct_change().fillna(0)

    # Initialize stats dictionary
    stats = {}

    # Calculate stats for each state
    for state in np.unique(aligned_states):
        # Create state mask and get state data
        state_mask = aligned_states == state
        state_returns = returns[state_mask]
        state_prices = sliced_data["close"][state_mask]

        # Calculate price-based metrics
        max_price = state_prices.expanding().max()
        min_price = state_prices.expanding().min()
        drawdown = (state_prices / max_price - 1) * 100
        runup = (state_prices / min_price - 1) * 100

        # Store statistics
        stats[f"State {state}"] = {
            "Count": len(state_returns),
            "Return Mean (%)": state_returns.mean() * 100,
            "Return Std (%)": state_returns.std() * 100,
            "Max Drawdown (%)": drawdown.min(),
            "Max Run-up (%)": runup.max(),
            "Mean Return/Max Drawdown": (
                state_returns.mean() / abs(drawdown.min() / 100) if drawdown.min() != 0 else np.inf
            ),
            "Mean Return/Max Run-up": (
                state_returns.mean() / (runup.max() / 100) if runup.max() != 0 else np.inf
            ),
        }

    # Convert to DataFrame and return
    return pd.DataFrame(stats).T


# %%
def calculate_features(bar: pd.DataFrame) -> pd.DataFrame:
    # 計算基礎特徵
    bar["price_range"] = bar["high"] - bar["low"]

    # 計算價量特徵, 包含lags
    close_series = bar["close"].to_numpy()
    volume_series = bar["volume"].to_numpy()

    for i in range(1, 72, 6):
        # 使用numpy計算returns和其他特徵
        returns = np.zeros_like(close_series)
        returns[i:] = (close_series[i:] - close_series[:-i]) / close_series[:-i]
        bar[f"returns_{i}"] = returns

        # 計算log returns
        log_returns = np.zeros_like(close_series)
        with np.errstate(divide="ignore"):
            log_returns[i:] = np.log(close_series[i:]) - np.log(close_series[:-i])
        bar[f"log_returns_{i}"] = log_returns

        # 計算其他特徵
        bar[f"log_returns_category_{i}"] = np.sign(log_returns)

        # 計算波動率 - 使用numpy的rolling window
        volatility = np.zeros_like(returns)
        for j in range(i, len(returns)):
            window = returns[max(0, j - i) : j]
            volatility[j] = np.nanstd(window) if len(window) > 0 else 0
        bar[f"volatility_{i}"] = volatility

        # 計算成交量移動平均 - 使用numpy的rolling window
        volume_ma = np.zeros_like(volume_series)
        for j in range(i, len(volume_series)):
            window = volume_series[max(0, j - i) : j]
            volume_ma[j] = np.nanmean(window) if len(window) > 0 else 0
        bar[f"volume_ma_{i}"] = volume_ma

        # 計算momentum
        momentum = np.zeros_like(close_series)
        momentum[i:] = close_series[i:] - close_series[:-i]
        bar[f"momentum_{i}"] = momentum

        # 計算volume momentum
        volume_momentum = np.zeros_like(volume_series)
        volume_momentum[i:] = volume_series[i:] - volume_series[:-i]
        bar[f"volume_momentum_{i}"] = volume_momentum

    # 刪除NaN值
    bar = bar.dropna()

    # 選擇用於建模的特徵
    feature_columns = [
        col for col in bar.columns if col not in ["ts_event", "close", "volume", "log_returns_1"]
    ]
    return bar[feature_columns]


# %%
def extract_pca_features(features: pd.DataFrame, scalar: StandardScaler, pca: PCA) -> np.ndarray:
    """
    提取並轉換特徵。

    1. 計算原始技術指標特徵
    2. 數據清理和驗證
    3. 標準化
    4. PCA降維

    Returns
    -------
    Optional[np.ndarray]
        降維後的特徵矩陣, 如果處理失敗則返回None
    """
    try:
        # 計算原始特徵
        if features.empty:
            logging.warning("No features calculated")
            return None

        # 檢查無效值
        if features.isna().any().any():
            logging.warning("Features contain NaN values")
            return None

        if np.isinf(features.to_numpy()).any().any():
            logging.warning("Features contain infinite values")
            return None

        # 標準化
        features_scaled = scalar.fit_transform(features)

        # 檢查標準化後的數據
        if np.isnan(features_scaled).any() or np.isinf(features_scaled).any():
            logging.warning("Scaling produced invalid values")
            return None

        # PCA降維
        features_pca = pca.fit_transform(features_scaled)

        # 驗證PCA結果
        if np.isnan(features_pca).any() or np.isinf(features_pca).any():
            logging.warning("PCA produced invalid values")
            return None

        # 記錄解釋方差比
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        logging.debug(f"PCA explained variance ratio: {cumulative_variance_ratio[-1]:.4f}")
        # n_features = int(len(features_pca[0]))
        return features_pca

    except Exception as e:
        logging.error(f"Feature extraction failed: {e!s}")
        logging.exception("Stack trace:")
        return None


# %%
def find_best_num_states(
    features: np.ndarray,
    min_states: int = 2,
    max_states: int = 10,
    improvement_threshold: float = 0.05,
) -> tuple:
    """
    使用交叉驗證找出最佳的隱藏狀態數量。

    Parameters
    ----------
    features : np.ndarray
        PCA轉換後的特徵矩陣
    min_states : int
        最小隱藏狀態數量
    max_states : int
        最大隱藏狀態數量
    improvement_threshold : float
        改善閾值, 低於此閾值的改善被視為不顯著

    Returns
    -------
    tuple
        (最佳狀態數, 所有states的結果列表)
    """
    # 設定交叉驗證的fold數
    n_folds = 3

    # 記錄每個state數量的結果
    results = []
    all_num_states = range(min_states, max_states + 1)

    for n_states in all_num_states:
        # logging.info(f"Testing model with {n_states} states")
        print(f"Testing model with {n_states} states")

        # 建立交叉驗證的folds
        fold_size = len(features) // n_folds
        fold_scores = []

        for i in range(n_folds):
            # 分割訓練和驗證數據
            val_idx = slice(i * fold_size, (i + 1) * fold_size)
            train_idx = list(range(i * fold_size)) + list(range((i + 1) * fold_size, len(features)))

            train_data = features[train_idx].reshape(1, -1, features.shape[-1])
            val_data = features[val_idx].reshape(1, -1, features.shape[-1])

            # 為每個fold創建新的HMM模型
            model = HMMModel(
                config=HMMConfig(
                    n_states=n_states,
                    emission_dim=features.shape[-1],
                    emission_type="gaussian",
                )
            )

            # 訓練模型
            model.fit(train_data, TrainingConfig(method="em", num_epochs=50))

            # 計算驗證集的marginal log probability
            val_ll = vmap(partial(model._model.marginal_log_prob, model._params))(val_data).sum()
            fold_scores.append(val_ll)

        avg_score = np.mean(fold_scores)
        results.append((avg_score, fold_scores))
        # logging.info(f"Average validation log likelihood: {avg_score:.2f}")
        print(f"Average validation log likelihood: {avg_score:.2f}")

    # 找出最佳的狀態數
    # best_idx = np.argmax([r[0] for r in results])
    # best_n_states = all_num_states[best_idx]

    # return best_n_states, results
    avg_scores = [r[0] for r in results]
    improvements = np.zeros_like(avg_scores)
    for i in range(1, len(avg_scores)):
        # 計算相對改善幅度
        relative_improvement = abs((avg_scores[i] - avg_scores[i - 1]) / avg_scores[i - 1])
        improvements[i] = relative_improvement

        # 如果改善幅度小於閾值, 選擇前一個狀態數
        if relative_improvement < improvement_threshold:
            best_n_states = all_num_states[i - 1]
            # logging.info(f"Selected {best_n_states} states due to diminishing returns "
            #             f"(improvement: {relative_improvement:.3f})")
            print(
                f"Selected {best_n_states} states due to diminishing returns "
                f"(improvement: {relative_improvement:.3f})"
            )
            return best_n_states, results

    # 如果所有改善都顯著, 返回最後一個狀態數
    best_n_states = all_num_states[-1]
    return best_n_states, results


# %% [markdown]
# ### Initialize `StandardScalar`, `PCA`, and first `HMM`
#

# %%
features_dim = 20

scalar = StandardScaler()
pca = PCA(n_components=features_dim)
hmm = HMMModel(
    config=HMMConfig(
        n_states=5,
        emission_dim=features_dim,
        emission_type="gaussian",
    )
)
# %% [markdown]
# ### Extract features and split train and test features
#

# %%
features = calculate_features(ada_perp)
pca_features = extract_pca_features(features, scalar, pca)
# print(pca_features.shape)
# print(pca_features[-1])
train_pca_features = pca_features[: 288 * 6].reshape(1, -1, pca.n_components_)
# print(train_pca_features[-1])
# print(train_pca_features.shape)
test_pca_features = pca_features[288 * 6 :]
# print(test_pca_features[-1])
# print(test_pca_features.shape)
# print(pca.components_)
# print(pca.explained_variance_ratio_)

# %% [markdown]
# ### First `HMM` training
#

# %%
print("Fitting HMM model...")
hmm.fit(train_pca_features, TrainingConfig(method="em", num_epochs=50))
print("HMM model fitted")

# %%
# Make predictions
print("Making predictions...")
predicted_states = hmm.predict(test_pca_features)
predicted_states_proba = hmm.predict_proba(test_pca_features)
print("Predictions made")
print(predicted_states)
print(len(predicted_states))
# print(predicted_states_proba)
print(predicted_states_proba[-1])
print(predicted_states_proba[-1][predicted_states[-1]])

# %% [markdown]
# ### Finding best number of states
#

# %%
# 找出最佳的隱藏狀態數量
print("Finding optimal number of hidden states...")
best_n_states, cv_results = find_best_num_states(
    pca_features, min_states=2, max_states=10, improvement_threshold=0.1  # 5%的改善閾值
)
print(f"Best number of states: {best_n_states}")

# %%
avg_val_lls, all_val_lls = zip(*cv_results)
# 首先檢查實際的數據長度
print(f"Length of avg_val_lls: {len(avg_val_lls)}")
print(f"Testing states range: {list(range(2, 9))}")  # 7個狀態數

# 修改繪圖函數的調用
# 找出最佳狀態數後的視覺化
plot_validation_results(
    all_num_states=range(2, len(avg_val_lls) + 2),  # 根據實際的結果長度調整範圍
    avg_val_lls=avg_val_lls,
    all_val_lls=all_val_lls,
)

# 使用最佳狀態數訓練模型
hmm = HMMModel(
    config=HMMConfig(
        n_states=2,
        emission_dim=pca.n_components_,
        emission_type="gaussian",
    )
)

# 訓練模型
hmm = hmm.fit(train_pca_features, TrainingConfig(method="em", num_epochs=50))

# 視覺化訓練過程
plot_training_progress(hmm.training_history)

# 獲取模型參數
params = hmm.params

# Make predictions
print("Making predictions...")
predicted_states = hmm.predict(test_pca_features)
predicted_states_proba = hmm.predict_proba(test_pca_features)
print("Predictions made")
print(predicted_states)
print(len(predicted_states))
# print(predicted_states_proba)
print(predicted_states_proba[-1])
print(predicted_states_proba[-1][predicted_states[-1]])

# %%
trained_states = hmm.predict(pca_features[: 864 * 2])
print(trained_states)
stats_df_train = calculate_state_statistics(ada_perp.iloc[: 864 * 2], trained_states)
print(stats_df_train)

# %%
# 計算並顯示統計數據
stats_df = calculate_state_statistics(
    data=ada_perp.iloc[288 * 6 :], states=predicted_states  # 使用測試集的數據
)
print("\nStatistics for each HMM state:")
print(stats_df)

# %%
# 使用函數繪製圖表
plot_hmm_states_with_data(
    data=ada_perp,
    states=predicted_states,
    start_idx=800,  # 使用測試集的起始索引
    title="HMM Hidden States and ADA/USDT Price Data",
)

# %%
