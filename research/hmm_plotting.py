import jax.numpy as jnp
import matplotlib.pyplot as plt
from dynamax.utils.plotting import CMAP
from dynamax.utils.plotting import COLORS
from dynamax.utils.plotting import white_to_color_cmap


def plot_gaussian_hmm(hmm, params, emissions, states, title="Emission Distributions", alpha=0.25):
    """
    繪製高斯 HMM 的發射分布和狀態分類結果。

    Parameters
    ----------
    hmm : HMMModel
        HMM 模型實例
    params : NamedTuple
        模型參數
    emissions : np.ndarray
        排放數據, shape 為 (num_timesteps, emission_dim)
    states : np.ndarray
        預測的狀態序列, shape 為 (num_timesteps,)
    title : str, optional
        圖表標題
    alpha : float, optional
        點和線的透明度
    """
    # 設定繪圖範圍
    lim = 1.1 * abs(emissions).max()
    XX, YY = jnp.meshgrid(jnp.linspace(-lim, lim, 100), jnp.linspace(-lim, lim, 100))
    grid = jnp.column_stack((XX.ravel(), YY.ravel()))

    plt.figure(figsize=(10, 8))

    # 為每個狀態繪製等高線和數據點
    for k in range(hmm.config.n_states):
        # 計算網格點的對數概率
        lls = hmm._model.emission_distribution(params, k).log_prob(grid)
        # 繪製等高線
        plt.contour(XX, YY, jnp.exp(lls).reshape(XX.shape), cmap=white_to_color_cmap(COLORS[k]))
        # 繪製屬於該狀態的數據點
        plt.plot(
            emissions[states == k, 0],
            emissions[states == k, 1],
            "o",
            mfc=COLORS[k],
            mec="none",
            ms=3,
            alpha=alpha,
        )

    # 繪製數據點的連接線, 顯示時間順序
    plt.plot(emissions[:, 0], emissions[:, 1], "-k", lw=1, alpha=alpha)

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)
    plt.gca().set_aspect(1.0)
    plt.tight_layout()


def plot_gaussian_hmm_data(hmm, params, emissions, states, timestamps=None, xlim=None):
    """
    繪製時間序列數據和對應的狀態序列。

    Parameters
    ----------
    hmm : HMMModel
        HMM 模型實例
    params : NamedTuple
        模型參數
    emissions : np.ndarray
        排放數據, shape 為 (num_timesteps, emission_dim)
    states : np.ndarray
        預測的狀態序列, shape 為 (num_timesteps,)
    timestamps : np.ndarray, optional
        時間戳數據, 用於 x 軸
    xlim : tuple, optional
        x 軸的範圍限制 (min, max)
    """
    num_timesteps = len(emissions)
    emission_dim = hmm.config.emission_dim
    means = params.emissions.means[states]
    lim = 1.05 * abs(emissions).max()

    # 建立子圖
    fig, axs = plt.subplots(emission_dim, 1, figsize=(15, 4 * emission_dim), sharex=True)
    if emission_dim == 1:
        axs = [axs]

    x = timestamps if timestamps is not None else range(num_timesteps)

    # 為每個維度繪製子圖
    for d in range(emission_dim):
        # 用顏色區塊表示狀態序列
        axs[d].imshow(
            states[None, :],
            aspect="auto",
            interpolation="none",
            cmap=CMAP,
            vmin=0,
            vmax=len(COLORS) - 1,
            extent=(min(x), max(x), -lim, lim),
        )
        # 繪製實際數據
        axs[d].plot(x, emissions[:, d], "-k", label="Observed")
        # 繪製狀態均值
        axs[d].plot(x, means[:, d], ":k", label="State Mean")
        axs[d].set_ylabel(f"PCA Component {d+1}")
        if d == 0:
            axs[d].legend()

    # 設定 x 軸範圍
    if xlim is not None:
        plt.xlim(xlim)

    axs[-1].set_xlabel("Time")
    axs[0].set_title("HMM State Sequence and Emissions")
    plt.tight_layout()


def plot_validation_results(all_num_states, avg_val_lls, all_val_lls):
    """
    繪製交叉驗證結果, 展示不同狀態數量下的模型表現。

    這個函數創建兩種視覺元素:
    1. 一條線顯示平均驗證對數似然隨狀態數的變化
    2. 散點顯示每個fold的具體表現, 有助於觀察方差

    Parameters
    ----------
    all_num_states : array-like
        測試的狀態數列表
    avg_val_lls : array-like
        每個狀態數的平均驗證對數似然
    all_val_lls : list of arrays
        每個狀態數在不同fold的驗證對數似然
    """
    # 將輸入轉換為JAX數組以確保一致性
    all_num_states = jnp.array(list(all_num_states))
    avg_val_lls = jnp.array(avg_val_lls)

    plt.figure(figsize=(10, 6))

    # 繪製平均驗證對數似然趨勢線
    plt.plot(all_num_states, avg_val_lls, "-ko", label="Average", linewidth=2)

    # 繪製每個fold的結果
    for i, (num_state, fold_vals) in enumerate(zip(all_num_states, all_val_lls)):
        # 將fold的數據轉換為JAX數組
        fold_vals = jnp.array(fold_vals)
        # 創建對應的x坐標數組
        x_coords = jnp.full(fold_vals.shape, num_state)

        plt.plot(
            x_coords,
            fold_vals,
            ".",
            alpha=0.5,
            label="Individual folds" if i == 0 else None,
            markersize=8,
        )

    plt.xlabel("Number of States (K)", fontsize=12)
    plt.ylabel("Validation Log Probability", fontsize=12)
    plt.title("Cross-validation Results vs Number of States", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # 添加更多視覺優化
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tick_params(labelsize=10)

    plt.tight_layout()


def plot_training_progress(training_history, true_lp=None):
    """
    繪製模型訓練過程中的指標變化。

    Parameters
    ----------
    training_history : list[dict]
        訓練歷史記錄, 每個元素是包含 'epoch' 和 ('log_prob' 或 'loss') 的字典
    true_lp : float, optional
        真實模型的對數似然值(如果有的話)
    """
    plt.figure(figsize=(10, 6))

    # 從訓練歷史中提取數據
    epochs = [entry["epoch"] for entry in training_history]

    # 判斷使用哪個指標（log_prob 或 loss）
    if "log_prob" in training_history[0]:
        values = [entry["log_prob"] for entry in training_history]
        ylabel = "Log Probability"
    else:
        values = [entry["loss"] for entry in training_history]
        ylabel = "Loss"

    # 轉換為 JAX 數組以確保兼容性
    epochs = jnp.array(epochs)
    values = jnp.array(values)

    # 繪製訓練進度
    plt.plot(epochs, values, "-o", label="Training Progress", linewidth=2, markersize=4, alpha=0.7)

    # 如果有真實值，添加參考線
    if true_lp is not None:
        plt.axhline(true_lp, color="k", linestyle=":", label="True Value")

    plt.xlabel("Training Epoch", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title("Model Training Progress", fontsize=14)

    # 美化圖表
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    # 移除多餘的邊框
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()
