"""
Enhanced Kalman Filter implementation optimized for financial time series.
Uses NumPy for vectorized operations and implements adaptive estimation.
"""

import numba
import numpy as np


@numba.jit(nopython=True)
def _update_adaptive_noise(
    innovation: np.ndarray, old_R: np.ndarray, forgetting_factor: float
) -> np.ndarray:
    """
    使用Numba加速的自適應噪聲估計。

    Parameters
    ----------
    innovation : np.ndarray
        新息序列
    old_R : np.ndarray
        上一時刻的測量噪聲協方差
    forgetting_factor : float
        遺忘因子 (0 < factor <= 1)
    """
    innovation_cov = np.outer(innovation, innovation)
    return forgetting_factor * old_R + (1 - forgetting_factor) * innovation_cov


class KalmanFilter:
    """
    增強的Kalman Filter實現，針對金融時間序列優化：
    - 自適應噪聲估計
    - 遺忘因子機制
    - 數值穩定性優化
    - 批量處理支持
    """

    def __init__(
        self,
        dim_state: int,
        initial_state: np.ndarray | None = None,
        initial_covariance: np.ndarray | None = None,
        process_noise_cov: np.ndarray | None = None,
        measure_noise_cov: np.ndarray | None = None,
        forgetting_factor: float = 0.98,
        adaptive_estimation: bool = True,
    ):
        """
        初始化增強的Kalman Filter。

        Parameters
        ----------
        dim_state : int
            狀態向量維度
        initial_state : Optional[np.ndarray]
            初始狀態向量
        initial_covariance : Optional[np.ndarray]
            初始狀態協方差
        process_noise_cov : Optional[np.ndarray]
            過程噪聲協方差矩陣 Q
        measure_noise_cov : Optional[np.ndarray]
            測量噪聲協方差矩陣 R
        forgetting_factor : float
            遺忘因子，用於自適應估計
        adaptive_estimation : bool
            是否啟用自適應估計
        """
        self.dim_state = dim_state

        # 初始化狀態
        self.x = initial_state if initial_state is not None else np.zeros(dim_state)
        self.P = initial_covariance if initial_covariance is not None else np.eye(dim_state)

        # 初始化噪聲協方差
        self.Q = process_noise_cov if process_noise_cov is not None else 0.01 * np.eye(dim_state)
        self.R = measure_noise_cov if measure_noise_cov is not None else 0.01 * np.eye(dim_state)

        # 自適應參數
        self.forgetting_factor = forgetting_factor
        self.adaptive_estimation = adaptive_estimation

        # 計算中間結果
        self.K = None  # Kalman增益
        self.innovation = None  # 新息
        self.S = None  # 新息協方差

        # 性能監控
        self.innovation_history = []
        self.estimation_error_history = []

    # @numba.jit(nopython=True)
    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Numba加速的預測步驟。

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            預測的狀態向量和協方差矩陣
        """
        # 狀態預測
        x_pred = self.x.copy()

        # 使用遺忘因子調整協方差預測
        P_pred = (
            self.P / self.forgetting_factor + self.Q
            if self.adaptive_estimation
            else self.P + self.Q
        )

        return x_pred, P_pred

    def update(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        優化的更新步驟，包含自適應估計和數值穩定性改進。

        Parameters
        ----------
        measurement : np.ndarray
            觀測向量

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            更新後的狀態估計和協方差矩陣
        """
        # 預測
        x_pred, P_pred = self.predict()

        # 計算新息
        self.innovation = measurement - x_pred
        self.innovation_history.append(self.innovation)

        # 自適應估計測量噪聲
        if self.adaptive_estimation and len(self.innovation_history) > 1:
            self.R = _update_adaptive_noise(self.innovation, self.R, self.forgetting_factor)

        # 計算新息協方差（使用Joseph形式提高數值穩定性）
        self.S = P_pred + self.R

        # 確保協方差矩陣的數值穩定性
        self.S = (self.S + self.S.T) / 2  # 確保對稱性
        min_eig = np.min(np.real(np.linalg.eigvals(self.S)))
        if min_eig < 1e-8:
            # 如果最小特徵值太小，添加一個小的對角矩陣
            self.S += (abs(min_eig) + 1e-8) * np.eye(self.dim_state)

        # 使用改進的Cholesky分解求解Kalman增益
        try:
            # 使用改進的數值方法
            L = np.linalg.cholesky(self.S)
            temp = np.linalg.solve(L, P_pred.T)
            self.K = temp.T @ np.linalg.solve(L.T, np.eye(self.dim_state))
        except np.linalg.LinAlgError:
            # 如果Cholesky分解失敗，使用SVD方法
            U, s, Vh = np.linalg.svd(self.S)
            s_inv = np.where(s > 1e-8, 1 / s, 0)  # 使用閾值過濾小奇異值
            self.K = P_pred @ (U * s_inv) @ Vh

        # 更新狀態估計
        self.x = x_pred + self.K @ self.innovation

        # 使用改進的Joseph形式更新協方差
        I = np.eye(self.dim_state)
        KH = self.K  # 在這裡H是單位矩陣
        self.P = (I - KH) @ P_pred @ (I - KH).T + self.K @ self.R @ self.K.T

        # 確保數值穩定性
        self.P = (self.P + self.P.T) / 2  # 確保對稱性
        min_eig = np.min(np.real(np.linalg.eigvals(self.P)))
        if min_eig < 1e-8:
            self.P += (abs(min_eig) + 1e-8) * np.eye(self.dim_state)

        # 記錄估計誤差
        estimation_error = np.linalg.norm(self.innovation)
        self.estimation_error_history.append(estimation_error)

        return self.x, self.P

    def smooth(self, measurements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        RTS平滑器，用於批量數據處理。

        Parameters
        ----------
        measurements : np.ndarray
            觀測數據序列

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            平滑後的狀態估計序列和對應的協方差
        """
        n_samples = len(measurements)

        # 前向濾波
        filtered_states = np.zeros((n_samples, self.dim_state))
        filtered_covs = np.zeros((n_samples, self.dim_state, self.dim_state))

        for t in range(n_samples):
            self.update(measurements[t])
            filtered_states[t] = self.x
            filtered_covs[t] = self.P

        # 後向平滑
        smoothed_states = filtered_states.copy()
        smoothed_covs = filtered_covs.copy()

        for t in range(n_samples - 2, -1, -1):
            # 計算平滑增益
            C = filtered_covs[t] @ np.linalg.pinv(filtered_covs[t] + self.Q)

            # 更新狀態和協方差
            state_diff = smoothed_states[t + 1] - filtered_states[t]
            smoothed_states[t] += C @ state_diff

            cov_diff = smoothed_covs[t + 1] - filtered_covs[t]
            smoothed_covs[t] += C @ cov_diff @ C.T

        return smoothed_states, smoothed_covs

    def get_state_info(self) -> dict:
        """
        獲取增強的狀態診斷信息。

        Returns
        -------
        dict
            包含所有狀態變數和性能指標的字典
        """
        # 計算性能指標
        recent_innovations = (
            np.array(self.innovation_history[-10:]) if self.innovation_history else np.array([])
        )
        recent_errors = (
            np.array(self.estimation_error_history[-10:])
            if self.estimation_error_history
            else np.array([])
        )

        return {
            "state_estimate": self.x.copy(),
            "state_covariance": self.P.copy(),
            "innovation": (self.innovation.copy() if self.innovation is not None else None),
            "innovation_covariance": self.S.copy() if self.S is not None else None,
            "kalman_gain": self.K.copy() if self.K is not None else None,
            "prediction_error": (
                np.linalg.norm(self.innovation) if self.innovation is not None else None
            ),
            "estimation_uncertainty": np.trace(self.P),
            "signal_noise_ratio": (
                np.trace(self.P) / np.trace(self.R) if self.R is not None else None
            ),
            # 新增的性能指標
            "innovation_stability": (
                np.std(recent_innovations) if len(recent_innovations) > 0 else None
            ),
            "error_trend": (np.mean(np.diff(recent_errors)) if len(recent_errors) > 1 else None),
            "adaptive_noise_level": (np.trace(self.R) if self.adaptive_estimation else None),
        }
