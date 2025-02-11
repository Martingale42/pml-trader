"""
Kalman Filter signal data class.
"""

import numpy as np
from nautilus_trader.core.data import Data


class KFStateData(Data):
    """
    包含Kalman Filter狀態和預測信息的信號類。
    用於從Actor發布狀態信息到Strategy。

    Parameters
    ----------
    state_estimate : np.ndarray
        狀態估計向量
    state_covariance : np.ndarray
        狀態估計協方差矩陣
    innovation : np.ndarray
        新息(measurement - prediction)
    innovation_covariance : np.ndarray
        新息協方差
    kalman_gain : np.ndarray
        Kalman增益矩陣
    prediction_error : float
        預測誤差大小
    estimation_uncertainty : float
        狀態估計的不確定性
    signal_noise_ratio : float
        信噪比
    """

    state_estimate: np.ndarray
    state_covariance: np.ndarray
    innovation: np.ndarray
    innovation_covariance: np.ndarray
    kalman_gain: np.ndarray
    prediction_error: float
    estimation_uncertainty: float
    signal_noise_ratio: float
    _ts_init: int  # Required by nautilus Data class
    _ts_event: int  # Required by nautilus Data class

    @property
    def ts_init(self) -> int:
        """Timestamp of signal creation."""
        return self._ts_init

    @property
    def ts_event(self) -> int:
        """Timestamp of event that generated signal."""
        return self._ts_event

    def get_state_vector(self) -> np.ndarray:
        """
        獲取狀態估計向量。

        Returns
        -------
        np.ndarray
            當前狀態估計
        """
        return self.state_estimate.copy()

    def get_uncertainty(self) -> float:
        """
        獲取狀態估計的不確定性度量。

        Returns
        -------
        float
            不確定性度量(協方差矩陣的跡)
        """
        return self.estimation_uncertainty

    def get_prediction_quality(self) -> float:
        """
        獲取預測質量指標。
        基於預測誤差和信噪比計算。

        Returns
        -------
        float
            預測質量分數 (0-1之間，越高表示預測越可靠)
        """
        # 將預測誤差標準化到0-1之間
        normalized_error = np.exp(-self.prediction_error)

        # 將信噪比標準化到0-1之間
        normalized_snr = 1 / (1 + np.exp(-self.signal_noise_ratio))

        # 綜合評分
        return 0.5 * (normalized_error + normalized_snr)
