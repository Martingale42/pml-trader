"""
Hidden Markov Model data classes.
"""

import numpy as np
from nautilus_trader.core.data import Data
from nautilus_trader.core.datetime import nanos_to_secs


class HMMStateData(Data):
    """
    HMM狀態數據類。
    用於在Actor和Strategy之間傳遞HMM狀態信息。

    Parameters
    ----------
    state : int
        當前預測的狀態
    state_proba : np.ndarray
        各狀態的概率分布, shape為(n_states,)
    log_likelihood : float
        模型對數似然, 用於評估模型擬合程度
    transition_matrix : np.ndarray
        狀態轉移矩陣, shape為(n_states, n_states)
    means : np.ndarray
        各狀態的均值, shape為(n_states, n_features)
    sds : np.ndarray
        各狀態的標準差, shape為(n_states, n_features)
    prediction_quality : float
        預測質量分數, 範圍[0,1]
    ts_init : int
        初始化時間戳(納秒)
    ts_event : int
        事件時間戳(納秒)
    """

    def __init__(
        self,
        state: int,
        state_proba: np.ndarray,
        log_likelihood: float,
        transition_matrix: np.ndarray,
        means: np.ndarray,
        sds: np.ndarray,
        prediction_quality: float,
        ts_init: int,
        ts_event: int,
    ):
        super().__init__()
        # 驗證輸入
        if not isinstance(state, int) or state < 0:
            raise ValueError("state must be a non-negative integer")
        if not isinstance(state_proba, np.ndarray) or state_proba.ndim != 1:
            raise ValueError("state_proba must be a 1D numpy array")
        if not isinstance(prediction_quality, float) or not 0 <= prediction_quality <= 1:
            raise ValueError("prediction_quality must be a float between 0 and 1")

        # 設置屬性
        self._state = state
        self._state_proba = state_proba
        self._log_likelihood = log_likelihood
        self._transition_matrix = transition_matrix
        self._means = means
        self._sds = sds
        self._prediction_quality = prediction_quality
        self._ts_init = ts_init
        self._ts_event = ts_event

    @property
    def state(self) -> int:
        """當前預測的狀態"""
        return self._state

    @property
    def state_proba(self) -> np.ndarray:
        """各狀態的概率分布"""
        return self._state_proba

    @property
    def log_likelihood(self) -> float:
        """模型對數似然"""
        return self._log_likelihood

    @property
    def transition_matrix(self) -> np.ndarray:
        """狀態轉移矩陣"""
        return self._transition_matrix

    @property
    def means(self) -> np.ndarray:
        """各狀態的均值"""
        return self._means

    @property
    def sds(self) -> np.ndarray:
        """各狀態的標準差"""
        return self._sds

    @property
    def prediction_quality(self) -> float:
        """預測質量分數"""
        return self._prediction_quality

    @property
    def ts_event(self) -> int:
        """事件時間戳(納秒)"""
        return self._ts_event

    @property
    def ts_init(self) -> int:
        """初始化時間戳(納秒)"""
        return self._ts_init

    def __str__(self) -> str:
        """返回數據對象的字符串表示"""
        return (
            f"HMMStateData(state={self.state}, "
            f"state_prob={self.state_proba[self.state]:.4f}, "
            f"quality={self.prediction_quality:.4f}, "
            f"ts_event={nanos_to_secs(self.ts_event)})"
        )
