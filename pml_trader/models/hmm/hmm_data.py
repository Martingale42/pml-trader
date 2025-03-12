"""
Hidden Markov Model data classes.
"""

from nautilus_trader.core.data import Data


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
        state_proba: float,
        ts_init: int,
        ts_event: int,
    ):
        super().__init__()
        # 設置屬性
        self._state = state
        self._state_proba = state_proba
        self._ts_init = ts_init
        self._ts_event = ts_event

    @property
    def state(self) -> int:
        """當前預測的狀態"""
        return self._state

    @property
    def state_proba(self) -> float:
        """各狀態的概率分布"""
        return self._state_proba

    @property
    def ts_event(self) -> int:
        """事件時間戳(納秒)"""
        return self._ts_event

    @property
    def ts_init(self) -> int:
        """初始化時間戳(納秒)"""
        return self._ts_init
