# examples/demo_hmm/demo_hmm_data.py
from nautilus_trader.core.data import Data


class DemoHMMStateData(Data):
    """
    最簡單的HMM狀態數據類。
    只包含狀態和概率兩個必要信息。
    """

    def __init__(
        self,
        state: int,
        state_proba: float,
        ts_init: int,
        ts_event: int,
    ):
        super().__init__()
        self._state = state
        self._state_proba = state_proba
        self._ts_init = ts_init
        self._ts_event = ts_event

    @property
    def state(self) -> int:
        return self._state

    @property
    def state_proba(self) -> float:
        return self._state_proba

    @property
    def ts_init(self) -> int:
        return self._ts_init

    @property
    def ts_event(self) -> int:
        return self._ts_event
