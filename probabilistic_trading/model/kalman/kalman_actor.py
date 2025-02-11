import numpy as np
from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.core.data import Data
from nautilus_trader.model import BarType
from nautilus_trader.model import DataType
from nautilus_trader.model.data import Bar
from nautilus_trader.model.identifiers import InstrumentId

from .kalman_data import KFStateData
from .kalman_model import KalmanFilter


class KFActorConfig(ActorConfig):
    """
    Configuration for the Kalman filter.

    Parameters
    ----------
    instrument_id : InstrumentId
        交易商品ID
    bar_type : BarType
        K線類型
    state_dim : int
        狀態向量維度
    process_noise : np.ndarray
        過程噪聲協方差矩陣 Q
    measure_noise : np.ndarray
        觀測噪聲協方差矩陣 R
    """

    instrument_id: InstrumentId
    bar_type: BarType
    state_dim: int = 1
    process_noise: np.ndarray | None = None
    measure_noise: np.ndarray | None = None


class KFActor(Actor):
    """
    使用Kalman Filter進行狀態估計的Actor。
    整合了KalmanFilterCore進行計算，並發布KalmanStateInfo。
    """

    def __init__(self, config: KFActorConfig):
        super().__init__(config)

        # 設定
        self.instrument_id = config.instrument_id
        self.bar_type = config.bar_type

        # 初始化噪聲矩陣
        state_dim = config.state_dim
        self.process_noise = (
            config.process_noise if config.process_noise is not None else np.eye(state_dim) * 0.001
        )
        self.measure_noise = (
            config.measure_noise if config.measure_noise is not None else np.eye(state_dim) * 0.001
        )

        # 創建Kalman Filter核心
        self.kf = KalmanFilter(
            dim_state=state_dim,
            process_noise_cov=self.process_noise,
            measure_noise_cov=self.measure_noise,
        )

        # 歷史資料
        self.last_measurement: np.ndarray | None = None

    def on_start(self):
        """Actor啟動時被呼叫。訂閱所需的數據流。"""
        self.subscribe_bars(self.bar_type)
        self.request_bars(self.bar_type)

    def on_historical_data(self, data: Data) -> None:
        """處理歷史數據。"""
        if not isinstance(data, Bar):
            return

        # 構建測量向量
        measurement = self._prepare_measurement(data)

        # 更新Kalman Filter
        self.kf.update(measurement)
        self.last_measurement = measurement

    def on_bar(self, bar: Bar) -> None:
        """處理實時K線數據。"""
        if self.last_measurement is None:
            return

        # 構建測量向量
        measurement = self._prepare_measurement(bar)

        # 使用Kalman filter預測和更新
        self.kf.update(measurement)

        # 獲取完整狀態信息
        state_info = self.kf.get_state_info()

        # 創建並發布狀態信息
        signal = KFStateData(
            state_estimate=state_info["state_estimate"],
            state_covariance=state_info["state_covariance"],
            innovation=state_info["innovation"],
            innovation_covariance=state_info["innovation_covariance"],
            kalman_gain=state_info["kalman_gain"],
            prediction_error=state_info["prediction_error"],
            estimation_uncertainty=state_info["estimation_uncertainty"],
            signal_noise_ratio=state_info["signal_noise_ratio"],
            _ts_init=self.clock.timestamp_ns(),
            _ts_event=bar.ts_event,
        )

        # 發布信號
        self.publish_data(DataType(signal), signal)

        # 更新歷史數據
        self.last_measurement = measurement

        # 記錄狀態
        self.log.debug(
            f"Updated Kalman Filter - "
            f"State: {state_info['state_estimate']}, "
            f"Uncertainty: {state_info['estimation_uncertainty']:.6f}, "
            f"SNR: {state_info['signal_noise_ratio']:.6f}"
        )

    def _prepare_measurement(self, bar: Bar) -> np.ndarray:
        """
        從Bar數據構建測量向量。
        可以根據需要擴展，加入更多特徵。

        Parameters
        ----------
        bar : Bar
            K線數據

        Returns
        -------
        np.ndarray
            測量向量
        """
        # 目前僅使用收盤價，可以擴展加入更多特徵
        return np.array([bar.close.as_double()])
