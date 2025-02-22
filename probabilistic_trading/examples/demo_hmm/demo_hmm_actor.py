# examples/demo_hmm/demo_hmm_actor.py
import numpy as np
from nautilus_trader.common.actor import Actor
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import DataType

from probabilistic_trading.model.hmm import HMMConfig
from probabilistic_trading.model.hmm import HMMModel
from probabilistic_trading.model.hmm import TrainingConfig

from .demo_hmm_data import DemoHMMStateData


class DemoHMMActor(Actor):
    """
    最簡單的HMM Actor實現。
    直接使用OHLCV數據訓練HMM並發布狀態。
    """

    def __init__(
        self,
        bar_type: BarType,
        min_bars: int = 100,  # 最小需要的K線數量
    ):
        super().__init__()
        self.bar_type = bar_type
        self.min_bars = min_bars

        # 初始化HMM模型
        self.model = HMMModel(
            config=HMMConfig(
                n_states=3,  # 使用3個狀態的簡單模型
                emission_dim=5,  # OHLCV 5個特徵
                emission_type="gaussian",
            )
        )

        # 追蹤是否已訓練
        self.is_trained = False

    def on_start(self):
        """訂閱K線數據"""
        self.subscribe_bars(self.bar_type)

    def on_bar(self, bar: Bar) -> None:
        """
        每收到一根K線就:
        1. 檢查是否有足夠數據來訓練
        2. 需要時訓練模型
        3. 進行預測並發布狀態
        """
        # 檢查數據量
        if self.cache.bar_count(self.bar_type) < self.min_bars:
            return
        try:
            # 準備特徵數據
            bars = self.cache.bars(self.bar_type)
            features = np.array(
                [
                    [float(b.open), float(b.high), float(b.low), float(b.close), float(b.volume)]
                    for b in bars
                ]
            )
            self.log.info(f"Latest feature: {features[-1]}")
            self.log.info(f"Feature has {features.shape} shape")
            self.log.info(f"Processing {len(features)} bars")
            # 重塑為需要的形狀 (1, n_timesteps, emission_dim)
            train_features = features.reshape(1, -1, 5)
            self.log.info(f"Reshaped train feature has {train_features.shape} shape")
            # 如果還沒訓練過,進行訓練
            if not self.is_trained:
                self.log.info("Initial HMM model training")
                self.model.fit(
                    train_features, training_config=TrainingConfig(method="em", num_epochs=50)
                )
                self.log.info("Model training completed")
                self.is_trained = True

            # 進行預測
            self.log.info("Predicting state")
            self.log.info(f"Predicting features has {features.shape} shape")
            states = self.model.predict(features)  # 取全部時間步的狀態
            probas = self.model.predict_proba(features)[0]  # 取最新狀態的概率矩陣

            state = states[0]  # 取最新狀態是在[0]
            state_proba = probas[state]  # 取最新狀態的機率

            # 發布狀態數據
            self.log.info(f"Predicted state: {state}, proba: {state_proba}")
            state_data = DemoHMMStateData(
                state=state,
                state_proba=state_proba,
                ts_init=bar.ts_init,
                ts_event=bar.ts_event,
            )
            self.log.info(f"Publishing state data: {state_data}")
            self.publish_data(DataType(DemoHMMStateData), state_data)

        except Exception as e:
            self.log.error(f"Error processing bar: {e}")
