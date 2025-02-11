import datetime as dt
import pickle  # noqa
from decimal import Decimal

import numpy as np  # noqa
import pandas as pd  # noqa
from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.data import Data
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import DataType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import PositionSide  # noqa
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import PositionId
from nautilus_trader.model.instruments import Instrument  # noqa
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.trading.strategy import Strategy

from probabilistic_trading.model.hmm.hmm_data import HMMStateData


class HMMStrategyConfig(StrategyConfig):
    instrument_id: InstrumentId
    # n_states: int = 3
    min_prob_threshold: float = 0.7
    position_size: Decimal = Decimal("100.0")
    # min_training_bars: int = 864
    # update_interval: int = 3
    # pca_components: int = 3


class HMMStrategy(Strategy):
    def __init__(self, config: HMMStrategyConfig):
        super().__init__(config)
        # 設置bar type
        self.bar_type = BarType.from_str(f"{config.instrument_id}-5-MINUTE-LAST-EXTERNAL")

    def on_start(self):
        """策略啟動時被呼叫"""
        self.start_time = dt.datetime.now()
        self.log.info(f"Strategy started at: {self.start_time}")

        self.subscribe_bars(self.bar_type)
        self.subscribe_data(DataType(HMMStateData))

    def on_data(self, data: Data) -> None:
        """
        處理HMM狀態數據。

        Parameters
        ----------
        data : HMMStateData
            HMM狀態數據
        """
        print("handling data received in on_data")
        if isinstance(data, HMMStateData):
            print(f"Received HMMStateData: {data}")
        else:
            print("No data received")
            return

        # 檢查預測質量
        if data.prediction_quality < 0.5:
            self.log.debug(f"Low prediction quality: {data.prediction_quality:.4f}")
            return

        # 獲取狀態和概率
        state = data.state
        state_prob = data.state_proba[state]

        # 執行交易
        self._execute_trades(state, state_prob)

        # 記錄狀態
        self.log.info(
            f"HMM State - State: {state}, "
            f"Probability: {state_prob:.4f}, "
            f"Quality: {data.prediction_quality:.4f}"
        )

    def on_bar(self, bar: Bar):
        self.log.info(repr(bar), LogColor.CYAN)
        self.log.info(f"Bar count: {self.cache.bar_count(self.bar_type)}")
        self.log.info(f"Bar shape: {len(self.cache.bars(self.bar_type))}")

    def _execute_trades(self, state: int, state_prob: float):
        """
        執行交易邏輯。

        Parameters
        ----------
        state : int
            當前狀態
        state_prob : float
            狀態概率
        """
        # BUY LOGIC
        if state == 0 and state_prob > self.config.min_prob_threshold:
            if self.portfolio.is_flat(self.config.instrument_id):
                self.cancel_all_orders(self.config.instrument_id)
                self._long()
            elif self.portfolio.is_net_short(self.config.instrument_id):
                self.close_all_positions(self.config.instrument_id)
                self.cancel_all_orders(self.config.instrument_id)
                self._long()
        # SELL LOGIC
        elif state == 1 and state_prob > self.config.min_prob_threshold:
            if self.portfolio.is_flat(self.config.instrument_id):
                self.cancel_all_orders(self.config.instrument_id)
                self._short()
            elif self.portfolio.is_net_long(self.config.instrument_id):
                self.close_all_positions(self.config.instrument_id)
                self.cancel_all_orders(self.config.instrument_id)
                self._short()

    def _long(self) -> None:
        """開多倉"""
        current_price = self.cache.bar(self.bar_type).close
        usdt_size = self.config.position_size
        quantity = usdt_size / Decimal(str(current_price))

        order: MarketOrder = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(quantity),
        )
        position_id = PositionId(f"{self.config.instrument_id}-LONG")
        self.submit_order(order, position_id)

    def _short(self) -> None:
        """開空倉"""
        current_price = self.cache.bar(self.bar_type).close
        usdt_size = self.config.position_size
        quantity = usdt_size / Decimal(str(current_price))

        order: MarketOrder = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(quantity),
        )
        position_id = PositionId(f"{self.config.instrument_id}-SHORT")
        self.submit_order(order, position_id)

    def on_stop(self):
        # Remember and log end time of strategy
        self.end_time = dt.datetime.now()
        self.log.info(f"Strategy finished at: {self.end_time}")
        self.log.info(f"Total bar count: {self.cache.bar_count(self.bar_type)}")
        self.cancel_all_orders(self.config.instrument_id)
        self.close_all_positions(self.config.instrument_id)
        self.unsubscribe_bars(self.bar_type)
        self.unsubscribe_data(DataType(HMMStateData))

    def on_reset(self) -> None:
        """重置策略時的操作"""

    def on_dispose(self) -> None:
        """清理資源"""
