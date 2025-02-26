# examples/demo_hmm/demo_hmm_strategy.py
from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.data import Data
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import DataType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

from .demo_hmm_data import DemoHMMStateData


class DemoHMMStrategyConfig(StrategyConfig, frozen=True):
    """策略配置"""

    instrument_id: InstrumentId
    bar_type: str
    position_size: Decimal = Decimal("1.0")
    prob_threshold: float = 0.8


class DemoHMMStrategy(Strategy):
    """
    最簡單的HMM策略實現。
    根據HMM狀態進行交易:
    - 狀態0: 做多
    - 狀態1: 做空
    - 狀態2: 保持
    """

    def __init__(self, config: DemoHMMStrategyConfig):
        super().__init__(config)
        self.bar_type = BarType.from_str(config.bar_type)
        self.processed_signals = 0
        self.current_state = None

    def on_start(self):
        """訂閱數據"""
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"No instrument found for {self.config.instrument_id}")
            return
        self.subscribe_bars(self.bar_type)
        self.subscribe_data(DataType(DemoHMMStateData))

    def on_data(self, data: Data) -> None:
        """
        處理HMM狀態數據並執行交易
        """
        if not isinstance(data, DemoHMMStateData):
            return

        self.processed_signals += 1
        if data.state_proba < self.config.prob_threshold:
            return
        self.log.info(f"Received HMMStateData: {data}")
        self.current_state = data.state
        # 只在狀態概率高於閾值時交易

    def on_bar(self, bar):
        """每次收到新K線時執行交易"""
        if self.current_state is None:
            return

        # 根據狀態執行交易
        if self.current_state == 0:  # 做多狀態
            if self.portfolio.is_net_long(self.config.instrument_id):
                return
            if self.portfolio.is_flat(self.config.instrument_id):
                self._submit_buy_order()
            elif self.portfolio.is_net_short(self.config.instrument_id):
                self.close_all_positions(self.config.instrument_id)
                self._submit_buy_order()

        elif self.current_state == 1:  # 做空狀態
            if self.portfolio.is_net_short(self.config.instrument_id):
                return
            if self.portfolio.is_flat(self.config.instrument_id):
                self._submit_sell_order()
            elif self.portfolio.is_net_long(self.config.instrument_id):
                self.close_all_positions(self.config.instrument_id)
                self._submit_sell_order()

    def _submit_buy_order(self):
        """提交買入訂單"""
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(self.config.position_size),
        )
        self.submit_order(order)

    def _submit_sell_order(self):
        """提交賣出訂單"""
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(self.config.position_size),
        )
        self.submit_order(order)

    def on_stop(self):
        """策略停止時輸出統計信息"""
        self.cancel_all_orders(self.config.instrument_id)
        self.close_all_positions(self.config.instrument_id)
        self.log.info(f"Total signals processed: {self.processed_signals}")
