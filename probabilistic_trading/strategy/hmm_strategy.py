from decimal import Decimal

from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.data import Data
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import DataType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import PositionId
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.trading.strategy import Strategy

from probabilistic_trading.model.hmm.hmm_data import HMMStateData


class HMMStrategyConfig(StrategyConfig):
    instrument_id: InstrumentId
    bar_type: str
    prob_threshold: float = 0.75
    position_size: Decimal = Decimal("100.0")


class HMMStrategy(Strategy):
    def __init__(self, config: HMMStrategyConfig):
        super().__init__(config)
        # 策略配置
        self.bar_type = BarType.from_str(config.bar_type)
        self.current_state = None
        self.state_proba = None

    def on_start(self):
        """策略啟動時被呼叫"""
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"No instrument found for {self.config.instrument_id}")
            return
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
        if isinstance(data, HMMStateData):
            self.log.debug(f"Received HMMStateData: {data.state}, {data.state_proba}")
        else:
            self.log.debug("No data received")
            return

        # 獲取狀態和概率
        self.current_state = data.state
        self.state_proba = data.state_proba

        if self.state_proba < self.config.prob_threshold:
            return

    def on_bar(self, bar: Bar):
        self.log.info(repr(bar), LogColor.CYAN)
        self.log.info(f"Current state: {self.current_state}, State proba: {self.state_proba}")
        # self.log.info(f"Bar count: {self.cache.bar_count(self.bar_type)}")
        # self.log.info(f"Bar shape: {len(self.cache.bars(self.bar_type))}")
        if self.current_state is None or self.state_proba is None:
            self.log.debug("Waiting for state data")
            return  # 等待狀態數據
        self._execute_trades()

    def _execute_trades(self):
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
        if self.current_state == 0 and self.state_proba > self.config.prob_threshold:
            if self.portfolio.is_net_long(self.config.instrument_id):
                return
            if self.portfolio.is_flat(self.config.instrument_id):
                self.cancel_all_orders(self.config.instrument_id)
                self._long()
            elif self.portfolio.is_net_short(self.config.instrument_id):
                self.cancel_all_orders(self.config.instrument_id)
                self.close_all_positions(self.config.instrument_id)
                self._long()
        # SELL LOGIC
        elif self.current_state == 1 and self.state_proba > self.config.prob_threshold:
            if self.portfolio.is_net_short(self.config.instrument_id):
                return
            if self.portfolio.is_flat(self.config.instrument_id):
                self.cancel_all_orders(self.config.instrument_id)
                self._short()
            elif self.portfolio.is_net_long(self.config.instrument_id):
                self.cancel_all_orders(self.config.instrument_id)
                self.close_all_positions(self.config.instrument_id)
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
        position_id = PositionId(
            f"{self.config.instrument_id}-LONG-{self.cache.bar_count(self.bar_type)}"
        )
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
        position_id = PositionId(
            f"{self.config.instrument_id}-SHORT-{self.cache.bar_count(self.bar_type)}"
        )
        self.submit_order(order, position_id)

    def on_stop(self):
        """策略停止時被呼叫"""
        self.cancel_all_orders(self.config.instrument_id)
        self.close_all_positions(self.config.instrument_id)
        self.unsubscribe_bars(self.bar_type)
        self.unsubscribe_data(DataType(HMMStateData))
        return

    def on_reset(self) -> None:
        """重置策略時的操作"""
        return

    def on_dispose(self) -> None:
        """清理資源"""
        return
