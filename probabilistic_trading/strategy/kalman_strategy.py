import pickle  # noqa
from decimal import Decimal

import numpy as np
import pandas as pd
from ..model.kalman.kalman_model import KalmanFilter
from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import BarType
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import PositionSide  # noqa
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import PositionId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.trading.strategy import Strategy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler  # noqa


class KFStrategyConfig(StrategyConfig):
    instrument_id: InstrumentId
    pca_features: int = 6
    position_size: Decimal = Decimal("100.0")


class KFStrategy(Strategy):
    def __init__(self, config: KFStrategyConfig):
        super().__init__(config)
        self.instrument: Instrument = None
        self.bar_type = BarType.from_str(f"{config.instrument_id}-5-MINUTE-LAST-EXTERNAL")
        self.n_features = 0  # 特徵數量

        # 初始化模型和特徵處理器
        self.kf = None
        self.scaler = None
        self.pca = None

    def on_start(self):
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.config.instrument_id}")
            self.stop()
            return

        self.kf = KalmanFilter(
            dim_state=self.n_features,
            initial_state=np.zeros(self.n_features),
            initial_covariance=0.25 * np.eye(self.n_features),  # 降低初始協方差
            process_noise_cov=0.05 * np.eye(self.n_features),  # 降低過程噪聲
            measure_noise_cov=0.05 * np.eye(self.n_features),  # 降低測量噪聲
            forgetting_factor=0.95,  # 降低遺忘因子以增加適應性
            adaptive_estimation=True,
        )
        self.scaler = None
        self.pca = PCA(n_components=self.config.pca_features)

        self.subscribe_bars(self.bar_type)
        self.request_bars(self.bar_type)

    def on_bar(self, bar: Bar) -> None:
        self.log.info(repr(bar), LogColor.CYAN)

        # 檢查是否有足夠的數據進行初始訓練
        if self.cache.bar_count(self.bar_type) < 100:
            self.log.info(f"Waiting for more bars: {self.cache.bar_count(self.bar_type)} / 100")
            return

        try:
            features = self._extract_pca_features()
            if features is None:
                return
            self._execute_trades()
        except Exception as e:
            self.log.error(f"Error in bar processing: {e!s}")
            return

    def _long_logic(self, features: pd.DataFrame):
        """做多邏輯"""
        return False

    def _short_logic(self, features: pd.DataFrame):
        """做空邏輯"""
        return False

    def _execute_trades(self):
        """執行交易邏輯"""
        # BUY LOGIC
        if self._long_logic():
            if self.portfolio.is_flat(self.config.instrument_id):
                self.cancel_all_orders(self.config.instrument_id)
                self._long()
            elif self.portfolio.is_net_short(self.config.instrument_id):
                self.close_all_positions(self.config.instrument_id)
                self.cancel_all_orders(self.config.instrument_id)
                self._long()
        # SELL LOGIC
        elif self._short_logic():
            if self.portfolio.is_flat(self.config.instrument_id):
                self.cancel_all_orders(self.config.instrument_id)
                self._short()
            elif self.portfolio.is_net_long(self.config.instrument_id):
                self.close_all_positions(self.config.instrument_id)
                self.cancel_all_orders(self.config.instrument_id)
                self._short()

    def _long(self) -> None:
        """開多倉"""
        order: MarketOrder = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(self._calculate_quantity()),
        )
        position_id = PositionId(f"{self.config.instrument_id}-LONG")
        self.submit_order(order, position_id)

    def _short(self) -> None:
        """開空倉"""
        order: MarketOrder = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(self._calculate_quantity()),
        )
        position_id = PositionId(f"{self.config.instrument_id}-SHORT")
        self.submit_order(order, position_id)

    def _calculate_quantity(self) -> Decimal:
        """計算交易數量"""
        if self.cache.last_price(self.config.instrument_id) is None:
            return Decimal("0")
        return Decimal(
            str(self.config.position_size / self.cache.last_price(self.config.instrument_id))
        )

    def _process_signal(self):
        """處理信號並生成特徵"""

    def _extract_pca_features(self) -> pd.DataFrame:
        """提取特徵"""
        if self._calculate_features().empty:
            return None
        else:
            return self.pca.fit_transform(self._calculate_features())

    def _calculate_features(self) -> pd.DataFrame:
        """
        從K線數據提取技術特徵。
        """
        # 確保有足夠數據
        if self.cache.bar_count(self.bar_type) < 120:
            return pd.DataFrame()

        bars = self.cache.bars(self.bar_type)[:]

        # 轉換K線數據為DataFrame
        df = pd.DataFrame(
            [
                {
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                    "ts_event": bar.ts_event,
                }
                for bar in bars[:]
            ]
        )

        # 計算基礎特徵
        df["price_range"] = df["high"] - df["low"]

        # 計算價量特徵，包含lags
        for i in range(1, 120, 24):
            df[f"returns_{i}"] = df["close"].pct_change(i)
            df[f"log_returns_{i}"] = np.log(df["close"]).diff(i)
            df[f"log_returns_category_{i}"] = np.sign(df[f"log_returns_{i}"])
            df[f"volatility_{i}"] = df[f"returns_{1}"].rolling(i).std()
            df[f"volume_ma_{i}"] = df["volume"].rolling(i).mean()
            df[f"momentum_{i}"] = df["close"] - df["close"].shift(i)
            df[f"volume_momentum_{i}"] = df["volume"] - df["volume"].shift(i)

        # 刪除NaN值
        df = df.dropna()
        # 選擇用於建模的特徵
        feature_columns = [
            col for col in df.columns if col not in ["ts_event", "close", "volume", "log_returns_1"]
        ]
        self.n_features = len(feature_columns)
        return df[feature_columns]

    # 以下為策略生命周期方法
    def on_stop(self):
        # Remember and log end time of strategy
        self.cancel_all_orders(self.config.instrument_id)
        self.close_all_positions(self.config.instrument_id)
        self.unsubscribe_bars(self.bar_type)

    def on_reset(self) -> None:
        """重置策略時的操作"""
        # 重置其他組件
        self.kf = None
        self.scaler = None
        self.pca = None

    def on_dispose(self) -> None:
        """清理資源"""
        self.kf = None
        self.scaler = None
        self.pca = None

    def on_save(self) -> dict[str, bytes]:
        """保存策略狀態"""

    def on_load(self, state: dict[str, bytes]) -> None:
        """加載策略狀態"""
