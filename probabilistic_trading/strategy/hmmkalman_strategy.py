import datetime as dt
import pickle
from decimal import Decimal

import numpy as np
import pandas as pd
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
from sklearn.preprocessing import StandardScaler

from ..model.hmm.hmm_model import HMMModel
from ..model.kalman.kalman_model import KalmanFilter


class HMMTradingConfig(StrategyConfig):
    instrument_id: InstrumentId
    pca_features: int = 4
    hmm_states: int = 3
    min_prob_threshold: float = 0.7
    position_size: Decimal = Decimal("100.0")


class HMMTradingStrategy(Strategy):
    def __init__(self, config: HMMTradingConfig):
        super().__init__(config)
        self.start_time = None
        self.end_time = None
        suffix_5m = "-5-MINUTE-LAST-EXTERNAL"
        self.instrument: Instrument = None
        self.bar_type_5m = BarType.from_str(f"{config.instrument_id}{suffix_5m}")
        self.n_features = 9  # 特徵數量
        # 初始化模型和特徵處理器
        self.kf = KalmanFilter(
            dim_state=self.n_features,
            initial_state=np.zeros(self.n_features),
            initial_covariance=0.25 * np.eye(self.n_features),  # 降低初始協方差
            process_noise_cov=0.25 * np.eye(self.n_features),  # 降低過程噪聲
            measure_noise_cov=0.25 * np.eye(self.n_features),  # 降低測量噪聲
            forgetting_factor=0.95,  # 降低遺忘因子以增加適應性
            adaptive_estimation=True,
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=config.pca_features)
        # 初始化HMM模型
        self.hmm = None  # 延遲初始化
        self.hmm_cache_size = 1000

        # 存儲歷史數據
        self.bars_5m = []

        # 記錄最後一次訓練的日期
        self.last_train_date = None

    def on_start(self):
        self.start_time = dt.datetime.now()
        self.log.info(f"Strategy started at: {self.start_time}")

        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.config.instrument_id}")
            self.stop()
            return

        self.subscribe_bars(self.bar_type_5m)
        self.request_bars(self.bar_type_5m)

    def on_bar(self, bar: Bar) -> None:
        self.log.info(repr(bar), LogColor.CYAN)
        self.bars_5m.append(bar)

        # 檢查是否有足夠的數據進行初始訓練
        if len(self.bars_5m) < 100:
            self.log.info(f"Waiting for more data: {len(self.bars_5m)}/100")
            return

        # 獲取當前bar的日期
        current_date = pd.Timestamp(bar.ts_event).date()
        try:
            features = self._process_signal()
            if features is None:
                return

            # 延遲初始化HMM模型
            if self.hmm is None:
                self.log.info("Initializing HMM model...")
                self.hmm = HMMModel(n_states=self.config.hmm_states, cache_size=self.hmm_cache_size)

            if not self._is_hmm_fitted():
                self.log.info("Initial training of HMM model...")
                try:
                    self.hmm.fit(features, batch_size=100)
                    self.last_train_date = current_date
                except Exception as e:
                    self.log.error(f"HMM training failed: {e!s}")
                    self._reset_hmm()
                return

            if self.last_train_date != current_date:
                self.log.info("Daily update of HMM model...")
                try:
                    self.hmm.fit(features)
                    self.last_train_date = current_date
                except Exception as e:
                    self.log.error(f"HMM update failed: {e!s}")
                    return

            try:
                state = self.hmm.predict(features)
                state_prob = self.hmm.predict_proba(features)
                # 添加模型診斷
                log_likelihood = self.hmm.log_probability(features)
                self.log.info(f"Model log-likelihood: {log_likelihood}")
                self._execute_trades(state, state_prob)
            except Exception as e:
                self.log.error(f"Prediction error: {e!s}")

        except Exception as e:
            self.log.error(f"Error in bar processing: {e!s}")
            self._reset_hmm()

    def _is_hmm_fitted(self):
        """檢查HMM模型是否已經訓練並可用"""
        if self.hmm is None:
            return False
        params = self.hmm.get_parameters()
        if params is None:
            return False
        return params.validate()

    def _reset_hmm(self):
        """重置 HMM 模型到初始狀態"""
        self.hmm = None
        self.last_train_date = None

    def _process_signal(self):
        """處理信號並生成特徵"""
        try:
            if len(self.bars_5m) < 100:
                self.log.info(f"Insufficient data: {len(self.bars_5m)}/100")
                return None

            # 1. 特徵提取
            features = self._extract_features()
            if features.empty:
                self.log.warning("Empty features dataframe")
                return None

            # 2. 處理缺失值
            missing_count = features.isnull().sum().sum()
            if missing_count > 0:
                self.log.warning(f"Found {missing_count} missing values")
                features = features.ffill().bfill()

            # 3. 檢查無效值
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                self.log.warning("Invalid values detected in features")
                return None

            # 4. 數據預處理和特徵選擇
            features_array = features.values

            # 計算特徵相關性
            corr_matrix = np.corrcoef(features_array.T)
            np.fill_diagonal(corr_matrix, 0)  # 忽略自相關

            # 移除高度相關的特徵
            high_corr_threshold = 0.95
            high_corr_pairs = np.where(np.abs(corr_matrix) > high_corr_threshold)
            features_to_remove = set()
            for i, j in zip(*high_corr_pairs):
                if i < j:  # 避免重複
                    # 選擇方差較小的特徵移除
                    var_i = np.var(features_array[:, i])
                    var_j = np.var(features_array[:, j])
                    features_to_remove.add(j if var_i > var_j else i)

            # 保留的特徵索引
            keep_features = [
                i for i in range(features_array.shape[1]) if i not in features_to_remove
            ]
            if len(keep_features) < 3:  # 確保至少保留3個特徵
                self.log.warning("Insufficient independent features")
                return None

            features_array = features_array[:, keep_features]
            selected_features_count = len(keep_features)
            self.log.info(f"Selected {selected_features_count} features")

            # 更新Kalman濾波器維度
            if selected_features_count != self.n_features:
                self.n_features = selected_features_count
                self.kf = KalmanFilter(
                    dim_state=self.n_features,
                    initial_state=np.zeros(self.n_features),
                    initial_covariance=0.05 * np.eye(self.n_features),
                    process_noise_cov=0.05 * np.eye(self.n_features),
                    measure_noise_cov=0.05 * np.eye(self.n_features),
                    forgetting_factor=0.95,
                    adaptive_estimation=True,
                )

            # 5. 數據預處理和Kalman濾波
            try:
                # 移除極端值
                for col in range(features_array.shape[1]):
                    q1, q3 = np.percentile(features_array[:, col], [25, 75])
                    iqr = q3 - q1
                    lower_bound = q1 - 3 * iqr
                    upper_bound = q3 + 3 * iqr
                    features_array[:, col] = np.clip(
                        features_array[:, col], lower_bound, upper_bound
                    )

                # 標準化特徵
                features_array = (features_array - np.mean(features_array, axis=0)) / (
                    np.std(features_array, axis=0) + 1e-8
                )

                # 檢查數據的條件數
                cov = np.cov(features_array.T)
                cond_num = np.linalg.cond(cov)
                if cond_num > 1e8:  # 降低條件數閾值
                    self.log.warning(f"Poor data conditioning (condition number: {cond_num:.2e})")
                    return None

                # 應用Kalman濾波
                features_smoothed = self.kf.smooth(features_array)[0]

                # 驗證平滑結果
                if not np.all(np.isfinite(features_smoothed)):
                    self.log.error("Kalman smoothing produced invalid values")
                    return None

            except np.linalg.LinAlgError as e:
                self.log.error(f"Kalman smoothing failed: {e!s}")
                return None

            # 6. 標準化
            try:
                features_scaled = self.scaler.fit_transform(features_smoothed)
            except Exception as e:
                self.log.error(f"Standardization failed: {e!s}")
                return None

            # 7. 檢查標準化後的數據
            if not np.all(np.isfinite(features_scaled)):
                self.log.warning("Invalid values after standardization")
                return None

            # 8. PCA降維
            try:
                features_pca = self.pca.fit_transform(features_scaled)
                explained_var = np.sum(self.pca.explained_variance_ratio_)
                self.log.info(f"PCA explained variance: {explained_var:.2%}")
            except Exception as e:
                self.log.error(f"PCA transformation failed: {e!s}")
                return None

            # 9. 最終檢查
            if not np.all(np.isfinite(features_pca)):
                self.log.warning("Invalid values after PCA")
                return None

            features_pca = features_pca.reshape(-1, self.config.pca_features)
            return features_pca

        except Exception as e:
            self.log.error(f"Signal processing error: {e!s}")
            return None

    def _execute_trades(self, state, state_prob):
        """執行交易邏輯"""
        # BUY LOGIC
        if state == 0 and state_prob[0] > self.config.min_prob_threshold:
            if self.portfolio.is_flat(self.config.instrument_id):
                self.cancel_all_orders(self.config.instrument_id)
                self._long()
            elif self.portfolio.is_net_short(self.config.instrument_id):
                self.close_all_positions(self.config.instrument_id)
                self.cancel_all_orders(self.config.instrument_id)
                self._long()
        # SELL LOGIC
        elif state == 1 and state_prob[1] > self.config.min_prob_threshold:
            if self.portfolio.is_flat(self.config.instrument_id):
                self.cancel_all_orders(self.config.instrument_id)
                self._short()
            elif self.portfolio.is_net_long(self.config.instrument_id):
                self.close_all_positions(self.config.instrument_id)
                self.cancel_all_orders(self.config.instrument_id)
                self._short()

    def _long(self) -> None:
        """開多倉"""
        current_price = self.bars_5m[-1].close
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
        current_price = self.bars_5m[-1].close
        usdt_size = self.config.position_size
        quantity = usdt_size / Decimal(str(current_price))

        order: MarketOrder = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(quantity),
        )
        position_id = PositionId(f"{self.config.instrument_id}-SHORT")
        self.submit_order(order, position_id)

    def _extract_features(self) -> pd.DataFrame:
        """
        從K線數據提取技術特徵。
        """
        # 確保有足夠數據
        if len(self.bars_5m) < 100:
            return pd.DataFrame()

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
                for bar in self.bars_5m[-100:]  # 使用最近100根K線
            ]
        )

        # 計算基礎特徵
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"]).diff()
        df["volatility"] = df["returns"].rolling(20).std()
        df["volume_ma"] = df["volume"].rolling(20).mean()
        df["volume_std"] = df["volume"].rolling(20).std()

        # 計算技術指標
        df["rsi"] = self._calculate_rsi(df["close"], 14)
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = self._calculate_bollinger_bands(
            df["close"], 20
        )
        # 刪除NaN值
        df = df.dropna()
        # 選擇用於建模的特徵
        feature_columns = [
            "returns",
            "log_returns",
            "volatility",
            "volume_ma",
            "volume_std",
            "rsi",
            "bb_upper",
            "bb_middle",
            "bb_lower",
        ]
        self.n_features = len(feature_columns)
        return df[feature_columns]

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """計算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> tuple:
        """計算布林通道"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * 2)
        lower = middle - (std * 2)
        return upper, middle, lower

    def on_stop(self):
        # Remember and log end time of strategy
        self.end_time = dt.datetime.now()
        self.log.info(f"Strategy finished at: {self.end_time}")
        self.cancel_all_orders(self.config.instrument_id)
        self.close_all_positions(self.config.instrument_id)
        self.unsubscribe_bars(self.bar_type_5m)

    def on_reset(self) -> None:
        """重置策略時的操作"""
        # 重置其他組件
        self.kf = KalmanFilter(
            dim_state=self.n_features,
            initial_state=np.zeros(self.n_features),
            initial_covariance=0.05 * np.eye(self.n_features),
            process_noise_cov=0.05 * np.eye(self.n_features),
            measure_noise_cov=0.05 * np.eye(self.n_features),
            forgetting_factor=0.95,
            adaptive_estimation=True,
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.config.pca_features)
        self.hmm = None
        self.bars_5m = []
        self.last_train_date = None

    def on_dispose(self) -> None:
        """清理資源"""
        # 清空所有數據
        self.bars_5m = None
        # 釋放模型
        self.hmm = None
        self.kf = None
        self.scaler = None
        self.pca = None

    def on_save(self) -> dict[str, bytes]:
        """保存策略狀態"""
        return {
            "hmm_model": pickle.dumps(self.hmm) if self.hmm is not None else None,
            "scaler_params": (
                pickle.dumps({"mean_": self.scaler.mean_, "scale_": self.scaler.scale_})
                if hasattr(self.scaler, "mean_")
                else None
            ),
            "pca_components": (
                pickle.dumps(self.pca.components_) if hasattr(self.pca, "components_") else None
            ),
        }

    def on_load(self, state: dict[str, bytes]) -> None:
        """加載策略狀態"""
        if state:
            if state.get("hmm_model"):
                self.hmm = pickle.loads(state["hmm_model"])

            if state.get("scaler_params"):
                scaler_params = pickle.loads(state["scaler_params"])
                self.scaler.mean_ = scaler_params["mean_"]
                self.scaler.scale_ = scaler_params["scale_"]

            if state.get("pca_components"):
                self.pca.components_ = pickle.loads(state["pca_components"])
