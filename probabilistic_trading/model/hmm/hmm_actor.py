"""
Hidden Markov Model actor implementation.
"""

import numpy as np
import pandas as pd
from nautilus_trader.common.actor import Actor
from nautilus_trader.common.component import TimeEvent
from nautilus_trader.config import ActorConfig
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import DataType
from nautilus_trader.model.identifiers import InstrumentId
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .hmm_data import HMMStateData
from .hmm_model import HMMConfig
from .hmm_model import HMMModel
from .hmm_model import TrainingConfig


class HMMActorConfig(ActorConfig):
    """
    Configuration for the HMM actor.

    Parameters
    ----------
    instrument_id : InstrumentId
        交易商品ID
    bar_type : BarType
        K線類型
    n_states : int
        隱藏狀態數量
    min_training_bars : int
        最小訓練數據量
    update_interval : int
        模型更新間隔(天)
    """

    instrument_id: InstrumentId
    bar_type: str
    n_states: int = 2  # HMM狀態數
    min_training_bars: int = 100  # 最小訓練數據量(狀態數 * 維度 * 10)
    pca_components: int = 5  # PCA降維後的特徵數

    # 新增重新訓練配置
    retrain_interval: int = 7 * 24  # 每多少小時重新訓練一次
    retrain_window_size: int = 672  # 重新訓練使用的資料量
    incremental_training: bool = False  # 是否採用增量訓練而非完全重新訓練


class HMMActor(Actor):
    """
    使用HMM進行狀態預測的Actor。
    整合了HMMModel進行計算, 並發布HMMSignal。
    """

    def __init__(self, config: HMMActorConfig):
        super().__init__(config)

        # 設定
        self.instrument_id = config.instrument_id
        self.bar_type = BarType.from_str(config.bar_type)
        self.min_training_bars = config.min_training_bars

        # 初始化模型和特徵處理器
        self.model = HMMModel(
            config=HMMConfig(
                n_states=config.n_states,  # 隱藏狀態數量
                emission_dim=config.pca_components,  # 使用PCA降維後的特徵
                emission_type="gaussian",
            )
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=config.pca_components)
        # 特徵維度
        self.n_features: int = 0
        self.is_trained = False

        # 初始化訓練追蹤
        self.training_count = 0

    def on_start(self):
        """Actor啟動時被呼叫。訂閱所需的數據流。"""
        self.subscribe_bars(self.bar_type)
        # 初始化重新訓練
        self.clock.set_timer(
            name="retrain_timer",
            start_time=self.clock.utc_now(),
            interval=pd.Timedelta(hours=self.config.retrain_interval),
            callback=self.on_event,
        )

    def on_stop(self):
        """Actor停止時被呼叫。"""
        self.unsubscribe_bars(self.bar_type)

    def on_reset(self):
        """重置Actor狀態。"""
        return

    def on_event(self, event):
        """處理各種事件, 包括計時器事件。"""
        if (
            isinstance(event, TimeEvent)
            and event.name == "retrain_timer"
            and self.training_count > 1
        ):
            self.log.info("重新訓練定時器觸發")
            # 啟動異步重新訓練
            self.retrain_model()

    def on_bar(self, bar: Bar) -> None:
        """
        處理實時K線數據。

        1. 檢查數據量是否足夠
        2. 提取並處理特徵
        3. 根據需要更新模型
        4. 生成預測並發布狀態數據

        Parameters
        ----------
        bar : Bar
            K線數據
        """
        if self.cache.bar_count(self.bar_type) < self.min_training_bars:
            self.log.debug(
                f"Waiting for more data: {self.cache.bar_count(self.bar_type)} / \
                    {self.min_training_bars}"
            )
            return
        try:
            self.log.info(f"current bars: {self.cache.bar_count(self.bar_type)}")
            # 檢查數據量
            features = self._calculate_features()
            # 提取特徵
            pca_features = self._extract_pca_features(features)
            self.log.info(f"Latest feature: {pca_features[-1]}")  # NEW: 最新的bar
            self.log.info(f"Feature has {pca_features.shape} shape")
            # self.log.info(f"Processing {len(pca_features)} bars")
            train_pca_features = pca_features.reshape(1, -1, self.n_features)
            # self.log.info(f"Reshaped train feature has {train_pca_features.shape} shape")

            if not self.is_trained:
                # 首次訓練
                self.log.info("Initial training of HMM model...")
                self.model.fit(
                    train_pca_features, training_config=TrainingConfig(method="em", num_epochs=50)
                )
                self.log.info("Initial training completed")
                self.is_trained = True
                self.training_count += 1

            # 進行預測
            # self.log.info("Predicting state")
            # self.log.info(f"Predicting features has {pca_features.shape} shape")
            # self.log.info(f"Last predicting features: {pca_features[-1]}")

            # 最後一步的狀態機率是更新在[0] NEW: 將bar逆序之後, 最後一個bar是最新的bar
            states = self.model.predict(pca_features)  # 取全部時間步的狀態
            probas = self.model.predict_proba(pca_features)[-1]
            state = states[-1]
            state_proba = probas[state]

            # 發布狀態數據
            # self.log.info(f"Predicted state: {state}, proba: {state_proba}")
            state_data = HMMStateData(
                state=state,
                state_proba=state_proba,
                ts_init=bar.ts_init,
                ts_event=bar.ts_event,
            )
            self.log.info(f"Publishing state: {state_data.state}, proba: {state_data.state_proba}")
            self.publish_data(DataType(HMMStateData), state_data)

        except Exception as e:
            self.log.error("Error processing bar", e)

    def _extract_pca_features(self, features: pd.DataFrame) -> np.ndarray | None:
        """
        提取並轉換特徵。

        1. 計算原始技術指標特徵
        2. 數據清理和驗證
        3. 標準化
        4. PCA降維

        Returns
        -------
        Optional[np.ndarray]
            降維後的特徵矩陣, 如果處理失敗則返回None
        """
        try:
            # 計算原始特徵
            if features.empty:
                self.log.warning("No features calculated")
                return None

            # 檢查無效值
            if features.isna().any().any():
                self.log.warning("Features contain NaN values")
                return None

            if np.isinf(features.to_numpy()).any().any():
                self.log.warning("Features contain infinite values")
                return None

            # 標準化
            features_scaled = self.scaler.fit_transform(features)

            # 檢查標準化後的數據
            if np.isnan(features_scaled).any() or np.isinf(features_scaled).any():
                self.log.warning("Scaling produced invalid values")
                return None

            # PCA降維
            features_pca = self.pca.fit_transform(features_scaled)

            # 驗證PCA結果
            if np.isnan(features_pca).any() or np.isinf(features_pca).any():
                self.log.warning("PCA produced invalid values")
                return None

            # 記錄解釋方差比
            explained_variance_ratio = self.pca.explained_variance_ratio_
            cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
            self.log.debug(f"PCA explained variance ratio: {cumulative_variance_ratio[-1]:.4f}")
            self.n_features = int(len(features_pca[-1]))
            return features_pca

        except Exception as e:
            self.log.error(f"Feature extraction failed: {e!s}")
            self.log.exception("Stack trace:")
            return None

    def _calculate_features(self, windows: int = -1) -> pd.DataFrame:
        """
        從K線數據提取技術特徵。

        Returns
        -------
        pd.DataFrame
            特徵DataFrame
        """
        # 確保有足夠數據
        if self.cache.bar_count(self.bar_type) < self.min_training_bars:
            return pd.DataFrame()

        bars = self.cache.bars(self.bar_type)[:windows]

        # 轉換K線數據為DataFrame
        df_bar = pd.DataFrame(
            [
                {
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                    "ts_event": bar.ts_event,
                }
                for bar in bars
            ]
        ).iloc[::-1]
        # 因為最新的 bar 被存在 `bars[0]`, 我們在這使用`.iloc[::-1]`將最新的 bar 放在最後
        # self.log.info(f"Bar head: {df_bar.tail()}")
        # 計算基礎特徵
        df_bar["price_range"] = df_bar["high"] - df_bar["low"]

        # 計算價量特徵, 包含lags
        close_series = df_bar["close"].to_numpy()
        volume_series = df_bar["volume"].to_numpy()

        for i in range(1, 72, 24):
            # 使用numpy計算returns和其他特徵
            returns = np.zeros_like(close_series)
            returns[i:] = (close_series[i:] - close_series[:-i]) / close_series[:-i]
            df_bar[f"returns_{i}"] = returns

            # 計算log returns
            log_returns = np.zeros_like(close_series)
            with np.errstate(divide="ignore"):
                log_returns[i:] = np.log(close_series[i:]) - np.log(close_series[:-i])
            df_bar[f"log_returns_{i}"] = log_returns

            # 計算其他特徵
            df_bar[f"log_returns_category_{i}"] = np.sign(log_returns)

            # 計算波動率 - 使用numpy的rolling window
            volatility = np.zeros_like(returns)
            for j in range(i, len(returns)):
                window = returns[max(0, j - i) : j]
                volatility[j] = np.nanstd(window) if len(window) > 0 else 0
            df_bar[f"volatility_{i}"] = volatility

            # 計算成交量移動平均 - 使用numpy的rolling window
            volume_ma = np.zeros_like(volume_series)
            for j in range(i, len(volume_series)):
                window = volume_series[max(0, j - i) : j]
                volume_ma[j] = np.nanmean(window) if len(window) > 0 else 0
            df_bar[f"volume_ma_{i}"] = volume_ma

            # 計算momentum
            momentum = np.zeros_like(close_series)
            momentum[i:] = close_series[i:] - close_series[:-i]
            df_bar[f"momentum_{i}"] = momentum

            # 計算volume momentum
            volume_momentum = np.zeros_like(volume_series)
            volume_momentum[i:] = volume_series[i:] - volume_series[:-i]
            df_bar[f"volume_momentum_{i}"] = volume_momentum

        # 刪除NaN值
        df_bar = df_bar.dropna()

        # 選擇用於建模的特徵
        feature_columns = [
            col
            for col in df_bar.columns
            if col not in ["ts_event", "close", "volume", "log_returns_1"]
        ]
        return df_bar[feature_columns]

    def retrain_model(self):
        """重新訓練模型"""
        try:
            # 檢查是否有足夠的新數據
            if self.cache.bar_count(self.bar_type) < self.config.retrain_window_size:
                self.log.warning(
                    f"數據不足, 無法重新訓練. 目前: {self.cache.bar_count(self.bar_type)}, \
                        需要: {self.config.retrain_window_size}"
                )
                return

            self.log.info("開始重新訓練 HMM 模型...")
            # 提取特徵
            features = self._calculate_features(windows=self.config.retrain_window_size)
            pca_features = self._extract_pca_features(features)
            if pca_features is None:
                self.log.error("特徵提取失敗, 無法重新訓練")
                return
            # 重新整形數據
            train_pca_features = pca_features.reshape(1, -1, self.n_features)
            # 配置訓練參數, 可能根據已訓練的次數動態調整
            train_config = TrainingConfig(
                method="em",
                num_epochs=max(20, 50 - self.training_count * 5),  # 隨著訓練次數增加, 減少迭代次數
            )
            if self.config.incremental_training and self.is_trained:
                # 增量訓練
                smoothing_factor = 0.3  # 可調整的參數
                self.model.incremental_fit(
                    train_pca_features,
                    training_config=train_config,
                    smoothing_factor=smoothing_factor,
                )
                self.log.info(f"HMM 模型增量訓練完成, 使用平滑因子 {smoothing_factor}。")
            else:
                # 完全重新訓練
                self.model.fit(train_pca_features, training_config=train_config)
                self.log.info("HMM 模型完全重新訓練完成。")

            # 更新訓練記錄
            self.training_count += 1

            self.log.info(f"HMM 模型重新訓練完成。這是第 {self.training_count} 次訓練。")

        except Exception as e:
            self.log.error(f"重新訓練過程中發生錯誤: {e!s}")
            self.log.exception("Stack trace:")

    def on_save(self) -> dict[str, bytes]:
        """保存Actor狀態。"""
        try:
            if not self.is_trained:
                return {}

            import pickle

            model_state = pickle.dumps(
                {
                    "model_params": self.model._params,
                    "model_props": self.model._props,
                    "is_trained": self.is_trained,
                    "training_count": self.training_count,
                    "last_train_time": self.last_train_time,
                    "scaler": self.scaler,
                    "pca": self.pca,
                    "n_features": self.n_features,
                }
            )

            return {"model_state": model_state}

        except Exception as e:
            self.log.error(f"保存模型時發生錯誤: {e!s}")
            return {}

    def on_load(self, state: dict[str, bytes]) -> None:
        """加載Actor狀態。"""
        try:
            if "model_state" not in state:
                self.log.warning("沒有找到模型狀態, 將重新訓練")
                return

            import pickle

            model_state = pickle.loads(state["model_state"])

            self.model._params = model_state["model_params"]
            self.model._props = model_state["model_props"]
            self.is_trained = model_state["is_trained"]
            self.training_count = model_state["training_count"]
            self.last_train_time = model_state["last_train_time"]
            self.scaler = model_state["scaler"]
            self.pca = model_state["pca"]
            self.n_features = model_state["n_features"]

            self.log.info("成功加載模型狀態")

        except Exception as e:
            self.log.error(f"加載模型狀態時發生錯誤: {e!s}")
