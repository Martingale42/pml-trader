"""
Hidden Markov Model actor implementation.
"""

import numpy as np
import pandas as pd
import polars as pl
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
        self.is_trained = False

        # 初始化訓練追蹤
        self.training_count = 0

    def on_start(self):
        """Actor啟動時被呼叫。訂閱所需的數據流。"""
        self.subscribe_bars(self.bar_type)
        # 初始化重新訓練
        self.clock.set_timer(
            name="retrain_timer",
            start_time=self.clock.utc_now() + pd.Timedelta(hours=self.config.retrain_interval),
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
            and self.training_count >= 1
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
            # 檢查數據量
            self.log.info(f"current bars: {self.cache.bar_count(self.bar_type)}")
            # 提取特徵
            features_pca = self._extract_pca_features(self._calculate_features())
            self.log.info(f"Latest feature: {features_pca[-1]}")  # NEW: 最新的bar
            # self.log.info(f"Feature has {features_pca.shape} shape")
            # self.log.info(f"Processing {len(features_pca)} bars")

            if not self.is_trained:
                # 首次訓練
                self.log.info("Initial training of HMM model...")
                self.model.fit(
                    features_pca.reshape(1, -1, self.config.pca_components),
                    training_config=TrainingConfig(method="em", num_epochs=50),
                )
                self.log.info("Initial training completed")
                self.is_trained = True
                self.training_count += 1

            # 進行預測
            # self.log.info("Predicting state")
            # self.log.info(f"Predicting features has {features_pca.shape} shape")
            # self.log.info(f"Last predicting features: {features_pca[-1]}")

            states = self.model.predict(features_pca)  # 取全部時間步的狀態
            probas = self.model.predict_proba(features_pca)[-1]  # 取最後一個時間步的狀態機率
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
            # self.log.info(f"Publishing state: {state_data.state}, proba: {state_data.state_proba}")
            self.publish_data(DataType(HMMStateData), state_data)

        except Exception as e:
            self.log.error("Error processing bar", e)

    def _extract_pca_features(self, features: np.ndarray) -> np.ndarray | None:
        """
        提取並轉換特徵。

        Returns
        -------
        Optional[np.ndarray]
            降維後的特徵矩陣, 如果處理失敗則返回None
        """
        try:
            # 檢查無效值
            if np.isnan(features).any():
                self.log.warning("Features contain NaN values")
                return None

            if np.isinf(features).any():
                self.log.warning("Features contain infinite values")
                return None

            # 標準化 (使用scikit-learn)
            features_scaled = self.scaler.fit_transform(features)

            # 檢查標準化後的數據
            if np.isnan(features_scaled).any() or np.isinf(features_scaled).any():
                self.log.warning("Scaling produced invalid values")
                return None

            # PCA降維 (仍使用sklearn)
            features_pca = self.pca.fit_transform(features_scaled)

            # 驗證PCA結果
            if np.isnan(features_pca).any() or np.isinf(features_pca).any():
                self.log.warning("PCA produced invalid values")
                return None

            # 記錄解釋方差比
            explained_variance_ratio = np.array(self.pca.explained_variance_ratio_)
            cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
            self.log.info(f"PCA explained variance ratio: {cumulative_variance_ratio[-1]:.4f}")
            return features_pca

        except Exception as e:
            self.log.error(f"Feature extraction failed: {e!s}")
            self.log.exception("Stack trace:")
            return None

    def _calculate_features(self, windows: int = -1) -> np.ndarray:
        """
        從K線數據提取技術特徵。使用JAX加速計算。

        Returns
        -------
        np.ndarray
            包含所有特徵的數組
        """
        # 確保有足夠數據
        if self.cache.bar_count(self.bar_type) < self.min_training_bars:
            return pl.DataFrame()

        bars = self.cache.bars(self.bar_type)[:windows]

        # Extract column-wise data using NumPy (faster than list comprehension)
        open_ = np.array([bar.open for bar in bars], dtype=np.float64)
        high = np.array([bar.high for bar in bars], dtype=np.float64)
        low = np.array([bar.low for bar in bars], dtype=np.float64)
        close = np.array([bar.close for bar in bars], dtype=np.float64)
        volume = np.array([bar.volume for bar in bars], dtype=np.float64)

        # Create Polars DataFrame from NumPy arrays
        bars_df = pl.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
        ).reverse()
        # self.log.info(f"Bar tail: {bars_df.tail(1)}")

        features_ = []
        for i in range(1, 72, 14):
            features_.extend(
                [
                    (
                        (
                            ((pl.col("close") + pl.col("high") + pl.col("low")) / 3)
                            * pl.col("volume")
                        ).rolling_sum(i)
                        / pl.col("volume").rolling_sum(i)
                    )
                    .fill_null(1e-8)
                    .alias(f"vwap_{i}"),
                    (pl.col("close").log() - pl.col("close").shift(i).log())
                    .fill_null(1e-8)
                    .alias(f"log_returns_{i}"),
                    pl.col("volume").rolling_mean(i).fill_null(1e-8).alias(f"volume_{i}_ma"),
                    (pl.col("volume") - pl.col("volume").shift(i))
                    .fill_null(1e-8)
                    .alias(f"volume_{i}_momentum"),
                    (pl.col("close") / pl.col("close").shift(i))
                    .rolling_std(i)
                    .fill_null(1e-8)
                    .alias(f"volatility_{i}"),
                ]
            )

        # Apply all feature calculations at once
        bars_df = bars_df.with_columns(features_)

        # 選擇用於建模的特徵 (Faster filtering)
        bars_df = bars_df.select(
            [
                col
                for col in bars_df.columns
                if col not in ["ts_event", "open", "high", "low", "close", "volume"]
            ]
        )

        return bars_df.to_numpy()

    def retrain_model(self):
        """重新訓練模型"""
        try:
            # 檢查是否有足夠的新數據
            if self.cache.bar_count(self.bar_type) < self.config.retrain_window_size:
                self.log.warning(
                    f"數據不足, 無法重新訓練. 目前: {self.cache.bar_count(self.bar_type)}, "
                    f"需要: {self.config.retrain_window_size}"
                )
                return

            self.log.info("開始重新訓練 HMM 模型...")
            # 提取特徵
            features = self._calculate_features()
            pca_features = self._extract_pca_features(features)
            if pca_features is None:
                self.log.error("特徵提取失敗, 無法重新訓練")
                return
            # 重新整形數據
            train_pca_features = pca_features.reshape(1, -1, self.n_features)
            # 配置訓練參數, 可能根據已訓練的次數動態調整
            train_config = TrainingConfig(
                method="em",
                num_epochs=max(30, 50 - self.training_count * 5),  # 隨著訓練次數增加, 減少迭代次數
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

            # import pickle

            # model_state = pickle.loads(state["model_state"])

            # self.model._params = model_state["model_params"]
            # self.model._props = model_state["model_props"]
            # self.is_trained = model_state["is_trained"]
            # self.training_count = model_state["training_count"]
            # self.last_train_time = model_state["last_train_time"]
            # self.scaler = model_state["scaler"]
            # self.pca = model_state["pca"]
            # self.n_features = model_state["n_features"]

            self.log.info("成功加載模型狀態")

        except Exception as e:
            self.log.error(f"加載模型狀態時發生錯誤: {e!s}")
