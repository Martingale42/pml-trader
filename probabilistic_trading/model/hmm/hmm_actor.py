"""
Hidden Markov Model actor implementation.
"""

import datetime as dt

import numpy as np
import pandas as pd
from nautilus_trader.common.actor import Actor
from nautilus_trader.config import ActorConfig
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import DataType
from nautilus_trader.model.identifiers import InstrumentId
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .hmm_data import HMMStateData
from .hmm_model import HMMModel


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
    n_states: int = 3  # HMM狀態數
    min_training_bars: int = 864  # 最小訓練數據量(3天的5分鐘K線)
    update_interval: int = 3  # 更新間隔(天)
    pca_components: int = 3  # PCA降維後的特徵數
    device: str = "cpu"  # 使用的設備
    verbose: bool = False  # 是否輸出詳細信息
    early_stopping_patience: int | None = None  # early stopping配置


class HMMActor(Actor):
    """
    使用HMM進行狀態預測的Actor。
    整合了HMMModel進行計算, 並發布HMMSignal。
    """

    def __init__(self, config: HMMActorConfig):
        super().__init__(config)

        # 設定
        self.instrument_id = config.instrument_id
        self.bar_type = BarType.from_str(f"{config.instrument_id}-5-MINUTE-LAST-EXTERNAL")
        self.min_training_bars = config.min_training_bars
        self.update_interval = config.update_interval

        # 初始化模型和特徵處理器
        self.model = HMMModel(
            n_states=config.n_states, device=config.device, verbose=config.verbose
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=config.pca_components)
        # 追蹤上次更新時間
        self.last_update: dt.date | None = None
        # 特徵維度
        self.n_features: int = 0

    def on_start(self):
        """Actor啟動時被呼叫。訂閱所需的數據流。"""
        self.subscribe_bars(self.bar_type)

    def on_stop(self):
        """Actor停止時被呼叫。"""
        self.unsubscribe_bars(self.bar_type)

    def on_reset(self):
        """重置Actor狀態。"""
        self.model = HMMModel(n_states=self.config.n_states)
        self.last_update = None

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
        try:
            # 檢查數據量
            if self.cache.bar_count(self.bar_type) < self.min_training_bars:
                self.log.debug(
                    f"Waiting for more data: {self.cache.bar_count(self.bar_type)} / \
                        {self.min_training_bars}"
                )
                return

            # 提取特徵
            features = self._extract_pca_features()
            if features is None:
                self.log.warning("Failed to extract features")
                return

            # 檢查是否需要更新模型
            current_date = dt.datetime.fromtimestamp(bar.ts_event / 1e9).date()
            model_updated = False

            if self.last_update is None:
                # 首次訓練
                self.log.info("Initial training of HMM model...")
                if self._train_model(features):
                    self.last_update = current_date
                    model_updated = True
            elif (current_date - self.last_update).days >= self.update_interval:
                # 定期更新
                self.log.info("Updating HMM model...")
                if self._train_model(features):
                    self.last_update = current_date
                    model_updated = True

            # 生成預測
            state = self.model.get_state(features)
            state_proba = self.model.predict_proba(features)[-1]
            log_likelihood = self.model.log_probability(features)
            prediction_quality = self.model.get_prediction_quality(features)

            # 檢查模型參數
            if (
                self.model.transition_matrix is None
                or self.model.means is None
                or self.model.sds is None
            ):
                self.log.error("Model parameters not available")
                return

            # 創建並發布狀態數據
            state_data = HMMStateData(
                state=state,
                state_proba=state_proba,
                log_likelihood=log_likelihood,
                transition_matrix=self.model.transition_matrix,
                means=self.model.means,
                sds=self.model.sds,
                prediction_quality=prediction_quality,
                ts_init=bar.ts_init,
                ts_event=bar.ts_event,
            )

            # 發布數據
            self.publish_data(DataType(state_data), state_data)

            # 記錄狀態
            log_msg = (
                f"HMM Update - State: {state}, "
                f"Probability: {float(state_proba[state]):.4f}, "
                f"Quality: {prediction_quality:.4f}"
            )
            if model_updated:
                log_msg += " (Model Updated)"
            self.log.debug(log_msg)

        except Exception as e:
            self.log.error(f"Error in on_bar: {e!s}")
            self.log.exception("Stack trace:")

    def _train_model(self, features: np.ndarray) -> bool:
        """
        訓練或更新模型。

        Parameters
        ----------
        features : np.ndarray
            特徵矩陣

        Returns
        -------
        bool
            訓練是否成功
        """
        try:
            self.model.fit(
                features,
                early_stopping_patience=self.config.early_stopping_patience,
            )
            return True
        except Exception as e:
            self.log.error(f"Model training failed: {e!s}")
            self.log.exception("Stack trace:")
            return False

    def _extract_pca_features(self) -> np.ndarray | None:
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
            features_df = self._calculate_features()
            if features_df.empty:
                self.log.warning("No features calculated")
                return None

            # 檢查無效值
            if features_df.isna().any().any():
                self.log.warning("Features contain NaN values")
                return None

            if np.isinf(features_df.to_numpy()).any().any():
                self.log.warning("Features contain infinite values")
                return None

            # 標準化
            features_scaled = self.scaler.fit_transform(features_df)

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

            return features_pca

        except Exception as e:
            self.log.error(f"Feature extraction failed: {e!s}")
            self.log.exception("Stack trace:")
            return None

    def _calculate_features(self) -> pd.DataFrame:
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

        bars = self.cache.bars(self.bar_type)[-4032:]

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

        # 計算價量特徵, 包含lags
        close_series = df["close"].to_numpy()
        volume_series = df["volume"].to_numpy()

        for i in range(1, 120, 24):
            # 使用numpy計算returns和其他特徵
            returns = np.zeros_like(close_series)
            returns[i:] = (close_series[i:] - close_series[:-i]) / close_series[:-i]
            df[f"returns_{i}"] = returns

            # 計算log returns
            log_returns = np.zeros_like(close_series)
            with np.errstate(divide="ignore"):
                log_returns[i:] = np.log(close_series[i:]) - np.log(close_series[:-i])
            df[f"log_returns_{i}"] = log_returns

            # 計算其他特徵
            df[f"log_returns_category_{i}"] = np.sign(log_returns)

            # 計算波動率 - 使用numpy的rolling window
            volatility = np.zeros_like(returns)
            for j in range(i, len(returns)):
                window = returns[max(0, j - i) : j]
                volatility[j] = np.nanstd(window) if len(window) > 0 else 0
            df[f"volatility_{i}"] = volatility

            # 計算成交量移動平均 - 使用numpy的rolling window
            volume_ma = np.zeros_like(volume_series)
            for j in range(i, len(volume_series)):
                window = volume_series[max(0, j - i) : j]
                volume_ma[j] = np.nanmean(window) if len(window) > 0 else 0
            df[f"volume_ma_{i}"] = volume_ma

            # 計算momentum
            momentum = np.zeros_like(close_series)
            momentum[i:] = close_series[i:] - close_series[:-i]
            df[f"momentum_{i}"] = momentum

            # 計算volume momentum
            volume_momentum = np.zeros_like(volume_series)
            volume_momentum[i:] = volume_series[i:] - volume_series[:-i]
            df[f"volume_momentum_{i}"] = volume_momentum

        # 刪除NaN值
        df = df.dropna()

        # 選擇用於建模的特徵
        feature_columns = [
            col for col in df.columns if col not in ["ts_event", "close", "volume", "log_returns_1"]
        ]
        self.n_features = len(feature_columns)
        return df[feature_columns]

    def on_save(self) -> dict:
        """
        保存Actor狀態。

        Returns
        -------
        dict
            包含Actor狀態的字典
        """
        return {
            "model_params": self.model.save_params(),
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }

    def on_load(self, state: dict) -> None:
        """
        加載Actor狀態。

        Parameters
        ----------
        state : dict
            包含Actor狀態的字典
        """
        if state:
            if state.get("model_params"):
                self.model.load_params(state["model_params"])
            if state.get("last_update"):
                self.last_update = dt.datetime.fromisoformat(state["last_update"]).date()
