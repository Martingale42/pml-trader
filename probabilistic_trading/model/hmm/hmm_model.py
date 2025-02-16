# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from dataclasses import dataclass
from functools import partial

# %%
import jax.numpy as jnp
import jax.random as jr
import optax
from dynamax.hidden_markov_model import DiagonalGaussianHMM
from dynamax.hidden_markov_model import GaussianHMM
from dynamax.hidden_markov_model import SharedCovarianceGaussianHMM
from dynamax.hidden_markov_model import SphericalGaussianHMM
from jax import vmap


# %%
@dataclass(frozen=True)
class HMMConfig:
    """
    HMM模型配置。

    Parameters
    ----------
    n_states : int
        隱藏狀態數量
    emission_dim : int
        觀測維度
    emission_type : str
        發射分布類型 ('gaussian', 'diagonal', 'spherical', 'shared')
    transition_matrix_stickiness : float, optional
        狀態轉移矩陣的粘性係數
    """

    n_states: int
    emission_dim: int
    emission_type: str = "gaussian"
    transition_matrix_stickiness: float = 10.0
    seed: int = 42

    def validate(self) -> None:
        if self.n_states <= 0:
            raise ValueError("n_states must be positive")
        if self.emission_dim <= 0:
            raise ValueError("emission_dim must be positive")
        valid_types = {"gaussian", "diagonal", "spherical", "shared"}
        if self.emission_type not in valid_types:
            raise ValueError(f"emission_type must be one of {valid_types}")


# %%
@dataclass
class TrainingConfig:
    """
    訓練配置。

    Parameters
    ----------
    method : str
        訓練方法 ('em' or 'sgd')
    num_epochs : int
        訓練輪數
    batch_size : Optional[int]
        SGD的批次大小, None表示使用全部數據
    learning_rate : float
        SGD的學習率
    """

    method: str = "em"
    num_epochs: int = 100
    batch_size: int | None = None
    learning_rate: float = 1e-2


# %%
class HMMModel:
    """使用dynamax實現的HMM模型。"""

    def __init__(self, config: HMMConfig):
        self.config = config
        config.validate()

        # 初始化模型
        self._init_model()

        self._is_fitted = False
        self.training_history: list[dict] = []

    def _init_model(self) -> None:
        """初始化dynamax HMM模型。"""
        model_classes = {
            "gaussian": GaussianHMM,
            "diagonal": DiagonalGaussianHMM,
            "spherical": SphericalGaussianHMM,
            "shared": SharedCovarianceGaussianHMM,
        }

        model_class = model_classes[self.config.emission_type]
        self._model = model_class(
            num_states=self.config.n_states,
            emission_dim=self.config.emission_dim,
            transition_matrix_stickiness=self.config.transition_matrix_stickiness,
        )

        # 初始化參數
        key = jr.PRNGKey(self.config.seed)
        self._params, self._props = self._model.initialize(key)

    def fit(
        self,
        train_data: jnp.ndarray,
        training_config: TrainingConfig,
    ) -> "HMMModel":
        """
        訓練模型。

        Parameters
        ----------
        train_data : jnp.ndarray
            訓練數據, shape為(n_batches, n_timesteps, emission_dim)
        training_config : TrainingConfig
            訓練配置

        Returns
        -------
        self : HMMModel
            訓練後的模型實例
        """
        if training_config.method == "em":
            self._fit_em(train_data, training_config)
        elif training_config.method == "sgd":
            self._fit_sgd(train_data, training_config)
        else:
            raise ValueError(f"Unknown training method: {training_config.method}")

        self._is_fitted = True
        return self

    def _fit_em(
        self,
        train_data: jnp.ndarray,
        training_config: TrainingConfig,
    ) -> None:
        """使用EM算法訓練模型。"""
        self._params, log_probs = self._model.fit_em(
            self._params,
            self._props,
            train_data,
            num_iters=training_config.num_epochs,
        )
        self.training_history = [{"epoch": i, "log_prob": lp} for i, lp in enumerate(log_probs)]

    def _fit_sgd(
        self,
        train_data: jnp.ndarray,
        training_config: TrainingConfig,
    ) -> None:
        """使用SGD算法訓練模型。"""
        optimizer = optax.sgd(learning_rate=training_config.learning_rate, momentum=0.95)

        self._params, losses = self._model.fit_sgd(
            self._params,
            self._props,
            train_data,
            optimizer=optimizer,
            batch_size=training_config.batch_size,
            num_epochs=training_config.num_epochs,
            key=jr.PRNGKey(self.config.seed),
        )
        self.training_history = [{"epoch": i, "loss": l} for i, l in enumerate(losses)]

    def predict(self, data: jnp.ndarray) -> jnp.ndarray:
        """
        使用Viterbi算法預測最可能的狀態序列。

        Parameters
        ----------
        data : jnp.ndarray
            輸入數據

        Returns
        -------
        jnp.ndarray
            預測的狀態序列
        """
        self._check_is_fitted()
        # self._validate_predict_data(data)

        # 確保數據是正確的形狀
        # if data.ndim == 2:
        #     data = data.reshape(1, *data.shape)

        return self._model.most_likely_states(self._params, data)[-1]  # 取第一個序列的結果

    def predict_proba(self, data: jnp.ndarray) -> jnp.ndarray:
        """
        計算狀態後驗概率。

        Parameters
        ----------
        data : jnp.ndarray
            輸入數據

        Returns
        -------
        jnp.ndarray
            後驗概率矩陣
        """
        self._check_is_fitted()
        # self._validate_predict_data(data)

        # 確保數據是正確的形狀
        # if data.ndim == 2:
        #     data = data.reshape(1, *data.shape)

        posterior = self._model.smoother(self._params, data)
        return posterior.smoothed_probs[-1]  # 取第一個序列的結果

    def filter(self, data: jnp.ndarray) -> jnp.ndarray:
        """
        使用前向演算法計算過濾後的狀態概率。

        Parameters
        ----------
        data : jnp.ndarray
            輸入數據

        Returns
        -------
        jnp.ndarray
            過濾後的狀態概率
        """
        self._check_is_fitted()
        posterior = self._model.filter(self._params, data)
        return posterior.filtered_probs

    def log_probability(self, data: jnp.ndarray) -> float:
        """計算數據的對數似然。"""
        self._check_is_fitted()
        return float(vmap(partial(self._model.marginal_log_prob, self._params))(data).sum())

    def _validate_data(self, data: jnp.ndarray) -> None:
        """驗證輸入數據的維度。"""
        if data.ndim != 3:
            raise ValueError(
                f"Expected 3D array (n_batches, n_timesteps, emission_dim), got {data.ndim}D"
            )
        if data.shape[-1] != self.config.emission_dim:
            raise ValueError(
                f"Expected emission dimension {self.config.emission_dim}, " f"got {data.shape[-1]}"
            )

    def _validate_predict_data(self, data: jnp.ndarray) -> None:
        """驗證預測數據的維度。"""
        if data.ndim not in (2, 3):
            raise ValueError(
                f"Data must be 2D (num_timesteps, emission_dim) or "
                f"3D (num_sequences, num_timesteps, emission_dim), "
                f"got {data.ndim}D"
            )
        if data.shape[-1] != self.config.emission_dim:
            raise ValueError(
                f"Expected emission dimension {self.config.emission_dim}, " f"got {data.shape[-1]}"
            )

    def _check_is_fitted(self) -> None:
        """檢查模型是否已訓練。"""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

    @property
    def transition_matrix(self) -> jnp.ndarray:
        """獲取轉移矩陣。"""
        self._check_is_fitted()
        return self._params.transitions.transition_matrix

    @property
    def params(self):
        """獲取模型參數。"""
        self._check_is_fitted()
        return self._params


# %%
