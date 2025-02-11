"""
Hidden Markov Model implementation using pomegranate.
"""

import pickle
from typing import Any

import numpy as np
import torch
from pomegranate.distributions import Normal
from pomegranate.hmm import DenseHMM


class HMMModel:
    """
    使用pomegranate實現的隱馬爾可夫模型。

    提供模型的建立、訓練、預測和參數管理功能。
    使用PyTorch作為計算後端, 支援GPU加速。
    支持任意維度的特徵輸入, 適配PCA降維後的特徵。
    """

    def __init__(
        self,
        n_states: int,
        random_seed: int = 42,
        device: str = "cpu",
        verbose: bool = False,
    ):
        """
        初始化HMM模型。

        Parameters
        ----------
        n_states : int
            隱藏狀態的數量
        random_seed : int
            隨機種子, 用於結果重現
        device : str
            使用的設備, 'cpu'或'cuda'
        verbose : bool
            是否輸出詳細信息
        """
        self.n_states = n_states
        self.random_seed = random_seed
        self.device = device
        self.verbose = verbose
        self.model: DenseHMM | None = None
        self.n_features: int | None = None

        # 模型參數
        self.transition_matrix: np.ndarray | None = None
        self.means: np.ndarray | None = None
        self.sds: np.ndarray | None = None

        # 訓練歷史
        self.training_history: list[dict[str, Any]] = []

        # 設置隨機種子
        torch.manual_seed(random_seed)
        np.random.Generator(np.random.PCG64(random_seed))

    def to(self, device: str) -> "HMMModel":
        """
        將模型移動到指定設備。

        Parameters
        ----------
        device : str
            目標設備, 'cpu'或'cuda'

        Returns
        -------
        HMMModel
            模型實例
        """
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        return self

    def build_model(self, data: np.ndarray) -> None:
        """
        建立pomegranate的HMM模型。

        Parameters
        ----------
        data : np.ndarray
            輸入數據, shape為(n_samples, n_features)
            n_features可以是任意維度, 通常是PCA降維後的特徵數
        """
        self.n_features = data.shape[1]
        if self.verbose:
            print(f"Building model with {self.n_features} features")

        # 創建多變量Normal分布列表
        distributions = [Normal(covariance_type="diag") for _ in range(self.n_states)]

        # 初始化轉移矩陣為均勻分布
        edges = np.ones((self.n_states, self.n_states)) / self.n_states
        starts = np.ones(self.n_states) / self.n_states
        ends = np.ones(self.n_states) / self.n_states

        # 創建DenseHMM模型
        self.model = DenseHMM(
            distributions=distributions,
            edges=edges,
            starts=starts,
            ends=ends,
            random_state=self.random_seed,
            verbose=self.verbose,
        )

        # 移動到指定設備
        self.model = self.model.to(self.device)

    def fit(
        self,
        data: np.ndarray,
        n_draws: int = 1000,
        tune: int = 1000,
        chains: int = 2,
        priors: np.ndarray | None = None,
        validation_data: np.ndarray | None = None,
        early_stopping_patience: int | None = None,
        min_improvement: float = 1e-4,
    ) -> None:
        """
        訓練模型。

        Parameters
        ----------
        data : np.ndarray
            訓練數據, shape為(n_samples, n_features)
            n_features可以是任意維度, 通常是PCA降維後的特徵數
        n_draws : int
            迭代次數
        tune : int
            預熱期長度
        chains : int
            MCMC鏈數量
        priors : Optional[np.ndarray]
            先驗概率, shape為(n_samples, n_states), 用於半監督學習
        validation_data : Optional[np.ndarray]
            驗證數據, 用於early stopping, shape需要與訓練數據相同
        early_stopping_patience : Optional[int]
            early stopping的耐心值, None表示不使用early stopping
        min_improvement : float
            最小改進閾值, 用於early stopping
        """
        # 檢查特徵維度
        if self.model is None:
            self.build_model(data)
        elif data.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, but got {data.shape[1]}")

        # 檢查驗證數據的特徵維度
        if validation_data is not None and validation_data.shape[1] != self.n_features:
            raise ValueError(
                f"Validation data expected {self.n_features} features, "
                f"but got {validation_data.shape[1]}"
            )

        # 將數據轉換為pomegranate格式
        X = torch.tensor(data.reshape(1, data.shape[0], data.shape[1]), device=self.device)
        if validation_data is not None:
            val_X = torch.tensor(
                validation_data.reshape(1, validation_data.shape[0], validation_data.shape[1]),
                device=self.device,
            )

        # 設置訓練參數
        if self.model is not None:
            self.model.max_iter = n_draws
            self.model.tol = 1e-5

            # 初始化early stopping變量
            best_val_loss = float("inf")
            patience_counter = 0
            best_model_state = None

            # 訓練模型
            for epoch in range(n_draws):
                if priors is not None:
                    priors_tensor = torch.tensor(priors, device=self.device)
                    improvement = self.model.summarize(X, priors=priors_tensor)
                else:
                    improvement = self.model.summarize(X)

                self.model.from_summaries()

                # 記錄訓練歷史
                self.training_history.append({"epoch": epoch, "improvement": improvement})

                # 驗證和early stopping
                if validation_data is not None and early_stopping_patience is not None:
                    val_loss = -self.model.log_probability(val_X).item()

                    if self.verbose:
                        print(f"Epoch {epoch}: val_loss = {val_loss:.4f}")

                    if val_loss < best_val_loss - min_improvement:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # 保存最佳模型狀態
                        best_model_state = {
                            "edges": (
                                self.model.edges.clone()
                                if hasattr(self.model.edges, "clone")
                                else self.model.edges
                            ),
                            "distributions": [
                                {
                                    "means": (
                                        d.means.clone() if hasattr(d.means, "clone") else d.means
                                    ),
                                    "covs": d.covs.clone() if hasattr(d.covs, "clone") else d.covs,
                                }
                                for d in self.model.distributions
                            ],
                        }
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            if self.verbose:
                                print(f"Early stopping at epoch {epoch}")
                            # 恢復最佳模型狀態
                            if best_model_state is not None:
                                self.model.edges = best_model_state["edges"]
                                for d, state in zip(
                                    self.model.distributions,
                                    best_model_state["distributions"],
                                ):
                                    d.means = state["means"]
                                    d.covs = state["covs"]
                            break

                if self.verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}: improvement = {improvement:.4f}")

            # 提取模型參數
            if self.model is not None:
                # 安全地轉換tensor到numpy
                self.transition_matrix = self._tensor_to_numpy(self.model.edges)
                self.means = np.array(
                    [self._tensor_to_numpy(d.means) for d in self.model.distributions]
                )
                self.sds = np.array(
                    [
                        np.sqrt(self._tensor_to_numpy(d.covs).diagonal())
                        for d in self.model.distributions
                    ]
                )

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        安全地將tensor轉換為numpy數組。

        Parameters
        ----------
        tensor : torch.Tensor
            輸入tensor

        Returns
        -------
        np.ndarray
            轉換後的numpy數組
        """
        if tensor is None:
            return np.array([])
        if isinstance(tensor, np.ndarray):
            return tensor
        if hasattr(tensor, "detach"):
            tensor = tensor.detach()
        if tensor.device != torch.device("cpu"):
            tensor = tensor.cpu()
        return tensor.numpy()

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        預測最可能的狀態序列。

        Parameters
        ----------
        data : np.ndarray
            輸入數據, shape為(n_samples, n_features)
            n_features必須與訓練數據的特徵維度相同

        Returns
        -------
        np.ndarray
            預測的狀態序列
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")

        # 檢查特徵維度
        if data.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, but got {data.shape[1]}")

        # 將數據轉換為pomegranate格式
        X = torch.tensor(data.reshape(1, data.shape[0], data.shape[1]), device=self.device)

        # 使用pomegranate的predict方法
        states = self.model.predict(X)
        return self._tensor_to_numpy(states[0])  # 返回第一個batch的結果

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        """
        計算每個時間點的狀態概率。

        Parameters
        ----------
        data : np.ndarray
            輸入數據, shape為(n_samples, n_features)
            n_features必須與訓練數據的特徵維度相同

        Returns
        -------
        np.ndarray
            狀態概率矩陣, shape為(n_samples, n_states)
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")

        # 檢查特徵維度
        if data.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, but got {data.shape[1]}")

        # 將數據轉換為pomegranate格式
        X = torch.tensor(data.reshape(1, data.shape[0], data.shape[1]), device=self.device)

        # 使用pomegranate的predict_proba方法
        probs = self.model.predict_proba(X)
        return self._tensor_to_numpy(probs[0])  # 返回第一個batch的結果

    def log_probability(self, data: np.ndarray) -> float:
        """
        計算數據的對數似然。

        Parameters
        ----------
        data : np.ndarray
            輸入數據, shape為(n_samples, n_features)
            n_features必須與訓練數據的特徵維度相同

        Returns
        -------
        float
            對數似然值
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")

        # 檢查特徵維度
        if data.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, but got {data.shape[1]}")

        # 將數據轉換為pomegranate格式
        X = torch.tensor(data.reshape(1, data.shape[0], data.shape[1]), device=self.device)

        # 計算對數似然
        log_prob = self.model.log_probability(X)
        return float(self._tensor_to_numpy(log_prob))

    def get_state(self, data: np.ndarray) -> int:
        """
        獲取最後一個時間點的狀態。

        Parameters
        ----------
        data : np.ndarray
            輸入數據, shape為(n_samples, n_features)
            n_features必須與訓練數據的特徵維度相同

        Returns
        -------
        int
            預測的狀態
        """
        states = self.predict(data)
        if len(states) == 0:
            raise ValueError("No states predicted")
        return int(states[-1])

    def get_state_probability(self, data: np.ndarray) -> float:
        """
        獲取最後一個時間點當前狀態的概率。

        Parameters
        ----------
        data : np.ndarray
            輸入數據, shape為(n_samples, n_features)
            n_features必須與訓練數據的特徵維度相同

        Returns
        -------
        float
            狀態概率
        """
        state = self.get_state(data)
        probs = self.predict_proba(data)
        if len(probs) == 0:
            raise ValueError("No probabilities predicted")
        return float(probs[-1][state])

    def get_state_mean(self, state: int) -> np.ndarray:
        """
        獲取指定狀態的均值。

        Parameters
        ----------
        state : int
            狀態索引

        Returns
        -------
        np.ndarray
            狀態均值向量, 長度為特徵維度
        """
        if self.means is None:
            raise ValueError("Model not fitted yet")
        if not 0 <= state < self.n_states:
            raise ValueError(f"Invalid state index: {state}")
        return self.means[state].copy()

    def get_state_std(self, state: int) -> np.ndarray:
        """
        獲取指定狀態的標準差。

        Parameters
        ----------
        state : int
            狀態索引

        Returns
        -------
        np.ndarray
            狀態標準差向量, 長度為特徵維度
        """
        if self.sds is None:
            raise ValueError("Model not fitted yet")
        if not 0 <= state < self.n_states:
            raise ValueError(f"Invalid state index: {state}")
        return self.sds[state].copy()

    def get_transition_probability(self, from_state: int, to_state: int) -> float:
        """
        獲取狀態轉移概率。

        Parameters
        ----------
        from_state : int
            起始狀態
        to_state : int
            目標狀態

        Returns
        -------
        float
            轉移概率
        """
        if self.transition_matrix is None:
            raise ValueError("Model not fitted yet")
        if not (0 <= from_state < self.n_states and 0 <= to_state < self.n_states):
            raise ValueError(f"Invalid state indices: {from_state}, {to_state}")
        return float(self.transition_matrix[from_state, to_state])

    def get_prediction_quality(self, data: np.ndarray) -> float:
        """
        計算預測質量指標。

        結合以下因素計算綜合質量分數:
        1. 狀態預測的置信度 (40%)
        2. 模型對數似然 (20%)
        3. 狀態轉移的穩定性 (20%)
        4. 數據與狀態均值的匹配度 (20%)

        Parameters
        ----------
        data : np.ndarray
            輸入數據, shape為(n_samples, n_features)
            n_features必須與訓練數據的特徵維度相同

        Returns
        -------
        float
            預測質量分數 (0-1之間)
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")

        try:
            # 1. 狀態預測的置信度 (40%)
            state_confidence = self.get_state_probability(data)

            # 2. 對數似然的貢獻 (20%)
            log_likelihood = self.log_probability(data)
            normalized_likelihood = 1 / (1 + np.exp(-log_likelihood))

            # 3. 狀態轉移穩定性 (20%)
            states = self.predict(data)
            if len(states) > 1:
                state_changes = np.sum(np.abs(np.diff(states)))
                stability = np.exp(-state_changes / len(states))
            else:
                stability = 1.0

            # 4. 數據與狀態均值的匹配度 (20%)
            current_state = states[-1]
            last_point = data[-1]
            state_mean = self.get_state_mean(current_state)
            state_std = self.get_state_std(current_state)
            z_scores = np.abs((last_point - state_mean) / (state_std + 1e-6))
            match_score = np.exp(-np.mean(z_scores))

            # 綜合評分
            quality_score = (
                0.4 * state_confidence
                + 0.2 * normalized_likelihood
                + 0.2 * stability
                + 0.2 * match_score
            )

            return float(np.clip(quality_score, 0, 1))

        except Exception as e:
            print(f"Error calculating prediction quality: {e}")
            return 0.0

    def get_training_history(self) -> list[dict[str, Any]]:
        """
        獲取訓練歷史。

        Returns
        -------
        list[dict[str, Any]]
            包含每個epoch的訓練信息的列表
        """
        return self.training_history.copy()

    def save_params(self) -> dict[str, Any]:
        """
        序列化模型參數。

        Returns
        -------
        dict[str, Any]
            包含模型參數的字典
        """
        return {
            "n_states": self.n_states,
            "n_features": self.n_features,
            "transition_matrix": (
                self.transition_matrix.tolist() if self.transition_matrix is not None else None
            ),
            "means": self.means.tolist() if self.means is not None else None,
            "sds": self.sds.tolist() if self.sds is not None else None,
        }

    def load_params(self, params: dict[str, Any]) -> None:
        """
        加載模型參數。

        Parameters
        ----------
        params : dict[str, Any]
            包含模型參數的字典
        """
        self.n_states = params["n_states"]
        self.n_features = params["n_features"]

        if params["transition_matrix"] is not None:
            self.transition_matrix = np.array(params["transition_matrix"])
        if params["means"] is not None:
            self.means = np.array(params["means"])
        if params["sds"] is not None:
            self.sds = np.array(params["sds"])

        # 如果有完整參數, 重建模型
        if all(v is not None for v in [self.transition_matrix, self.means, self.sds]):
            try:
                distributions = [
                    Normal(
                        means=torch.tensor(mu, device=self.device),
                        covs=torch.diag(torch.tensor(sd * sd, device=self.device)),
                        covariance_type="diag",
                    )
                    for mu, sd in zip(self.means, self.sds)
                ]
                self.model = DenseHMM(
                    distributions=distributions,
                    edges=torch.tensor(self.transition_matrix, device=self.device),
                    starts=torch.tensor(self.transition_matrix[0], device=self.device),
                    ends=torch.tensor(np.ones(self.n_states) / self.n_states, device=self.device),
                    random_state=self.random_seed,
                    verbose=self.verbose,
                ).to(self.device)
            except Exception as e:
                print(f"Error rebuilding model: {e}")
                self.model = None

    def save_to_file(self, filepath: str) -> None:
        """
        將模型保存到文件。

        Parameters
        ----------
        filepath : str
            保存路徑
        """
        state = {
            "params": self.save_params(),
            "device": self.device,
            "random_seed": self.random_seed,
            "verbose": self.verbose,
            "training_history": self.training_history,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load_from_file(cls, filepath: str) -> "HMMModel":
        """
        從文件加載模型。

        Parameters
        ----------
        filepath : str
            文件路徑

        Returns
        -------
        HMMModel
            加載的模型實例
        """
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        model = cls(
            n_states=state["params"]["n_states"],
            random_seed=state["random_seed"],
            device=state["device"],
            verbose=state["verbose"],
        )
        model.load_params(state["params"])
        model.training_history = state["training_history"]
        return model
