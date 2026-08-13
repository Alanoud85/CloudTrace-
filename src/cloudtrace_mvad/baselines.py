from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


class FrequencyRarityBaseline:
    def __init__(self, columns: list[int]):
        self.columns = columns

    def fit(self, x: np.ndarray):
        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        return np.mean(np.maximum(x[:, self.columns], 0.0), axis=1)


class PCAReconstructionBaseline:
    def __init__(self, variance: float = 0.95):
        self.pca = PCA(n_components=variance, svd_solver="full")

    def fit(self, x: np.ndarray):
        self.pca.fit(x)
        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        z = self.pca.transform(x)
        rec = self.pca.inverse_transform(z)
        return np.mean((x - rec) ** 2, axis=1)


class IsolationForestBaseline:
    def __init__(self, seed: int):
        self.model = IsolationForest(n_estimators=300, contamination="auto", random_state=seed, n_jobs=-1)

    def fit(self, x: np.ndarray):
        self.model.fit(x)
        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(x)


class OneClassSVMBaseline:
    def __init__(self, nu: float = 0.01):
        self.model = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)

    def fit(self, x: np.ndarray):
        self.model.fit(x)
        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(x)
