from __future__ import annotations

import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class HallucinationProbe:
    def __init__(self) -> None:
        self.model = make_pipeline(
            StandardScaler(),
            PCA(n_components=256, random_state=42),
            LogisticRegression(
                C=0.01,
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            ),
        )

        self._threshold = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        self.model.fit(X, y)
        return self

    def fit_hyperparameters(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "HallucinationProbe":

        probs = self.predict_proba(X_val)[:, 1]

        best_threshold = 0.5
        best_f1 = -1.0

        for t in np.linspace(0.1, 0.9, 161):
            preds = (probs >= t).astype(int)

            score = accuracy_score(y_val, preds)

            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (
            self.predict_proba(X)[:, 1] >= self._threshold
        ).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)