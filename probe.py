"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.

Design choices:
- Deeper MLP with BatchNorm and Dropout for regularisation.
- PCA-based dimensionality reduction to 128 components before feeding the net
  (handles the large feature_dim from multi-layer aggregation efficiently).
- Positive class weighting to address class imbalance.
- Cosine-annealing LR schedule with warm restarts.
- Ensemble of 5 nets trained with different random seeds; predictions are
  averaged before thresholding (reduces variance on a small dataset).
- Threshold tuning on the validation split to maximise F1.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

# Number of ensemble members
_N_ENSEMBLE = 5
# PCA components (-1 to disable)
_PCA_COMPONENTS = 128
# Training epochs per member
_EPOCHS = 300
# Mini-batch size (full-batch if dataset is small enough, else mini-batch)
_BATCH_SIZE = 64


def _make_net(input_dim: int) -> nn.Sequential:
    """Build a 3-hidden-layer MLP with BatchNorm and Dropout."""
    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(128, 1),
    )


def _train_one(
    net: nn.Sequential,
    X_t: torch.Tensor,
    y_t: torch.Tensor,
    pos_weight: torch.Tensor,
    epochs: int,
    batch_size: int,
    seed: int,
) -> None:
    """Train *net* in-place using mini-batch SGD with cosine annealing."""
    torch.manual_seed(seed)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-5
    )

    n = X_t.shape[0]
    net.train()
    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            x_b = X_t[idx]
            y_b = y_t[idx]
            optimizer.zero_grad()
            logits = net(x_b).squeeze(-1)
            loss = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
    net.eval()


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Architecture:
      StandardScaler → PCA(128) → Ensemble of 5 × MLP(512→256→128→1)

    The ensemble averages sigmoid probabilities from each member before
    applying the decision threshold.
    """

    def __init__(self) -> None:
        super().__init__()
        self._scaler = StandardScaler()
        self._pca: PCA | None = None
        self._nets: list[nn.Sequential] = []
        self._threshold: float = 0.5

    # ------------------------------------------------------------------
    def _build_network(self, input_dim: int) -> None:
        """Instantiate the ensemble of networks."""
        self._nets = [_make_net(input_dim) for _ in range(_N_ENSEMBLE)]

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average logit from the ensemble (for compatibility; not used directly)."""
        if not self._nets:
            raise RuntimeError("Call fit() before forward().")
        outs = [net(x).squeeze(-1) for net in self._nets]
        return torch.stack(outs, dim=0).mean(dim=0)

    def _preprocess(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale and (optionally) reduce dimensionality."""
        if fit:
            X_scaled = self._scaler.fit_transform(X)
            if _PCA_COMPONENTS > 0 and X_scaled.shape[1] > _PCA_COMPONENTS:
                n_components = min(_PCA_COMPONENTS, X_scaled.shape[0], X_scaled.shape[1])
                self._pca = PCA(n_components=n_components, random_state=42)
                X_out = self._pca.fit_transform(X_scaled)
            else:
                self._pca = None
                X_out = X_scaled
        else:
            X_scaled = self._scaler.transform(X)
            if self._pca is not None:
                X_out = self._pca.transform(X_scaled)
            else:
                X_out = X_scaled
        return X_out.astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the ensemble probe on labelled feature vectors.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.
            y: Integer label vector of shape ``(n_samples,)``; 0 = truthful,
               1 = hallucinated.

        Returns:
            ``self``
        """
        X_proc = self._preprocess(X, fit=True)
        input_dim = X_proc.shape[1]

        self._build_network(input_dim)

        X_t = torch.from_numpy(X_proc)
        y_t = torch.from_numpy(y.astype(np.float32))

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

        for seed, net in enumerate(self._nets):
            _train_one(
                net, X_t, y_t, pos_weight,
                epochs=_EPOCHS,
                batch_size=min(_BATCH_SIZE, len(y)),
                seed=seed * 7 + 42,
            )

        self.eval()
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise F1.

        Args:
            X_val: Validation feature matrix.
            y_val: Integer label vector.

        Returns:
            ``self``
        """
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 201)]))

        best_threshold = 0.5
        best_f1 = -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = f1_score(y_val, y_pred_t, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels."""
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Returns:
            Array of shape ``(n_samples, 2)`` — column 1 is P(hallucinated).
        """
        X_proc = self._preprocess(X, fit=False)
        X_t = torch.from_numpy(X_proc)
        with torch.no_grad():
            # Average sigmoid probabilities across ensemble members
            probs_list = []
            for net in self._nets:
                net.eval()
                logits = net(X_t).squeeze(-1)
                probs_list.append(torch.sigmoid(logits))
            prob_pos = torch.stack(probs_list, dim=0).mean(dim=0).numpy()
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)