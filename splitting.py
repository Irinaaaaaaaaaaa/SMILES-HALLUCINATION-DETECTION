"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.

Strategy
--------
Stratified 5-fold cross-validation.  Within each fold the remaining 80 % of
the data (the four non-test folds) are further split 85/15 into train and
validation subsets using stratified random sampling.  This gives:
  - ~68 % train, ~12 % val, ~20 % test per fold.
All proportions are maintained class-balanced via stratification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# Number of cross-validation folds.
_N_FOLDS = 5
# Fraction of the train+val pool reserved for validation within each fold.
_VAL_FRACTION = 0.15
_RANDOM_STATE = 42


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,       # kept for API compatibility (unused in k-fold)
    val_size: float = 0.15,
    random_state: int = _RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Split dataset indices into stratified k-fold train/val/test subsets.

    Uses ``StratifiedKFold`` so that each fold's test set contains roughly
    ``1 / _N_FOLDS`` of the data with the same class ratio as the full set.
    Within each fold the non-test indices are further split into train and
    validation subsets (stratified).

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
        df:           Optional full DataFrame (unused here but kept for API).
        test_size:    Ignored (test fraction determined by ``_N_FOLDS``).
        val_size:     Fraction of non-test data reserved for validation.
        random_state: Random seed for reproducible splits.

    Returns:
        A list of ``(idx_train, idx_val, idx_test)`` tuples, one per fold.
    """
    idx = np.arange(len(y))

    skf = StratifiedKFold(n_splits=_N_FOLDS, shuffle=True, random_state=random_state)

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []

    for fold, (idx_train_val, idx_test) in enumerate(skf.split(idx, y)):
        # Stratified split of the non-test portion into train and val.
        relative_val = val_size / (1.0 - 1.0 / _N_FOLDS)
        # Guard: if relative_val >= 1, skip validation split.
        if relative_val >= 1.0 or len(np.unique(y[idx_train_val])) < 2:
            idx_train, idx_val = idx_train_val, None
        else:
            idx_train, idx_val = train_test_split(
                idx_train_val,
                test_size=relative_val,
                random_state=random_state + fold,
                stratify=y[idx_train_val],
            )

        splits.append((idx_train, idx_val, idx_test))

    return splits
