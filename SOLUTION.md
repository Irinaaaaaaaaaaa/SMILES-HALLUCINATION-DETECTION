# SOLUTION.md — SMILES-2026 Hallucination Detection

## Reproducibility Instructions

### Environment

```
Python 3.10+
CUDA GPU recommended (Google Colab T4 works fine)
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the solution

```bash
python solution.py
```

This will:
1. Load `data/dataset.csv` and `data/test.csv`
2. Extract hidden states from Qwen2.5-0.5B
3. Run 5-fold cross-validation
4. Save `results.json` and `predictions.csv`

No changes to `solution.py`, `model.py`, or `evaluate.py` are required.
`USE_GEOMETRIC` in `solution.py` can be set to `True` to enable geometric features (optional, may improve AUROC slightly).

---

## Final Solution Description

### Modified files

- `aggregation.py` — multi-layer feature extraction + geometric features
- `probe.py` — deep MLP ensemble with PCA preprocessing
- `splitting.py` — stratified 5-fold cross-validation

---

### aggregation.py — Multi-layer mean/max/last-token fusion

**Default approach (USE_GEOMETRIC=False):**

Instead of taking only the last token from the final layer, the aggregation now:

1. **Selects the last 8 transformer layers** (layers 17–24 out of 25 total including the embedding layer). The upper layers encode more abstract, task-relevant representations while the lower layers carry mostly syntactic information.
2. **Mean-pools over real (non-padding) tokens** for each selected layer. This is more robust than last-token pooling because: (a) the hallucination signal can be diffuse across the sequence, and (b) mean pooling reduces positional bias.
3. **Appends the last-real-token** of the final layer — captures the autoregressive "summary" state.
4. **Appends max-pooling** of the final layer — captures the strongest activations.

This produces a feature vector of size `10 × 896 = 8960` (8 mean-pooled layers + last token + max pool).

**Geometric features (USE_GEOMETRIC=True):**

Additional hand-crafted features:
- **Layer-wise L2 norms** of the mean-pooled representation per layer (25 values) — captures how the magnitude of representations evolves through the network.
- **Inter-layer cosine similarities** between consecutive layers (24 values) — measures representation drift. Hallucinated responses may show different inter-layer dynamics.
- **Norm ratio** (last layer / first transformer layer) — captures overall amplification.
- **Std of layer norms** — spread of activation magnitudes.
- **Mean and min cosine similarity** — aggregate drift statistics.

**Why this works:** Hallucination is a failure of factual grounding. The intermediate layers of the transformer encode progressively more abstract information; using multiple layers gives the probe access to both syntactic features (lower layers) and semantic/factual representations (upper layers). Mean-pooling is more stable than last-token for variable-length sequences.

---

### probe.py — Deep MLP Ensemble with PCA

**Architecture:**
```
StandardScaler → PCA(128) → Ensemble[5 × MLP(512→256→128→1)]
```

**PCA dimensionality reduction:**
- The multi-layer feature vector has ~8960 dimensions with only ~689 training samples — a severe high-dimensional small-data regime.
- PCA to 128 components retains >95% of variance typically, dramatically reduces overfitting, and speeds up training.

**MLP per ensemble member:**
```
Linear(128, 512) → BatchNorm1d → GELU → Dropout(0.3)
Linear(512, 256) → BatchNorm1d → GELU → Dropout(0.2)  
Linear(256, 128) → BatchNorm1d → GELU → Dropout(0.1)
Linear(128, 1)
```
- **BatchNorm** stabilises training on high-variance PCA features.
- **GELU** activations perform better than ReLU on transformer-derived features.
- **Dropout** with decreasing rates provides strong regularisation in the early layers where overfitting risk is highest.

**Training:**
- **AdamW** optimizer (weight_decay=1e-4) with cosine annealing warm restarts (T_0=50).
- **Positive class weighting** (neg/pos ratio) to handle the class imbalance.
- **Gradient clipping** (max norm 1.0) for stable training.
- **300 epochs**, mini-batch size 64.

**Ensemble:**
- 5 members trained with different random seeds.
- Predictions are averaged at the probability level (sigmoid outputs averaged before thresholding).
- Reduces variance, especially important with a small dataset (~550 training samples per fold).

**Threshold tuning:**
- After training, `fit_hyperparameters` sweeps 201 threshold candidates on the validation split and picks the one maximising F1.
- This is crucial because class imbalance means the default 0.5 threshold is suboptimal.

---

### splitting.py — Stratified 5-Fold Cross-Validation

**Why k-fold instead of a single split:**
- With only 689 samples, a single 70/15/15 split is highly sensitive to which samples end up in the test set.
- 5-fold CV uses 100% of the data for testing (each sample is in exactly one test fold) and gives a much more reliable estimate of generalisation performance.
- The reported metrics (accuracy, F1, AUROC) are averaged across all 5 folds.

**Structure per fold:**
- **Test**: ~20% (~138 samples) — held out entirely.
- **Val**: ~12% of non-test (~66 samples) — used for threshold tuning only.
- **Train**: ~68% (~485 samples) — used for fitting scaler, PCA, and MLP.

All splits are stratified to preserve the class ratio.

---

## What Contributed Most to Improving the Metric

1. **Multi-layer aggregation** (biggest gain) — using 8 layers × mean-pool instead of just the last token's final layer dramatically increases the information available to the probe. The probe can learn which layer patterns discriminate hallucinations.

2. **PCA + deep MLP** — without PCA, the ~8960-dimensional features cause severe overfitting with 550 training samples. PCA brings this to a manageable 128 dimensions. The deeper network then has capacity to learn non-linear decision boundaries.

3. **Ensemble of 5 models** — measurably reduces variance on such a small dataset. Averaging probabilities before thresholding is strictly better than majority voting.

4. **5-fold cross-validation** — not a direct accuracy improvement, but ensures the reported numbers are reliable and the final model is trained on more data.

---

## Experiments and Failed Attempts

### Attempted but not included in the final solution

**1. All-layer concatenation (layers 0–24)**
- Produces a 25 × 896 = 22400-dimensional feature. Even with PCA, training was slower and AUROC did not improve over the last-8-layers variant.
- Discarded: lower layers add noise rather than signal for hallucination detection.

**2. Attention-weighted pooling**
- Tried weighting token positions by their attention scores from the last layer before pooling.
- The attention matrices are not directly available from the `output_hidden_states` forward pass without also requesting `output_attentions=True`, which doubles memory usage and was not compatible with the BATCH_SIZE=4 setup within the fixed infrastructure.
- Discarded: infrastructure constraint.

**3. Logistic regression and SVM as probe**
- Faster to train and often competitive on small datasets.
- With PCA(128) features, a linear SVM achieved ~72% test AUROC vs ~78% for the MLP ensemble.
- Discarded: MLP ensemble consistently outperformed.

**4. Using only geometric features (no raw hidden states)**
- Layer norms + cosine similarities alone gave ~65% AUROC, well below the full representation.
- Discarded: geometric features are useful as a supplement but not as a replacement.

**5. Larger PCA (256, 512 components)**
- Marginal improvement in training AUROC but worse test AUROC (overfitting).
- Optimal was 128 components.

**6. LSTM/Transformer probe over the sequence of layer representations**
- A small LSTM reading the 25-step sequence of mean-pooled layer representations (each 896-dim).
- Much slower training; did not outperform the flat-concat + MLP approach, likely because the dataset is too small to train a sequential model well.
- Discarded.
