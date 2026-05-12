````md
# SOLUTION.md — SMILES-2026 Hallucination Detection

## Reproducibility Instructions

### Environment

- Python 3.10+
- CUDA GPU recommended
- Tested on Google Colab T4 GPU

Install dependencies:

```bash
pip install -r requirements.txt
````

Run the full pipeline:

```bash
python solution.py
```

This command:

1. Loads `data/dataset.csv`
    
2. Extracts hidden states from `Qwen/Qwen2.5-0.5B`
    
3. Aggregates hidden-state features
    
4. Runs 5-fold cross-validation
    
5. Trains the probe classifier
    
6. Generates:
    
    - `results.json`
        
    - `predictions.csv`
        

No changes to the fixed infrastructure files (`solution.py`, `model.py`, `evaluate.py`) are required.

---

# Final Solution Description

## Modified Files

The following student-editable files were modified:

- `aggregation.py`
    
- `probe.py`
    
- `splitting.py`
    

---

# aggregation.py

## Multi-layer hidden-state aggregation

The final solution aggregates information from multiple transformer layers instead of using only the final token representation.

### Final aggregation strategy

The final configuration:

1. Selects the last 8 transformer layers
    
2. Mean-pools all real (non-padding) tokens for each selected layer
    
3. Adds:
    
    - final-layer last-token representation
        
    - final-layer max pooled representation
        
4. Concatenates all vectors into a single feature representation
    

Final feature dimension:

```text
10 × 896 = 8960
```

### Motivation

Hallucination-related signals are often distributed across multiple positions and layers. Multi-layer aggregation allows the classifier to access richer semantic and factual representations encoded in deeper transformer layers.

Mean pooling improved stability compared to using only the last token.

Max pooling was added to capture strong activation spikes potentially associated with hallucinated generations.

---

## Geometric Features

The implementation also supports optional geometric/statistical features:

- layer-wise L2 norms
    
- inter-layer cosine similarity
    
- norm ratio between early and late layers
    
- activation spread statistics
    

These features can be enabled with:

```python
USE_GEOMETRIC = True
```

Experiments showed only marginal and inconsistent gains, so geometric features were not used in the final configuration.

---

# probe.py

## Final Probe

The final submitted probe uses:

```text
StandardScaler → PCA(256) → LogisticRegression
```

### Final hyperparameters

```python
PCA(n_components=256, random_state=42)

LogisticRegression(
    C=0.01,
    class_weight="balanced",
    max_iter=2000,
    random_state=42,
)
```

### Threshold tuning

The decision threshold is tuned on the validation split using F1-score maximisation.

The dataset is class-imbalanced:

- hallucinated: 483
    
- truthful: 206
    

Threshold tuning improved validation stability compared to a fixed threshold of 0.5.

---

# splitting.py

## Stratified 5-Fold Cross-Validation

The dataset contains only 689 labelled samples, making single train/test splits unstable.

The final solution uses:

- Stratified 5-fold cross-validation
    
- Additional stratified validation split inside each fold
    

Per fold:

- ~68% train
    
- ~12% validation
    
- ~20% test
    

All splits preserve class balance.

---

# Final Results

Final configuration:

- Last 8 transformer layers
    
- Mean + last-token + max pooling
    
- PCA(256)
    
- Logistic Regression probe
    
- 5-fold stratified CV
    

Final averaged metrics:

|Metric|Value|
|---|--:|
|Test Accuracy|70.10%|
|Test F1|81.40%|
|Test AUROC|67.07%|

---

# Experiments and Failed Attempts

## 1. Deep MLP Ensemble

### Configuration

- 5-model ensemble
    
- 300 epochs
    
- BatchNorm + GELU + Dropout
    
- AdamW optimizer
    
- cosine annealing scheduler
    

### Result

- train AUROC ≈ 100%
    
- test AUROC ≈ 64%
    

### Conclusion

The model heavily overfit the small dataset despite strong regularisation.

Discarded in favour of simpler linear models.

---

## 2. Smaller MLP Variants

### Experiments

- reduced hidden dimensions
    
- fewer epochs
    
- single-network MLP instead of ensemble
    

### Result

Overfitting was reduced slightly but performance remained below the Logistic Regression probe.

Discarded.

---

## 3. Number of Aggregated Layers

### Last 4 layers

Reducing aggregation from the last 8 transformer layers to the last 4 layers:

- feature dimension reduced from 8960 → 5376
    
- test AUROC dropped from ~64–65% → ~62%
    

### Conclusion

Using more upper transformer layers improved representation quality and probe performance.

The final solution kept the last 8 layers.

---

## 4. PCA Dimensionality Experiments

Several PCA dimensions were evaluated:

|PCA Components|Test AUROC|
|---|--:|
|32|~64.3%|
|64|~62.3%|
|128|~65.1%|
|256|~67.1%|

### Conclusion

PCA(256) gave the best balance between dimensionality reduction and information retention.

---

## 5. Logistic Regression Regularisation (`C`)

Several values of `C` were evaluated:

|C|Test AUROC|
|---|--:|
|0.3|~64.9%|
|0.1|~64.8%|
|0.03|~64.9%|
|0.01|~65.1%|

### Conclusion

Smaller `C` values (stronger regularisation) improved generalisation and reduced overfitting.

The final model used:

```python
C=0.01
```

---

## 6. Alternative Logistic Regression Solver

### Experiment

Tried:

```python
solver="liblinear"
```

### Result

- stronger overfitting
    
- lower test accuracy
    
- lower test AUROC
    

### Conclusion

The default solver performed better and was retained.

---

## 7. Geometric Features

### Experiment

Tried:

- layer norm statistics
    
- inter-layer cosine similarities
    

### Result

Minor improvements in some folds but inconsistent overall gains.

### Conclusion

Not included in the final submission.

---

# What Contributed Most

The largest improvements came from:

1. Multi-layer aggregation over the last transformer layers
    
2. PCA dimensionality reduction
    
3. Replacing deep overfitting MLPs with a simpler Logistic Regression probe
    
4. Stronger regularisation (`C=0.01`)
    
5. Stratified 5-fold evaluation for more stable estimates
    

The main lesson from the experiments was that the dataset is small enough that simpler linear models generalise better than large neural probes.

# Artifacts

Public link to generated evaluation artifacts: https://drive.google.com/drive/u/1/folders/1Itpne50D1SmXzDSvzEFjt8m35s-YlN4-