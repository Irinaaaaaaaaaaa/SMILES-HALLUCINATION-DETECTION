# SMILES-2026 Hallucination Detection — SOLUTION

## Final Results

5-fold stratified cross-validation (seed = 42), averaged across folds:

| Checkpoint | Accuracy | F1 | AUROC |
|---|---:|---:|---:|
| Majority-class baseline | 70.10% | 82.42% | N/A |
| Probe — train split | 97.99% | 98.61% | 100.00% |
| Probe — validation split | 77.88% | 84.68% | 79.30% |
| Probe — test split | 74.60% | 82.89% | 78.82% |

Final feature dimension: **1792**


---

# Reproducibility

## Environment

Tested on:

- Google Colab T4 GPU
- Python 3.11
- PyTorch + HuggingFace Transformers
- scikit-learn

## Commands

```bash
git clone <YOUR_REPOSITORY_LINK>
cd SMILES-HALLUCINATION-DETECTION

pip install -r requirements.txt

python solution.py
````

Running `solution.py` generates:

* `results.json`
* `predictions.csv`


---

# Final Solution

The final solution modifies:

* `aggregation.py`
* `probe.py`
* `splitting.py`

The overall pipeline is:

1. Extract hidden states from Qwen/Qwen2.5-0.5B
2. Aggregate response-token representations from middle transformer layers
3. Train a lightweight logistic regression classifier

---

# Aggregation Strategy

The final aggregation strategy focuses only on the model response tokens rather than the entire prompt.

## Key design choices

### Response-only pooling

Instead of pooling over the full prompt + response sequence, the aggregation only uses hidden states corresponding to the generated response.

This significantly improved generalization because hallucination-related signals are concentrated in the generated answer rather than in the user prompt.

---

### Middle transformer layers

The final solution uses hidden states from transformer layers:

* Layer 12
* Layer 13

Middle layers consistently performed better than final layers during experiments.

---

### Max pooling

For each selected layer:

* hidden states over response tokens are extracted
* element-wise max pooling is applied

The pooled vectors from layers 12 and 13 are concatenated.

Final feature size:

```text
2 × 896 = 1792
```

---

# Probe Classifier

The final classifier is:

```python
StandardScaler +
LogisticRegression(C=0.01)
```

Configuration:

* strong L2 regularization (`C=0.01`)
* `max_iter=2000`
* `random_state=42`

No PCA or neural probe was used in the final submission.

---

# Why This Worked

The largest improvements came from:

1. Using only response tokens
2. Switching from mean pooling to max pooling
3. Using middle transformer layers instead of the final layers
4. Stronger regularization in logistic regression

The final approach improved both validation and test AUROC substantially compared to the initial baseline implementations.

---

# Experiments and Failed Attempts

## 1. Last 8 layers + mean pooling

Initial implementation:

* last 8 transformer layers
* mean pooling over all tokens
* final-token representation
* optional max pooling

Result:

* severe overfitting
* test AUROC around 64–67%

---

## 2. Geometric handcrafted features

Tried features such as:

* layer norm statistics
* cosine drift between layers
* activation spread

These features did not improve validation or test metrics and were removed from the final solution.

---

## 3. PCA dimensionality reduction

Tested PCA with:

* 32 components
* 64 components
* 128 components
* 256 components

Results:

* low-dimensional PCA strongly reduced performance
* PCA(256) improved results slightly
* however, the response-only aggregation strategy without PCA performed best overall

---

## 4. Different Logistic Regression regularization strengths

Tested:

* `C=0.3`
* `C=0.1`
* `C=0.03`
* `C=0.01`

Best results were achieved with:

```python
C = 0.01
```

Smaller values improved generalization and reduced overfitting.

---

## 5. Alternative solvers

Tested:

```python
solver="liblinear"
```

This reduced overall performance and increased overfitting.

---

## 6. PCA + Logistic Regression pipeline

A pipeline with:

```python
StandardScaler ->
PCA ->
LogisticRegression
```

was tested extensively.

Although it stabilized training, the final response-only feature aggregation outperformed all PCA-based variants.

---

# Final Notes

The final solution remains lightweight, reproducible, and fully compatible with the provided infrastructure.

No fixed infrastructure files were modified.

The final submission uses:

* response-token hidden states
* layers 12 and 13
* max pooling
* logistic regression with strong regularization



# Artifacts

Public link to generated evaluation artifacts: https://drive.google.com/drive/u/1/folders/1Itpne50D1SmXzDSvzEFjt8m35s-YlN4-