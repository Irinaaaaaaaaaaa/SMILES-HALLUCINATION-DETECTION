"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Strategy: multi-layer fusion.
      - Select the last 8 transformer layers (ignoring embedding layer at idx 0).
      - For each selected layer, compute mean-pooling over real tokens.
      - Also take the last-token representation from the final layer.
      - Concatenate all layer representations into one flat vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(k * hidden_dim,)`` where k is the
        number of selected layers (mean pool) plus 1 (last token).
    """
    # Mask for real tokens: shape (seq_len, 1)
    mask = attention_mask.to(hidden_states.device).float().unsqueeze(-1)
    n_real = mask.sum().clamp(min=1.0)

    # Select the last 8 transformer layers (indices 1..n_layers-1, step back 8)
    # hidden_states[0] = token embeddings, hidden_states[1..] = transformer layers
    n_layers = hidden_states.shape[0]
    # Pick last 8 transformer layers (or all if fewer)
    start_layer = max(1, n_layers - 8)
    selected_layers = hidden_states[start_layer:]  # (k, seq_len, hidden_dim)

    parts = []

    # Mean pool over real tokens for each selected layer
    for layer_idx in range(selected_layers.shape[0]):
        layer = selected_layers[layer_idx]           # (seq_len, hidden_dim)
        mean_pool = (layer * mask).sum(dim=0) / n_real  # (hidden_dim,)
        parts.append(mean_pool)

    # Last real token of the final transformer layer
    real_positions = attention_mask.to(hidden_states.device).nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())
    last_token = hidden_states[-1][last_pos]         # (hidden_dim,)
    parts.append(last_token)

    # Max pool over real tokens of the final layer
    final_layer = hidden_states[-1]                  # (seq_len, hidden_dim)
    # Zero out padding positions before max
    masked_final = final_layer * mask + (1 - mask) * (-1e9)
    max_pool = masked_final.max(dim=0).values        # (hidden_dim,)
    parts.append(max_pool)

    feature = torch.cat(parts, dim=0)
    return feature


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Features extracted:
      1. Layer-wise L2 norms of the mean-pooled representation (one per layer).
      2. Inter-layer cosine similarity between consecutive layers (representation drift).
      3. Ratio of last-layer norm to first-layer norm.
      4. Standard deviation of layer norms (spread of activations).

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.
    """
    mask = attention_mask.to(hidden_states.device).float().unsqueeze(-1)
    n_real = mask.sum().clamp(min=1.0)

    n_layers = hidden_states.shape[0]

    # Mean-pool each layer over real tokens
    # Shape: (n_layers, hidden_dim)
    mean_pooled = (hidden_states * mask.unsqueeze(0)).sum(dim=1) / n_real

    # 1. L2 norms per layer
    layer_norms = mean_pooled.norm(dim=-1)  # (n_layers,)

    # 2. Inter-layer cosine similarities (consecutive layers)
    cos_sims = []
    for i in range(1, n_layers):
        sim = F.cosine_similarity(
            mean_pooled[i - 1].unsqueeze(0),
            mean_pooled[i].unsqueeze(0),
        )
        cos_sims.append(sim)
    cos_sims = torch.stack(cos_sims)  # (n_layers - 1,)

    # 3. Norm ratio last / first (skip embedding layer for meaningful ratio)
    norm_ratio = (layer_norms[-1] / layer_norms[1].clamp(min=1e-8)).unsqueeze(0)

    # 4. Std of layer norms
    norm_std = layer_norms.std().unsqueeze(0)

    # 5. Mean and min cosine similarity (how much representations drift overall)
    mean_cos = cos_sims.mean().unsqueeze(0)
    min_cos = cos_sims.min().unsqueeze(0)

    geo = torch.cat([layer_norms, cos_sims, norm_ratio, norm_std, mean_cos, min_cos], dim=0)
    return geo.float()


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
