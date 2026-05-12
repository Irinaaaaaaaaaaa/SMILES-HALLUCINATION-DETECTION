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

import os

import pandas as pd
import torch
from transformers import AutoTokenizer


LAYERS = (12, 13)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _load_prompt_lengths() -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    lengths: list[int] = []

    for filename in ("dataset.csv", "test.csv"):
        path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)

        for prompt in df["prompt"]:
            ids = tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]

            lengths.append(len(ids))

    return lengths


_PROMPT_LENGTHS = _load_prompt_lengths()
_COUNTER = 0


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    global _COUNTER

    device = hidden_states.device

    real_positions = attention_mask.to(device).nonzero(as_tuple=False).squeeze(-1)
    n_real = int(real_positions.numel())

    sample_idx = _COUNTER
    _COUNTER += 1

    prompt_len = (
        _PROMPT_LENGTHS[sample_idx]
        if sample_idx < len(_PROMPT_LENGTHS)
        else n_real
    )

    if prompt_len >= n_real - 1:
        response_positions = real_positions[-1:]
    else:
        response_positions = real_positions[prompt_len:]

    parts = []

    for layer_idx in LAYERS:
        layer = hidden_states[layer_idx]  # (seq_len, hidden_dim)
        response_hidden = layer.index_select(0, response_positions)
        max_pool = response_hidden.float().max(dim=0).values
        parts.append(max_pool)

    return torch.cat(parts, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.zeros(0, device=hidden_states.device)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features