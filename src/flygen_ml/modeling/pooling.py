from __future__ import annotations

import numpy as np


def mean_pool_embeddings(embeddings: np.ndarray, segment_indices: np.ndarray | list[int]) -> np.ndarray:
    selected = embeddings[np.asarray(segment_indices, dtype=int)]
    if selected.size == 0:
        raise ValueError("cannot pool an empty segment set")
    return selected.mean(axis=0)


def attention_pool_embeddings():
    raise NotImplementedError("Attention pooling scaffold only.")
