"""Embedding pipeline for Loqi.

Wraps sentence-transformers to provide a simple interface for encoding
text into vectors. Supports caching to avoid re-encoding the same text.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np


class EmbeddingModel:
    """Wrapper around sentence-transformers for text embedding.

    Lazily loads the model on first use to avoid import-time overhead.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)

    @property
    def dimension(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into embeddings.

        Returns ndarray of shape (len(texts), dimension) with dtype float32.
        """
        self._load()
        return self._model.encode(texts, convert_to_numpy=True).astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding vector.

        Returns ndarray of shape (dimension,) with dtype float32.
        """
        return self.encode([text])[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Compute cosine similarities between a query and all corpus vectors.

    Args:
        query: shape (dimension,)
        corpus: shape (n, dimension)

    Returns:
        shape (n,) array of similarities
    """
    if corpus.shape[0] == 0:
        return np.array([], dtype=np.float32)

    query_norm = query / (np.linalg.norm(query) + 1e-8)
    corpus_norms = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
    return corpus_norms @ query_norm
