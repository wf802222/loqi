"""Flat RAG baseline — pure vector similarity retrieval.

This is the simplest retrieval system: encode query and documents,
return top-k by cosine similarity. No graph, no triggers, no learning.
Serves as the floor that all Loqi variants must beat.
"""

from __future__ import annotations

import numpy as np

from loqi.benchmarks.schema import Document
from loqi.eval.protocol import RetrievalResult, RetrievalSystem
from loqi.graph.embeddings import EmbeddingModel, cosine_similarity_matrix


class FlatRAG(RetrievalSystem):
    """Vector similarity retrieval with no graph structure."""

    def __init__(self, embedding_model: EmbeddingModel | None = None):
        self._model = embedding_model or EmbeddingModel()
        self._documents: list[Document] = []
        self._doc_embeddings: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "flat-rag"

    def index(self, documents: list[Document]) -> None:
        self._documents = list(documents)
        if not documents:
            self._doc_embeddings = None
            return

        texts = [f"{d.title}: {d.text}" if d.title else d.text for d in documents]
        self._doc_embeddings = self._model.encode(texts)

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        if not self._documents or self._doc_embeddings is None:
            return RetrievalResult()

        query_embedding = self._model.encode_single(query)
        similarities = cosine_similarity_matrix(query_embedding, self._doc_embeddings)

        # Get top-k indices sorted by similarity descending
        top_indices = np.argsort(similarities)[::-1][:top_k]

        retrieved_docs = [self._documents[i] for i in top_indices]
        retrieved_ids = [d.id for d in retrieved_docs]

        return RetrievalResult(
            retrieved_ids=retrieved_ids,
            retrieved_docs=retrieved_docs,
            metadata={"similarities": {
                self._documents[i].id: float(similarities[i])
                for i in top_indices
            }},
        )

    def reset(self) -> None:
        self._documents = []
        self._doc_embeddings = None
