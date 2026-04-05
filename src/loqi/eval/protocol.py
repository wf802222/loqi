"""Evaluation protocol — the interface every retrieval system must implement.

This is the contract between the system-under-test and the evaluation harness.
Any system variant (flat RAG, GraphRAG, Loqi-full, Loqi-no-triggers, etc.)
implements RetrievalSystem and can be evaluated uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from loqi.benchmarks.schema import BenchmarkExample, Document


@dataclass(frozen=True)
class RetrievalResult:
    """What the retrieval system returns for a single query.

    Fields:
        retrieved_ids: Document IDs in ranked order (most relevant first).
        retrieved_docs: The actual Document objects (parallel to retrieved_ids).
        triggered_memories: Memory IDs that fired via associative triggers
            (pre-retrieval). Empty for systems without triggers.
        metadata: Any system-specific debug info (edge weights, traversal paths, etc.).
    """

    retrieved_ids: list[str] = field(default_factory=list)
    retrieved_docs: list[Document] = field(default_factory=list)
    triggered_memories: set[str] = field(default_factory=set)
    metadata: dict = field(default_factory=dict)


class RetrievalSystem(ABC):
    """Interface that all system variants must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this system variant (e.g., 'loqi-full', 'flat-rag')."""

    @abstractmethod
    def index(self, documents: list[Document]) -> None:
        """Index a set of documents for later retrieval.

        Called once per benchmark example before retrieve().
        For systems with persistent state (Hebbian learning), this may be
        called across multiple examples to accumulate knowledge.
        """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        """Retrieve documents relevant to the query.

        Returns documents in ranked order. For systems with triggers,
        triggered_memories should contain the IDs of memories that fired
        on the query context before retrieval.
        """

    def reset(self) -> None:
        """Reset system state between benchmark examples.

        Override for systems with persistent state (Hebbian edges, trigger indices).
        Default: no-op (stateless systems don't need this).
        """

    def update(self, query: str, result: RetrievalResult,
               useful_ids: set[str]) -> None:
        """Hebbian update: strengthen edges for useful retrievals.

        Called after synthesis marks which retrieved documents were useful.
        Override for systems with learning. Default: no-op.

        Args:
            query: The original query.
            result: What was retrieved.
            useful_ids: Document IDs that synthesis marked as useful.
        """
