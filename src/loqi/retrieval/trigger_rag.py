"""Trigger-aware retrieval system.

Wraps any base retrieval system (flat RAG, graph RAG) with Loqi's
associative trigger layer. The trigger scan runs BEFORE retrieval
and injects relevant memories at the top of results.

This is the core of Loqi's novelty: memories surface because the
context matches their trigger pattern, not because the query
mentions them. "Build a React dropdown" fires the COC rule even
though the query never mentions coding standards.

CLOSED LOOP: The trigger scan reads BOTH explicit triggers (extracted
at index time) AND Hebbian triggers (created by the promoter during
learning). This is what closes the feedback loop — patterns the system
learns through use become triggers that fire on future queries.
"""

from __future__ import annotations

from loqi.benchmarks.schema import Document
from loqi.eval.protocol import RetrievalResult, RetrievalSystem
from loqi.graph.embeddings import EmbeddingModel
from loqi.graph.models import Trigger, TriggerOrigin
from loqi.pipeline.config import LOQI_FULL, PipelineConfig
from loqi.triggers.extractor import extract_triggers
from loqi.triggers.matcher import match_triggers


class TriggerRAG(RetrievalSystem):
    """Retrieval system with associative trigger pre-injection.

    Wraps a base retrieval system and adds trigger scanning:
    1. On index(): extract explicit triggers from documents
    2. On retrieve(): scan ALL triggers (explicit + Hebbian), inject matches
    3. On update(): delegate to base system (which runs Hebbian learning)
    """

    def __init__(
        self,
        base_system: RetrievalSystem,
        config: PipelineConfig = LOQI_FULL,
        embedding_model: EmbeddingModel | None = None,
    ):
        self._base = base_system
        self._config = config
        self._model = embedding_model or EmbeddingModel()
        self._explicit_triggers: list[Trigger] = []
        self._doc_map: dict[str, Document] = {}

    @property
    def name(self) -> str:
        return f"trigger+{self._base.name}"

    def index(self, documents: list[Document]) -> None:
        # In persistent mode, accumulate documents instead of replacing
        for doc in documents:
            self._doc_map[doc.id] = doc

        # Extract explicit triggers from new documents
        if self._config.enable_triggers:
            for doc in documents:
                content = f"{doc.title}\n\n{doc.text}" if doc.title else doc.text
                doc_triggers = extract_triggers(doc.id, content, self._model)
                self._explicit_triggers.extend(doc_triggers)

        # Index in the base system too
        self._base.index(documents)

    def _get_all_triggers(self) -> list[Trigger]:
        """Collect ALL triggers: explicit (from index) + Hebbian (from learning).

        This is what closes the feedback loop. The Hebbian promoter
        creates new triggers in the graph store during learning. By
        reading them here, those learned triggers can fire on future
        queries — the system genuinely improves through use.
        """
        all_triggers = list(self._explicit_triggers)

        # Pull Hebbian-created triggers from the base system's graph store
        if hasattr(self._base, '_store'):
            store_triggers = self._base._store.get_all_triggers()
            for t in store_triggers:
                if t.origin == TriggerOrigin.HEBBIAN:
                    all_triggers.append(t)

        return all_triggers

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        triggered_memories: set[str] = set()
        injected_docs: list[Document] = []

        # --- TRIGGER SCAN (pre-retrieval) ---
        # Reads both explicit AND Hebbian triggers (closed loop)
        if self._config.enable_triggers:
            all_triggers = self._get_all_triggers()
            if all_triggers:
                query_embedding = self._model.encode_single(query)

                fired = match_triggers(
                    all_triggers,
                    query,
                    query_embedding,
                    threshold=self._config.trigger_confidence_threshold,
                )

                # Collect unique triggered document IDs
                seen_doc_ids: set[str] = set()
                for trigger, score in fired:
                    if len(seen_doc_ids) >= self._config.trigger_max_injections:
                        break
                    doc_id = trigger.associated_node_id
                    if doc_id not in seen_doc_ids and doc_id in self._doc_map:
                        seen_doc_ids.add(doc_id)
                        triggered_memories.add(doc_id)
                        injected_docs.append(self._doc_map[doc_id])

        # --- NORMAL RETRIEVAL ---
        remaining_k = max(1, top_k - len(injected_docs))
        base_result = self._base.retrieve(query, top_k=remaining_k)

        # --- MERGE: triggers first, then base results (deduped) ---
        final_ids = [d.id for d in injected_docs]
        final_docs = list(injected_docs)

        for doc_id, doc in zip(base_result.retrieved_ids, base_result.retrieved_docs):
            if doc_id not in triggered_memories:
                final_ids.append(doc_id)
                final_docs.append(doc)

        # Trim to top_k
        final_ids = final_ids[:top_k]
        final_docs = final_docs[:top_k]

        return RetrievalResult(
            retrieved_ids=final_ids,
            retrieved_docs=final_docs,
            triggered_memories=triggered_memories,
            metadata={
                "triggers_fired": len(triggered_memories),
                "triggers_explicit": len(self._explicit_triggers),
                "triggers_hebbian": sum(
                    1 for t in self._get_all_triggers()
                    if t.origin == TriggerOrigin.HEBBIAN
                ),
                "base_result_count": len(base_result.retrieved_ids),
                **base_result.metadata,
            },
        )

    def reset(self) -> None:
        self._explicit_triggers = []
        self._doc_map = {}
        self._base.reset()

    def update(self, query: str, result: RetrievalResult,
               useful_ids: set[str]) -> None:
        self._base.update(query, result, useful_ids)
