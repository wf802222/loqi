"""Section-level retrieval system -- architecture v2.

Operates on section-level memory objects, not documents. This is the
retrieval layer that completes the architecture v2 pipeline:

  write-time processing (MemoryWriter)
    -> consolidation (Consolidator)
    -> section-level retrieval (this module)
    -> query-time orchestration

Three retrieval channels, reported separately:
  1. Semantic: cosine similarity between query and section embeddings
  2. Triggers: pre-retrieval pattern matching on section-level triggers
  3. Graph: edge traversal from entry sections to connected sections

Each channel's contribution is tracked in the result metadata
so the benchmark can attribute recall to the right mechanism.
"""

from __future__ import annotations

import numpy as np

from loqi.benchmarks.schema import Document
from loqi.graph.embeddings import EmbeddingModel, cosine_similarity_matrix
from loqi.graph.models import Node, NodeType, TriggerOrigin
from loqi.graph.store import GraphStore
from loqi.graph.writer import MemoryWriter
from loqi.hebbian.consolidator import Consolidator
from loqi.hebbian.decay import DecayManager
from loqi.hebbian.episode import Episode, EpisodeLog
from loqi.hebbian.promoter import EdgePromoter
from loqi.hebbian.updater import HebbianUpdater
from loqi.eval.protocol import RetrievalResult, RetrievalSystem
from loqi.pipeline.config import PipelineConfig
from loqi.triggers.extractor import extract_triggers
from loqi.triggers.matcher import match_triggers


class SectionRetrieval(RetrievalSystem):
    """Section-level retrieval with triggers, graph traversal, and Hebbian learning.

    This is the full architecture v2 system:
    - index() uses MemoryWriter to create section nodes
    - retrieve() queries sections via three channels (semantic + triggers + graph)
    - update() runs Hebbian learning on section-level co-activations
    - consolidate() runs offline dreaming
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        embedding_model: EmbeddingModel | None = None,
    ):
        from loqi.pipeline.config import LOQI_FULL
        self._config = config or LOQI_FULL
        self._model = embedding_model or EmbeddingModel()
        self._store = GraphStore(":memory:")
        self._writer = MemoryWriter(self._store, self._model)
        self._episode_log = EpisodeLog()
        self._updater = HebbianUpdater(self._store, self._episode_log, self._config)
        self._promoter = EdgePromoter(
            self._store, self._episode_log, self._config, self._model
        )
        self._decay = DecayManager(self._store, self._config)
        self._consolidator = Consolidator(
            self._store, self._episode_log, self._config, self._model
        )

        # LLM gate (v2.5)
        self._trigger_gate = None
        if self._config.enable_llm_gate:
            from loqi.llm.trigger_gate import TriggerGate
            self._trigger_gate = TriggerGate()

        # Section-level state for retrieval
        self._section_nodes: list[Node] = []
        self._section_embeddings: np.ndarray | None = None
        self._explicit_triggers = []

    @property
    def name(self) -> str:
        return "loqi-v2"

    def index(self, documents: list[Document]) -> None:
        """Ingest documents as section-level memory objects."""
        for doc in documents:
            # Pass raw text to MemoryWriter — it handles splitting
            # Don't prepend title (the markdown already has # Title)
            sections = self._writer.ingest_document(doc.id, doc.title, doc.text)

            # Extract triggers at section level
            if self._config.enable_triggers:
                for section in sections:
                    section_content = f"{section.title}\n{section.content}"
                    section_triggers = extract_triggers(
                        section.id, section_content, self._model
                    )
                    self._explicit_triggers.extend(section_triggers)

        # Rebuild the section index for retrieval
        self._rebuild_section_index()

    def _rebuild_section_index(self) -> None:
        """Rebuild the in-memory section index from the graph store."""
        all_nodes = self._store.get_all_nodes()
        self._section_nodes = [
            n for n in all_nodes
            if n.node_type == NodeType.SECTION and n.embedding is not None
        ]
        if self._section_nodes:
            self._section_embeddings = np.array(
                [n.embedding for n in self._section_nodes], dtype=np.float32
            )
        else:
            self._section_embeddings = None

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        if not self._section_nodes or self._section_embeddings is None:
            return RetrievalResult()

        query_embedding = self._model.encode_single(query)

        # === Channel 1: Semantic similarity ===
        similarities = cosine_similarity_matrix(
            query_embedding, self._section_embeddings
        )
        semantic_k = min(top_k, len(self._section_nodes))
        semantic_indices = np.argsort(similarities)[::-1][:semantic_k]
        semantic_ids = [self._section_nodes[i].id for i in semantic_indices]
        semantic_scores = {
            self._section_nodes[i].id: float(similarities[i])
            for i in semantic_indices
        }

        # === Channel 2: Trigger firing with discipline ===
        triggered_section_ids: set[str] = set()
        llm_suppressed_ids: list[str] = []
        llm_latency: float = 0.0
        if self._config.enable_triggers:
            all_triggers = list(self._explicit_triggers)

            # Include Hebbian triggers (lower confidence by default)
            for t in self._store.get_all_triggers():
                if t.origin == TriggerOrigin.HEBBIAN:
                    all_triggers.append(t)

            if all_triggers:
                fired = match_triggers(
                    all_triggers, query, query_embedding,
                    threshold=self._config.trigger_confidence_threshold,
                )

                max_inject = self._config.trigger_max_injections
                total_sections = len(self._section_nodes)

                # Guard 1: deduplicate by section ID, keep highest score
                unique_fired: dict[str, float] = {}
                for trigger, score in fired:
                    nid = trigger.associated_node_id
                    if nid not in unique_fired or score > unique_fired[nid]:
                        unique_fired[nid] = score

                # Guard 2: suppress if >30% of sections trigger (generic)
                if len(unique_fired) > total_sections * 0.3:
                    unique_fired = {}

                # Guard 3: score-gap filter (tail is noise)
                if unique_fired:
                    top_score = max(unique_fired.values())
                    unique_fired = {
                        nid: s for nid, s in unique_fired.items()
                        if s >= top_score * 0.5
                    }

                # Guard 4: ARBITRATION — require support from at least
                # one other channel (semantic OR graph). A trigger alone
                # is not enough — it needs corroboration.
                # This prevents triggers from injecting sections that
                # have no other signal of relevance, while preserving
                # the cases where triggers find things semantic misses
                # (as long as graph edges provide the corroboration).
                min_semantic_for_trigger = 0.03
                arbitrated: dict[str, float] = {}
                for nid, trig_score in unique_fired.items():
                    sem_score = semantic_scores.get(nid, 0.0)
                    has_semantic = sem_score >= min_semantic_for_trigger
                    # Check graph support: is this section a neighbor
                    # of any top semantic entry node?
                    has_graph = False
                    for entry_id in semantic_ids[:5]:
                        if self._store.get_edge(entry_id, nid) is not None:
                            has_graph = True
                            break
                    if has_semantic or has_graph:
                        arbitrated[nid] = trig_score

                # Guard 5: LLM gate (v2.5) — ask SmolLM if each
                # surviving candidate is actually relevant.
                llm_suppressed_ids = []
                llm_latency = 0.0
                if self._trigger_gate and arbitrated:
                    section_info = {}
                    for nid in arbitrated:
                        node = self._store.get_node(nid)
                        if node:
                            parent = node.parent_id or ""
                            section_info[nid] = (node.title, node.content, parent)
                    arbitrated, llm_suppressed_ids, llm_latency = (
                        self._trigger_gate.filter_triggers(
                            query, arbitrated, section_info
                        )
                    )

                # Take top N by score
                sorted_fired = sorted(
                    arbitrated.items(), key=lambda x: x[1], reverse=True
                )
                for nid, score in sorted_fired[:max_inject]:
                    triggered_section_ids.add(nid)

        # === Channel 3: Graph neighborhood (2-hop BFS) ===
        graph_discovered_ids: set[str] = set()
        graph_scores: dict[str, float] = {}
        if self._config.enable_graph:
            entry_ids = set(semantic_ids[:5]) | triggered_section_ids
            visited = set(entry_ids)

            # Hop 1: direct neighbors of entry nodes
            hop1_frontier = []
            for entry_id in entry_ids:
                neighbors = self._store.get_neighbors(entry_id, min_weight=0.10)
                for neighbor_id, edge in neighbors:
                    if neighbor_id in visited:
                        continue
                    neighbor_node = self._store.get_node(neighbor_id)
                    if neighbor_node and neighbor_node.node_type == NodeType.SECTION:
                        visited.add(neighbor_id)
                        graph_discovered_ids.add(neighbor_id)
                        graph_scores[neighbor_id] = edge.weight
                        hop1_frontier.append(neighbor_id)

            # Hop 2: neighbors of hop-1 discoveries
            for h1_id in hop1_frontier:
                neighbors = self._store.get_neighbors(h1_id, min_weight=0.15)
                for neighbor_id, edge in neighbors:
                    if neighbor_id in visited:
                        continue
                    neighbor_node = self._store.get_node(neighbor_id)
                    if neighbor_node and neighbor_node.node_type == NodeType.SECTION:
                        visited.add(neighbor_id)
                        graph_discovered_ids.add(neighbor_id)
                        # 2-hop score is the product of edge weights
                        h1_weight = graph_scores.get(h1_id, 0.2)
                        graph_scores[neighbor_id] = h1_weight * edge.weight

        # === Merge and rank ===
        scores: dict[str, float] = {}
        best_semantic = max(semantic_scores.values()) if semantic_scores else 0.3

        # Semantic scores — the foundation
        for sid, score in semantic_scores.items():
            scores[sid] = score

        # Triggered sections: additive boost, not flat replacement.
        # A trigger should help a section rank higher, but should NOT
        # override a clearly dominant semantic winner.
        # Rule: trigger bonus = 0.15 (enough to promote but not dominate)
        # Cap: triggered score cannot exceed best semantic score + 0.05
        trigger_bonus = 0.15
        for sid in triggered_section_ids:
            base = scores.get(sid, 0.0)
            boosted = base + trigger_bonus
            # Cap: don't exceed best semantic + small margin
            boosted = min(boosted, best_semantic + 0.05)
            scores[sid] = max(scores.get(sid, 0), boosted)

        # Graph-discovered sections: additive boost proportional to edge weight,
        # but never exceeding the best semantic score. Graph should HELP ranking,
        # not overwhelm semantic relevance.
        best_semantic = max(semantic_scores.values()) if semantic_scores else 0.3
        for sid in graph_discovered_ids:
            g_score = graph_scores.get(sid, 0.1)
            idx = next(
                (i for i, n in enumerate(self._section_nodes) if n.id == sid),
                None,
            )
            base_sim = float(similarities[idx]) if idx is not None else 0.0
            boosted = base_sim + g_score * 0.3
            # Cap: graph boost should not exceed best semantic match
            boosted = min(boosted, best_semantic)
            scores[sid] = max(scores.get(sid, 0), boosted)

        # Sort by score, take top_k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        retrieved_ids = [sid for sid, _ in ranked]
        retrieved_docs = []
        for sid in retrieved_ids:
            node = self._store.get_node(sid)
            if node:
                retrieved_docs.append(Document(
                    id=node.id,
                    title=node.title,
                    text=node.content,
                ))

        return RetrievalResult(
            retrieved_ids=retrieved_ids,
            retrieved_docs=retrieved_docs,
            triggered_memories=triggered_section_ids,
            metadata={
                "semantic_section_ids": semantic_ids[:top_k],
                "triggered_section_ids": sorted(triggered_section_ids),
                "graph_discovered_section_ids": sorted(graph_discovered_ids),
                "triggers_explicit": len(self._explicit_triggers),
                "triggers_hebbian": sum(
                    1 for t in self._store.get_all_triggers()
                    if t.origin == TriggerOrigin.HEBBIAN
                ),
                "llm_gate_suppressed": llm_suppressed_ids if self._trigger_gate else [],
                "llm_gate_latency_ms": llm_latency if self._trigger_gate else 0,
            },
        )

    def reset(self) -> None:
        self._store.clear()
        self._section_nodes = []
        self._section_embeddings = None
        self._explicit_triggers = []
        self._episode_log.clear()

    def update(self, query: str, result: RetrievalResult,
               useful_ids: set[str]) -> None:
        """Hebbian learning on section-level co-activations."""
        context_embedding = self._model.encode_single(query)

        episode = Episode(
            context=query,
            context_embedding=context_embedding,
            retrieved_ids=result.retrieved_ids,
            triggered_ids=result.triggered_memories,
            useful_ids=useful_ids,
        )

        self._updater.update(episode)

        if self._config.enable_hebbian:
            useful_list = list(useful_ids)
            for i, id_a in enumerate(useful_list):
                for id_b in useful_list[i + 1:]:
                    self._promoter.check_and_promote(id_a, id_b)
                    self._promoter.check_and_promote(id_b, id_a)

        self._decay.tick()

    def consolidate(self):
        """Run offline consolidation (dreaming)."""
        report = self._consolidator.consolidate()
        self._rebuild_section_index()
        return report
