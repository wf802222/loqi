"""Graph-based retrieval system.

Implements the focused and diffuse retrieval passes from Loqi's architecture.
Uses semantic similarity to build edges at index time, then traverses the
graph to find multi-hop connections that flat vector search misses.

This system can operate in three modes via PipelineConfig:
  - graph-only: focused pass only (no diffuse, no triggers, no Hebbian)
  - loqi-no-diffuse: focused + triggers (no diffuse)
  - full: focused + diffuse + triggers + Hebbian
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

from loqi.benchmarks.schema import Document
from loqi.eval.protocol import RetrievalResult, RetrievalSystem
from loqi.graph.embeddings import EmbeddingModel, cosine_similarity_matrix
from loqi.graph.models import Edge, EdgeType, Node
from loqi.graph.store import GraphStore
from loqi.hebbian.decay import DecayManager
from loqi.hebbian.episode import Episode, EpisodeLog
from loqi.hebbian.promoter import EdgePromoter
from loqi.hebbian.updater import HebbianUpdater
from loqi.pipeline.config import GRAPH_ONLY, PipelineConfig


class GraphRAG(RetrievalSystem):
    """Graph-based retrieval with focused and optional diffuse passes."""

    def __init__(
        self,
        config: PipelineConfig = GRAPH_ONLY,
        embedding_model: EmbeddingModel | None = None,
        edge_threshold: float = 0.25,
    ):
        self._config = config
        self._model = embedding_model or EmbeddingModel()
        self._edge_threshold = edge_threshold
        self._store = GraphStore(":memory:")
        self._documents: list[Document] = []
        self._doc_embeddings: np.ndarray | None = None
        self._rng = random.Random(config.random_seed)

        # Hebbian learning components
        self._episode_log = EpisodeLog()
        self._updater = HebbianUpdater(self._store, self._episode_log, config)
        self._promoter = EdgePromoter(self._store, self._episode_log, config, self._model)
        self._decay = DecayManager(self._store, config)

    @property
    def name(self) -> str:
        return self._config.variant_name

    def index(self, documents: list[Document]) -> None:
        if not documents:
            return

        # Filter out documents we've already indexed (by ID)
        existing_ids = {d.id for d in self._documents}
        new_docs = [d for d in documents if d.id not in existing_ids]

        if not new_docs:
            return

        # Encode new documents
        texts = [f"{d.title}: {d.text}" if d.title else d.text for d in new_docs]
        new_embeddings = self._model.encode(texts)

        # Add nodes to graph store
        for doc, emb in zip(new_docs, new_embeddings):
            node = Node(id=doc.id, title=doc.title, content=doc.text, embedding=emb)
            self._store.add_node(node)

        # Accumulate documents and embeddings
        self._documents.extend(new_docs)
        if self._doc_embeddings is not None:
            self._doc_embeddings = np.concatenate(
                [self._doc_embeddings, new_embeddings], axis=0
            )
        else:
            self._doc_embeddings = new_embeddings

        # Build edges among the new documents and between new and existing
        self._build_edges()

    def _build_edges(self) -> None:
        """Create edges between documents with similarity above threshold."""
        if self._doc_embeddings is None or len(self._documents) < 2:
            return

        n = len(self._documents)
        # Compute full similarity matrix
        norms = self._doc_embeddings / (
            np.linalg.norm(self._doc_embeddings, axis=1, keepdims=True) + 1e-8
        )
        sim_matrix = norms @ norms.T

        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i, j])
                if sim >= self._edge_threshold:
                    # Determine edge type based on similarity strength
                    if sim >= 0.6:
                        edge_type = EdgeType.HARD
                    elif sim >= 0.4:
                        edge_type = EdgeType.SOFT
                    else:
                        edge_type = EdgeType.DIFFUSE

                    # Bidirectional edges
                    self._store.add_edge(Edge(
                        source_id=self._documents[i].id,
                        target_id=self._documents[j].id,
                        weight=sim,
                        edge_type=edge_type,
                    ))
                    self._store.add_edge(Edge(
                        source_id=self._documents[j].id,
                        target_id=self._documents[i].id,
                        weight=sim,
                        edge_type=edge_type,
                    ))

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        if not self._documents or self._doc_embeddings is None:
            return RetrievalResult()

        query_embedding = self._model.encode_single(query)

        # Find entry nodes via semantic similarity
        similarities = cosine_similarity_matrix(query_embedding, self._doc_embeddings)
        entry_count = min(self._config.focused_top_k, len(self._documents))
        entry_indices = np.argsort(similarities)[::-1][:entry_count]
        entry_ids = [self._documents[i].id for i in entry_indices]

        # Focused pass: traverse high-weight edges from entry nodes
        # Returns (doc_id, graph_score) pairs
        focused_results = self._focused_pass(entry_ids)
        focused_ids = [doc_id for doc_id, _ in focused_results]

        # Diffuse pass (optional)
        diffuse_ids = []
        if self._config.enable_diffuse:
            diffuse_ids = self._diffuse_pass(entry_ids, focused_ids)

        # Build entry similarity map for ranking
        entry_sims = {
            self._documents[i].id: float(similarities[i])
            for i in entry_indices
        }

        # Merge and rank
        scored = self._merge_and_rank(
            query_embedding, entry_sims, focused_results, diffuse_ids
        )

        # Take top-k
        top_results = scored[:top_k]
        retrieved_ids = [doc_id for doc_id, _ in top_results]
        retrieved_docs = []
        doc_map = {d.id: d for d in self._documents}
        for doc_id in retrieved_ids:
            if doc_id in doc_map:
                retrieved_docs.append(doc_map[doc_id])

        return RetrievalResult(
            retrieved_ids=retrieved_ids,
            retrieved_docs=retrieved_docs,
            metadata={
                "entry_nodes": entry_ids[:5],
                "focused_count": len(focused_results),
                "diffuse_count": len(diffuse_ids),
            },
        )

    def _focused_pass(self, entry_ids: list[str]) -> list[tuple[str, float]]:
        """Traverse high-weight edges from entry nodes.

        BFS up to max_depth, following all edges above a weight threshold.
        Returns (doc_id, graph_score) pairs where graph_score reflects
        the traversal path quality: product of edge weights along the path,
        scaled by the entry node's query similarity.

        The graph score captures: "how good is the connection from a relevant
        entry node to this discovered node?"
        """
        visited = set(entry_ids)
        # frontier: (node_id, accumulated_path_score)
        frontier = [(nid, 1.0) for nid in entry_ids]
        discovered: list[tuple[str, float]] = []

        max_frontier = 100  # Cap fan-out to prevent O(N) expansion on dense graphs

        for depth in range(self._config.focused_max_depth):
            next_frontier = []
            for node_id, path_score in frontier:
                neighbors = self._store.get_neighbors(
                    node_id, min_weight=0.25
                )
                for neighbor_id, edge in neighbors:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        new_score = path_score * edge.weight
                        discovered.append((neighbor_id, new_score))
                        next_frontier.append((neighbor_id, new_score))
            # Keep only the highest-scoring frontier nodes
            if len(next_frontier) > max_frontier:
                next_frontier.sort(key=lambda x: x[1], reverse=True)
                next_frontier = next_frontier[:max_frontier]
            frontier = next_frontier
            if not frontier:
                break

        return discovered

    def _diffuse_pass(
        self, entry_ids: list[str], focused_ids: list[str]
    ) -> list[str]:
        """Traverse low-weight and diffuse edges for speculative connections.

        Random walk with novelty penalty — penalize recently accessed nodes
        to force exploration of less-visited parts of the graph.
        """
        visited = set(entry_ids) | set(focused_ids)
        discovered = []
        walk_nodes = list(entry_ids)

        for _ in range(self._config.diffuse_top_k * 3):  # More walks than needed, deduplicate
            if not walk_nodes:
                break

            # Pick a random starting point
            current = self._rng.choice(walk_nodes)

            # Take a random walk of 2-3 steps
            walk_length = self._rng.randint(2, 3)
            for _ in range(walk_length):
                neighbors = self._store.get_neighbors(current, min_weight=0.0)
                if not neighbors:
                    break

                # Weight selection by inverse of access count (novelty)
                # and edge weight with temperature
                candidates = []
                weights = []
                for n_id, edge in neighbors:
                    if n_id in visited:
                        continue
                    node = self._store.get_node(n_id)
                    novelty = 1.0 / (1.0 + (node.access_count if node else 0) * self._config.diffuse_novelty_penalty)
                    w = (edge.weight ** (1.0 / self._config.diffuse_temperature)) * novelty
                    candidates.append(n_id)
                    weights.append(w)

                if not candidates:
                    break

                # Weighted random selection
                total = sum(weights)
                if total == 0:
                    break
                probs = [w / total for w in weights]
                current = self._rng.choices(candidates, weights=probs, k=1)[0]

                if current not in visited:
                    visited.add(current)
                    discovered.append(current)

            if len(discovered) >= self._config.diffuse_top_k:
                break

        return discovered[:self._config.diffuse_top_k]

    def _merge_and_rank(
        self,
        query_embedding: np.ndarray,
        entry_sims: dict[str, float],
        focused_results: list[tuple[str, float]],
        diffuse_ids: list[str],
    ) -> list[tuple[str, float]]:
        """Merge results from all passes and rank by combined score.

        Scoring strategy:
        - Entry nodes: raw query similarity (same as flat RAG)
        - Focused nodes: blend of query similarity and graph path score.
          The graph score captures "this node is strongly connected to a
          relevant entry node" — it should be trusted even when the node's
          direct similarity to the query is low (the multi-hop case).
        - Diffuse nodes: small boost for speculative connections.
        """
        doc_idx = {d.id: i for i, d in enumerate(self._documents)}
        scores: dict[str, float] = {}

        # Entry nodes: raw query similarity
        for doc_id, sim in entry_sims.items():
            scores[doc_id] = sim

        # Focused nodes: additive boost gated by minimum relevance.
        # Graph-discovered docs get a bonus on top of their query similarity,
        # but only if they pass a minimum similarity floor. This prevents
        # totally irrelevant graph neighbors from outranking genuinely
        # similar documents that flat RAG would have found.
        min_sim_floor = 0.05  # Must have SOME relevance to the query
        for doc_id, graph_score in focused_results:
            if doc_id in doc_idx:
                idx = doc_idx[doc_id]
                sim = float(cosine_similarity_matrix(
                    query_embedding, self._doc_embeddings[idx:idx+1]
                )[0])
                if sim < min_sim_floor:
                    continue  # Skip truly irrelevant graph neighbors
                # Additive boost: base similarity + scaled graph path quality
                best_entry_sim = max(entry_sims.values()) if entry_sims else 0
                graph_bonus = graph_score * best_entry_sim * 0.3
                scores[doc_id] = max(scores.get(doc_id, 0), sim + graph_bonus)

        # Diffuse nodes: query similarity + small speculative bonus
        for doc_id in diffuse_ids:
            if doc_id in doc_idx:
                idx = doc_idx[doc_id]
                sim = float(cosine_similarity_matrix(
                    query_embedding, self._doc_embeddings[idx:idx+1]
                )[0])
                scores[doc_id] = max(scores.get(doc_id, 0), sim + 0.02)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def reset(self) -> None:
        self._store.clear()
        self._documents = []
        self._doc_embeddings = None
        self._rng = random.Random(self._config.random_seed)
        self._episode_log.clear()

    def update(self, query: str, result: RetrievalResult,
               useful_ids: set[str]) -> None:
        """Hebbian learning pipeline: update → promote → decay.

        Called after synthesis marks which retrieved documents were useful.
        The three steps mirror biological learning:
        1. Strengthen useful connections (LTP — long-term potentiation)
        2. Promote well-established connections (myelination)
        3. Decay unused connections (synaptic pruning)
        """
        # Build episode from this retrieval
        context_embedding = None
        if self._model and query:
            context_embedding = self._model.encode_single(query)

        episode = Episode(
            context=query,
            context_embedding=context_embedding,
            retrieved_ids=result.retrieved_ids,
            triggered_ids=result.triggered_memories,
            useful_ids=useful_ids,
        )

        # Step 1: Update edge weights based on usefulness
        self._updater.update(episode)

        # Step 2: Check if any edges qualify for promotion
        if self._config.enable_hebbian:
            useful_list = list(useful_ids)
            for i, id_a in enumerate(useful_list):
                for id_b in useful_list[i + 1:]:
                    self._promoter.check_and_promote(id_a, id_b)
                    self._promoter.check_and_promote(id_b, id_a)

        # Step 3: Periodic decay (runs every N queries)
        self._decay.tick()
