"""Edge promoter — the pathway from observation to conviction.

Edges earn their way through the hierarchy:
  DIFFUSE → SOFT → HARD → Trigger

Each promotion requires repeated validated usefulness, not just
frequency. An edge that co-activates 100 times but is never useful
will decay and die, not promote.

The ultimate promotion — HARD edge to Trigger — is where the system
creates NEW triggers that nobody wrote. The system literally grows
its own trigger layer from usage patterns. This is the novel
contribution that closes the feedback loop.
"""

from __future__ import annotations

import re

import numpy as np

from loqi.graph.embeddings import EmbeddingModel
from loqi.graph.models import EdgeType, Trigger, TriggerOrigin
from loqi.graph.store import GraphStore
from loqi.hebbian.episode import EpisodeLog
from loqi.pipeline.config import PipelineConfig


class EdgePromoter:
    """Promotes edges through the hierarchy based on usefulness."""

    def __init__(
        self,
        store: GraphStore,
        episode_log: EpisodeLog,
        config: PipelineConfig,
        embedding_model: EmbeddingModel | None = None,
    ):
        self._store = store
        self._log = episode_log
        self._config = config
        self._model = embedding_model

    def check_and_promote(self, source_id: str, target_id: str) -> str | None:
        """Check if an edge qualifies for promotion and promote it.

        Returns the new edge type name if promoted, None if not.
        """
        edge = self._store.get_edge(source_id, target_id)
        if edge is None:
            return None

        count = edge.co_activation_count

        if (
            edge.edge_type == EdgeType.DIFFUSE
            and count >= self._config.hebbian_promotion_threshold_soft
        ):
            self._store.promote_edge(source_id, target_id, EdgeType.SOFT)
            return "soft"

        if (
            edge.edge_type == EdgeType.SOFT
            and count >= self._config.hebbian_promotion_threshold_hard
        ):
            self._store.promote_edge(source_id, target_id, EdgeType.HARD)
            return "hard"

        if (
            edge.edge_type == EdgeType.HARD
            and count >= self._config.hebbian_promotion_threshold_trigger
        ):
            self._promote_to_trigger(source_id, target_id)
            return "trigger"

        return None

    def check_all_edges(self) -> list[tuple[str, str, str]]:
        """Check all edges for promotion eligibility.

        Returns list of (source_id, target_id, new_type) for each promotion.
        """
        promotions = []
        nodes = self._store.get_all_nodes()

        for node in nodes:
            neighbors = self._store.get_neighbors(node.id)
            for neighbor_id, edge in neighbors:
                result = self.check_and_promote(node.id, neighbor_id)
                if result is not None:
                    promotions.append((node.id, neighbor_id, result))

        return promotions

    def _promote_to_trigger(self, source_id: str, target_id: str) -> None:
        """Create a new Hebbian trigger from a promoted HARD edge.

        This is where the system grows its own trigger layer. The new
        trigger's pattern is derived from the contexts where this edge
        was usefully co-activated — the system has learned "when I see
        contexts like THESE, I should surface THIS memory."
        """
        # Build trigger pattern from episode history
        context_embeddings = self._log.context_embeddings_for_edge(
            source_id, target_id
        )

        # Average the context embeddings to get a "typical context" for this edge
        pattern_embedding = None
        if context_embeddings:
            pattern_embedding = np.mean(context_embeddings, axis=0).astype(np.float32)

        # Extract keywords from the contexts where this edge was useful
        keywords = self._extract_keywords_from_episodes(source_id, target_id)

        trigger_id = f"hebbian_{source_id}_{target_id}"

        trigger = Trigger(
            id=trigger_id,
            pattern=keywords,
            pattern_embedding=pattern_embedding,
            associated_node_id=target_id,
            confidence=0.7,  # Starts lower than explicit triggers — must earn full confidence
            origin=TriggerOrigin.HEBBIAN,
        )

        self._store.add_trigger(trigger)

    def _extract_keywords_from_episodes(
        self, source_id: str, target_id: str
    ) -> list[str]:
        """Extract keywords from contexts where an edge was usefully co-activated.

        These keywords become the new trigger's pattern — they represent
        "the kind of work context where these two memories are both relevant."
        """
        episodes = self._log.useful_episodes_with_edge(source_id, target_id)
        if not episodes:
            return []

        # Collect all context words
        word_counts: dict[str, int] = {}
        stop_words = frozenset({
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "in", "on", "at", "to", "for", "of", "with", "by", "from",
            "and", "but", "or", "not", "so", "yet",
            "i", "me", "my", "we", "you", "he", "she", "it", "they",
            "this", "that", "what", "which", "who", "how", "if",
        })

        for ep in episodes:
            tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", ep.context.lower())
            for token in tokens:
                if token not in stop_words and len(token) >= 3:
                    word_counts[token] = word_counts.get(token, 0) + 1

        # Keep the most frequent words across episodes (recurring = signal)
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:15]]
