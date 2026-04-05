"""Consolidation engine — offline memory processing during downtime.

This is the "dreaming" phase of Loqi's architecture. It runs between
work sessions, reviewing recent experience and restructuring the
memory graph for better future retrieval.

The brain analogy: during sleep, the hippocampus replays recent
experiences while the neocortex reorganizes long-term storage.
Consolidation does the same: replay recent episodes, strengthen
validated patterns, decay noise, discover latent bridges, and
mine trigger candidates.

Consolidation orchestrates existing components:
  - HebbianUpdater (replay strengthening)
  - EdgePromoter (batch promotion check)
  - DecayManager (batch decay/pruning)
Plus adds new capabilities:
  - Bridge discovery (A->B strong, B->C strong => propose A->C)
  - Trigger candidate mining (repeated useful patterns => propose trigger)

This is NOT query-time logic. It runs in the background.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from loqi.graph.embeddings import EmbeddingModel
from loqi.graph.models import Edge, EdgeType, NodeType, Trigger, TriggerOrigin
from loqi.graph.store import GraphStore
from loqi.hebbian.decay import DecayManager
from loqi.hebbian.episode import EpisodeLog
from loqi.hebbian.promoter import EdgePromoter
from loqi.pipeline.config import PipelineConfig


@dataclass
class ConsolidationReport:
    """What happened during a consolidation cycle."""

    episodes_replayed: int = 0
    edges_strengthened: int = 0
    promotions: list[tuple[str, str, str]] = field(default_factory=list)
    bridges_created: int = 0
    trigger_candidates: int = 0
    decay_summary: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "Consolidation Report",
            f"  Episodes replayed: {self.episodes_replayed}",
            f"  Edges strengthened: {self.edges_strengthened}",
            f"  Promotions: {len(self.promotions)}",
            f"  Bridges discovered: {self.bridges_created}",
            f"  Trigger candidates: {self.trigger_candidates}",
        ]
        if self.decay_summary:
            for k, v in self.decay_summary.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


class Consolidator:
    """Runs offline memory consolidation during downtime.

    Call consolidate() between work sessions or during idle periods.
    It reviews recent episodes and restructures the graph.
    """

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
        self._promoter = EdgePromoter(store, episode_log, config, embedding_model)
        self._decay = DecayManager(store, config)
        self._last_consolidated_count = 0

    def consolidate(self) -> ConsolidationReport:
        """Run a full consolidation cycle.

        Order matters: decay FIRST (weaken stale connections), then
        replay (re-strengthen recently useful ones). This way stale
        edges get pruned but fresh experience is preserved.

        Steps:
        1. Decay cycle (prune weak edges, decay stale triggers)
        2. Replay recent episodes — re-strengthen useful co-activations
        3. Check all edges for promotion
        4. Discover latent bridges (A->B, B->C => propose A->C)
        5. Mine trigger candidates from repeated useful contexts
        """
        report = ConsolidationReport()

        # Step 1: Decay first — weaken stale connections
        report.decay_summary = self._decay.run_decay_cycle()

        # Step 2: Replay recent episodes — freshen useful patterns
        report.episodes_replayed = self._replay_recent_episodes()
        report.edges_strengthened = report.episodes_replayed

        # Step 3: Batch promotion check
        report.promotions = self._promoter.check_all_edges()

        # Step 4: Bridge discovery
        report.bridges_created = self._discover_bridges()

        # Step 5: Trigger candidate mining
        report.trigger_candidates = self._mine_trigger_candidates()

        # Mark how far we've consolidated
        self._last_consolidated_count = len(self._log)

        return report

    def _replay_recent_episodes(self) -> int:
        """Re-strengthen edges from episodes since last consolidation.

        This is the "replay" phase — the system revisits recent
        experience and reinforces patterns that were useful. Like
        the hippocampus replaying the day's events during sleep.
        """
        episodes = self._log.episodes
        new_episodes = episodes[self._last_consolidated_count:]

        strengthened = 0
        for ep in new_episodes:
            useful = list(ep.useful_ids)
            for i, id_a in enumerate(useful):
                for id_b in useful[i + 1:]:
                    # Check both edge directions (edges are directional)
                    for src, dst in [(id_a, id_b), (id_b, id_a)]:
                        edge = self._store.get_edge(src, dst)
                        if edge is not None:
                            self._store.strengthen_edge(
                                src, dst,
                                self._config.hebbian_strengthen_rate * 0.5,
                            )
                            strengthened += 1

        return strengthened

    def _discover_bridges(self) -> int:
        """Find latent connections: if A->B is strong and B->C is strong,
        propose A->C as a DIFFUSE edge.

        This is the "insight" step — discovering connections that were
        never directly observed but are implied by the graph structure.
        Like waking up with an idea that connects two things you were
        thinking about separately.
        """
        created = 0
        min_weight = 0.4  # Only bridge between reasonably strong edges
        nodes = self._store.get_all_nodes()

        # Only consider section nodes for bridging
        section_nodes = [n for n in nodes if n.node_type == NodeType.SECTION]

        for node_b in section_nodes:
            # Find strong incoming connections to B
            # (We check outgoing from all nodes to B)
            neighbors_of_b = self._store.get_neighbors(
                node_b.id, min_weight=min_weight
            )

            if len(neighbors_of_b) < 2:
                continue

            # For each pair of B's strong neighbors (A and C),
            # check if A->C already exists. If not, propose it.
            neighbor_ids = [nid for nid, _ in neighbors_of_b]

            for i, id_a in enumerate(neighbor_ids):
                for id_c in neighbor_ids[i + 1:]:
                    # Skip if edge already exists
                    if self._store.get_edge(id_a, id_c) is not None:
                        continue

                    # Get the weights of A->B and B->C edges
                    edge_ab = self._store.get_edge(node_b.id, id_a)
                    edge_bc = self._store.get_edge(node_b.id, id_c)

                    if edge_ab is None or edge_bc is None:
                        continue

                    # Bridge weight = geometric mean of the two edges
                    bridge_weight = (edge_ab.weight * edge_bc.weight) ** 0.5
                    if bridge_weight < 0.15:
                        continue

                    # Create the bridge as a DIFFUSE edge
                    self._store.add_edge(Edge(
                        source_id=id_a,
                        target_id=id_c,
                        weight=bridge_weight * 0.5,  # Speculative, start weak
                        edge_type=EdgeType.DIFFUSE,
                    ))
                    self._store.add_edge(Edge(
                        source_id=id_c,
                        target_id=id_a,
                        weight=bridge_weight * 0.5,
                        edge_type=EdgeType.DIFFUSE,
                    ))
                    created += 1

        return created

    def _mine_trigger_candidates(self) -> int:
        """Look for recurring patterns in useful retrievals that could
        become new triggers.

        If the same section keeps appearing as useful in similar contexts,
        and no trigger exists for that pattern yet, propose one.
        """
        if not self._model:
            return 0

        episodes = self._log.episodes
        if len(episodes) < 3:
            return 0

        # Count how often each section appears as useful
        useful_counts: Counter[str] = Counter()
        for ep in episodes:
            for uid in ep.useful_ids:
                useful_counts[uid] += 1

        # Sections that are frequently useful are trigger candidates
        threshold = max(3, len(episodes) // 5)
        candidates = [
            node_id for node_id, count in useful_counts.items()
            if count >= threshold
        ]

        # Check if a trigger already exists for each candidate
        existing_triggers = self._store.get_all_triggers()
        triggered_nodes = {t.associated_node_id for t in existing_triggers}

        created = 0
        for node_id in candidates:
            if node_id in triggered_nodes:
                continue

            # Build a trigger pattern from the contexts where this node was useful
            context_embeddings = []
            context_words: Counter[str] = Counter()

            for ep in episodes:
                if node_id in ep.useful_ids:
                    if ep.context_embedding is not None:
                        context_embeddings.append(ep.context_embedding)
                    # Extract keywords from context
                    import re
                    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", ep.context.lower())
                    stop = {"the", "a", "an", "is", "are", "was", "to", "for", "of",
                            "in", "on", "and", "or", "with", "by", "from", "this", "that"}
                    for t in tokens:
                        if t not in stop and len(t) >= 3:
                            context_words[t] += 1

            if not context_embeddings:
                continue

            # Average embedding as trigger pattern
            pattern_emb = np.mean(context_embeddings, axis=0).astype(np.float32)

            # Top recurring keywords as pattern words
            keywords = [w for w, _ in context_words.most_common(12)]

            trigger = Trigger(
                id=f"consolidated_{node_id}",
                pattern=keywords,
                pattern_embedding=pattern_emb,
                associated_node_id=node_id,
                confidence=0.5,  # Starts low — must earn confidence through use
                origin=TriggerOrigin.HEBBIAN,
            )
            self._store.add_trigger(trigger)
            created += 1

        return created
