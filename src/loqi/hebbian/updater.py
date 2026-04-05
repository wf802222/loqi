"""Hebbian updater — usefulness-gated online learning.

Called after each retrieval when the system receives feedback about
which retrieved memories were actually useful. This is the "reward
signal" in RL terms.

Key principle: only strengthen connections that LED TO USEFUL RESULTS.
Co-occurrence alone is not enough. A memory that was retrieved but
didn't help should not get reinforced — and may get slightly weakened.

This is what separates Loqi from naive "fire together = wire together"
Hebbian learning. The gate is usefulness, not frequency.
"""

from __future__ import annotations

from loqi.graph.models import Edge, EdgeType
from loqi.graph.store import GraphStore
from loqi.hebbian.episode import Episode, EpisodeLog
from loqi.pipeline.config import PipelineConfig


class HebbianUpdater:
    """Processes feedback after each retrieval to update the graph."""

    def __init__(self, store: GraphStore, episode_log: EpisodeLog, config: PipelineConfig):
        self._store = store
        self._log = episode_log
        self._config = config

    def update(self, episode: Episode) -> None:
        """Process a single episode's feedback.

        This is the core learning step. Called after synthesis marks
        which retrieved documents were useful.
        """
        # Record the episode for future reference
        self._log.record(episode)

        if not self._config.enable_hebbian:
            return

        self._strengthen_useful_pairs(episode)
        self._handle_useless_retrievals(episode)
        self._update_trigger_feedback(episode)

    def _strengthen_useful_pairs(self, episode: Episode) -> None:
        """Strengthen edges between nodes that were BOTH useful.

        If two useful nodes have an existing edge, strengthen it.
        If they don't have an edge, create a new DIFFUSE edge —
        the system just discovered these two memories are
        useful together, even though they weren't previously connected.
        """
        useful = list(episode.useful_ids)

        for i, id_a in enumerate(useful):
            # Skip nodes that don't exist in the store
            if self._store.get_node(id_a) is None:
                continue
            self._store.update_node_access(id_a)

            for id_b in useful[i + 1:]:
                if self._store.get_node(id_b) is None:
                    continue
                # Check if edge exists in either direction
                edge_ab = self._store.get_edge(id_a, id_b)
                edge_ba = self._store.get_edge(id_b, id_a)

                if edge_ab is not None:
                    self._store.strengthen_edge(
                        id_a, id_b, self._config.hebbian_strengthen_rate
                    )
                else:
                    # Create new speculative edge from co-activation
                    self._store.add_edge(Edge(
                        source_id=id_a,
                        target_id=id_b,
                        weight=0.2,
                        edge_type=EdgeType.DIFFUSE,
                    ))

                if edge_ba is not None:
                    self._store.strengthen_edge(
                        id_b, id_a, self._config.hebbian_strengthen_rate
                    )
                else:
                    self._store.add_edge(Edge(
                        source_id=id_b,
                        target_id=id_a,
                        weight=0.2,
                        edge_type=EdgeType.DIFFUSE,
                    ))

    def _handle_useless_retrievals(self, episode: Episode) -> None:
        """Slightly weaken edges between useful and non-useful nodes.

        A memory that was retrieved alongside useful memories but was
        NOT itself useful is a mild signal that the connection is noisy.
        We apply a small decay — not punitive, just a gentle correction.

        This is the "synaptic depression" analogue: pathways that
        fire but don't contribute get slightly weakened over time.
        """
        useful = episode.useful_ids
        useless = episode.useless_ids

        half_decay = self._config.hebbian_decay_rate * 0.5

        for u_id in useful:
            for n_id in useless:
                edge = self._store.get_edge(u_id, n_id)
                if edge is not None:
                    self._store.decay_edge(u_id, n_id, half_decay)

                edge_rev = self._store.get_edge(n_id, u_id)
                if edge_rev is not None:
                    self._store.decay_edge(n_id, u_id, half_decay)

    def _update_trigger_feedback(self, episode: Episode) -> None:
        """Update trigger statistics based on usefulness.

        Triggers that fired and their memory was useful → reward.
        Triggers that fired but their memory was NOT useful → slight decay.
        This makes triggers self-correcting over time.
        """
        for trigger in self._store.get_all_triggers():
            doc_id = trigger.associated_node_id

            if doc_id in episode.triggered_ids:
                was_useful = doc_id in episode.useful_ids
                self._store.update_trigger_fire(trigger.id, was_useful)

                if not was_useful:
                    self._store.decay_trigger(
                        trigger.id, self._config.hebbian_decay_rate
                    )
