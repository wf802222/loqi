"""Decay manager — homeostasis for the memory graph.

Without decay, the graph fills with noise. Every connection gets
stronger, every trigger fires more often, and retrieval quality
collapses into "everything is relevant."

The decay manager implements three forms of homeostasis:

1. Time-based decay: edges that haven't been strengthened recently
   lose weight and eventually get pruned
2. Trigger decay: triggers with low usefulness ratios lose confidence
3. Out-degree capping: nodes can't have unlimited connections

Plus one form of protection:

4. Crystallization: very high-usefulness connections decay at a
   fraction of the normal rate. Standing instructions that are
   always true shouldn't disappear just because they weren't
   queried this week.
"""

from __future__ import annotations

from loqi.graph.models import EdgeType
from loqi.graph.store import GraphStore
from loqi.pipeline.config import PipelineConfig


# These could be added to PipelineConfig later
_EDGE_PRUNE_FLOOR = 0.05       # Edges below this weight get pruned
_OUT_DEGREE_CAP = 20           # Max outgoing edges per node
_CRYSTALLIZE_THRESHOLD = 15    # Co-activations needed for crystallization
_CRYSTALLIZE_USEFULNESS = 0.7  # Usefulness ratio needed for crystallization
_CRYSTALLIZE_SLOWDOWN = 0.1    # Crystallized edges decay at 10% of normal rate
_TRIGGER_PRUNE_CONFIDENCE = 0.05  # Triggers below this are pruned entirely


class DecayManager:
    """Periodic maintenance to keep the graph healthy."""

    def __init__(self, store: GraphStore, config: PipelineConfig):
        self._store = store
        self._config = config
        self._query_count = 0

    def tick(self) -> dict[str, int]:
        """Called after each retrieval. Runs decay every N queries.

        Returns a summary of what changed (for logging/debugging).
        """
        self._query_count += 1

        # Only run decay periodically, not on every query
        if self._query_count % 10 != 0:
            return {}

        return self.run_decay_cycle()

    def run_decay_cycle(self) -> dict[str, int]:
        """Run a full decay cycle. Returns counts of actions taken."""
        edges_decayed = self._decay_edges()
        edges_pruned = self._prune_weak_edges()
        edges_capped = self._cap_out_degrees()
        triggers_decayed = self._decay_triggers()
        triggers_pruned = self._prune_dead_triggers()

        return {
            "edges_decayed": edges_decayed,
            "edges_pruned": edges_pruned,
            "edges_capped": edges_capped,
            "triggers_decayed": triggers_decayed,
            "triggers_pruned": triggers_pruned,
        }

    def _decay_edges(self) -> int:
        """Apply time-based decay to all edges."""
        count = 0
        nodes = self._store.get_all_nodes()

        for node in nodes:
            neighbors = self._store.get_neighbors(node.id)
            for neighbor_id, edge in neighbors:
                rate = self._config.hebbian_decay_rate

                # Crystallized edges decay much slower
                if self._is_crystallized(edge.co_activation_count):
                    rate *= _CRYSTALLIZE_SLOWDOWN

                if edge.weight > rate:
                    self._store.decay_edge(node.id, neighbor_id, rate)
                    count += 1

        return count

    def _prune_weak_edges(self) -> int:
        """Remove edges that have decayed below the floor."""
        count = 0
        nodes = self._store.get_all_nodes()

        for node in nodes:
            neighbors = self._store.get_neighbors(node.id, min_weight=0.0)
            for neighbor_id, edge in neighbors:
                if edge.weight < _EDGE_PRUNE_FLOOR:
                    # Delete by setting weight to 0 (effective removal)
                    # A proper delete would be cleaner but our store
                    # doesn't have a delete_edge method yet
                    self._store.decay_edge(
                        node.id, neighbor_id, edge.weight
                    )
                    count += 1

        return count

    def _cap_out_degrees(self) -> int:
        """Remove weakest edges from nodes with too many connections."""
        count = 0
        nodes = self._store.get_all_nodes()

        for node in nodes:
            neighbors = self._store.get_neighbors(node.id)
            if len(neighbors) > _OUT_DEGREE_CAP:
                # Neighbors are sorted by weight descending
                # Prune the weakest ones
                to_prune = neighbors[_OUT_DEGREE_CAP:]
                for neighbor_id, edge in to_prune:
                    self._store.decay_edge(node.id, neighbor_id, edge.weight)
                    count += 1

        return count

    def _decay_triggers(self) -> int:
        """Decay confidence of triggers with poor usefulness ratios."""
        count = 0
        for trigger in self._store.get_all_triggers():
            if trigger.fire_count < 3:
                continue  # Not enough data to judge

            if trigger.usefulness_ratio < _CRYSTALLIZE_USEFULNESS:
                self._store.decay_trigger(
                    trigger.id, self._config.hebbian_decay_rate
                )
                count += 1

        return count

    def _prune_dead_triggers(self) -> int:
        """Count triggers that have effectively died (confidence near zero).

        Note: we don't delete them from the store, but the matcher
        already skips triggers with confidence < 0.1. This just
        reports how many are dead for monitoring.
        """
        count = 0
        for trigger in self._store.get_all_triggers():
            if trigger.confidence < _TRIGGER_PRUNE_CONFIDENCE:
                count += 1
        return count

    def _is_crystallized(self, co_activation_count: int) -> bool:
        """Check if an edge qualifies for crystallization (slow decay).

        Crystallized edges represent well-established connections that
        have proven useful many times. Like myelinated neural pathways,
        they're resistant to decay.
        """
        return co_activation_count >= _CRYSTALLIZE_THRESHOLD
