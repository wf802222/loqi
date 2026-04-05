"""Episode logging for Hebbian learning.

An episode records what happened during a single retrieval event:
what was the context, what was retrieved, what triggers fired,
and what was marked useful. This is the "experience buffer" that
all learning components draw from.

Think of it like a lab notebook for the memory system — every
retrieval is an experiment, and the episode log records the results
so the system can learn from them later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np


@dataclass
class Episode:
    """A single retrieval event with its outcome."""

    context: str
    context_embedding: np.ndarray | None = None
    retrieved_ids: list[str] = field(default_factory=list)
    triggered_ids: set[str] = field(default_factory=set)
    useful_ids: set[str] = field(default_factory=set)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def useless_ids(self) -> set[str]:
        """Retrieved documents that were NOT marked useful."""
        return set(self.retrieved_ids) - self.useful_ids

    @property
    def useful_trigger_ids(self) -> set[str]:
        """Triggers that fired AND their memory was useful."""
        return self.triggered_ids & self.useful_ids

    @property
    def useless_trigger_ids(self) -> set[str]:
        """Triggers that fired but their memory was NOT useful."""
        return self.triggered_ids - self.useful_ids


class EpisodeLog:
    """Append-only log of retrieval episodes.

    The log is the raw material for:
    - HebbianUpdater: which edges to strengthen/weaken
    - EdgePromoter: what contexts an edge fires in (for trigger pattern creation)
    - DecayManager: which edges/triggers haven't been used
    - Consolidator (future): replay and compress past experience
    """

    def __init__(self, max_episodes: int = 10000):
        self._episodes: list[Episode] = []
        self._max_episodes = max_episodes

    def record(self, episode: Episode) -> None:
        """Record a new episode."""
        self._episodes.append(episode)
        # Evict oldest if over capacity
        if len(self._episodes) > self._max_episodes:
            self._episodes = self._episodes[-self._max_episodes:]

    @property
    def episodes(self) -> list[Episode]:
        return list(self._episodes)

    def __len__(self) -> int:
        return len(self._episodes)

    def episodes_with_node(self, node_id: str) -> list[Episode]:
        """Find all episodes where a specific node was retrieved or triggered."""
        return [
            ep for ep in self._episodes
            if node_id in ep.retrieved_ids or node_id in ep.triggered_ids
        ]

    def episodes_with_edge(self, source_id: str, target_id: str) -> list[Episode]:
        """Find episodes where both nodes of an edge were co-activated."""
        return [
            ep for ep in self._episodes
            if source_id in set(ep.retrieved_ids) | ep.triggered_ids
            and target_id in set(ep.retrieved_ids) | ep.triggered_ids
        ]

    def useful_episodes_with_edge(self, source_id: str, target_id: str) -> list[Episode]:
        """Find episodes where both nodes were co-activated AND both useful."""
        return [
            ep for ep in self._episodes
            if source_id in ep.useful_ids and target_id in ep.useful_ids
        ]

    def context_embeddings_for_edge(
        self, source_id: str, target_id: str
    ) -> list[np.ndarray]:
        """Get context embeddings from episodes where an edge was usefully co-activated.

        Used by the promoter to build trigger patterns: when an edge promotes
        to a trigger, these embeddings define "what kind of context should
        the new trigger fire on?"
        """
        embeddings = []
        for ep in self.useful_episodes_with_edge(source_id, target_id):
            if ep.context_embedding is not None:
                embeddings.append(ep.context_embedding)
        return embeddings

    def clear(self) -> None:
        self._episodes.clear()
