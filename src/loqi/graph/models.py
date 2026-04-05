"""Core data models for Loqi's memory graph.

These models define the three fundamental structures:
  - Node: a memory unit (document, chunk, or standing instruction)
  - Edge: a weighted connection between nodes (hard/soft/diffuse)
  - Trigger: a pre-retrieval pattern that fires on context match

All models use Pydantic for validation and serialization.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class EdgeType(str, enum.Enum):
    """Edge types in the Hebbian promotion hierarchy.

    Promotion path: DIFFUSE -> SOFT -> HARD -> (promotes to Trigger)
    """

    HARD = "hard"        # explicit link (e.g., wikilink, citation)
    SOFT = "soft"        # semantic similarity above threshold
    DIFFUSE = "diffuse"  # speculative, from co-activation or random walk


class NodeType(str, enum.Enum):
    """What level of the memory hierarchy this node represents."""

    DOCUMENT = "document"  # container / provenance boundary
    SECTION = "section"    # primary memory object (##-delimited)


class TriggerOrigin(str, enum.Enum):
    """How a trigger was created."""

    EXPLICIT = "explicit"        # user stated a rule directly
    INFERRED = "inferred"        # LLM inferred from conversation
    HEBBIAN = "hebbian"          # promoted from repeated useful diffuse retrievals


class Node(BaseModel):
    """A memory node in the graph.

    Hierarchy:
      Document (container) -> Section (primary memory object)

    Sections are the canonical memory unit — what gets retrieved, linked,
    strengthened, and resurfaced. Documents are provenance containers.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    title: str = ""
    content: str = ""
    node_type: NodeType = NodeType.SECTION
    parent_id: str | None = None  # document ID for sections, None for documents
    embedding: np.ndarray | None = Field(default=None, exclude=True)
    access_count: int = 0
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Edge(BaseModel):
    """A weighted edge between two nodes.

    Edges strengthen through co-activation (Hebbian learning) and can
    promote through the hierarchy: diffuse -> soft -> hard -> trigger.
    """

    source_id: str
    target_id: str
    weight: float = 0.5
    edge_type: EdgeType = EdgeType.DIFFUSE
    co_activation_count: int = 0
    last_strengthened: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def is_promotable(self) -> bool:
        """Whether this edge has been co-activated enough to consider promotion."""
        return self.co_activation_count > 0


class Trigger(BaseModel):
    """An associative trigger that fires on context match.

    Triggers are created at write time and inject their associated node
    into the retrieval context before the query is processed.
    """

    id: str
    pattern: list[str]
    pattern_embedding: np.ndarray | None = Field(default=None, exclude=True)
    associated_node_id: str
    confidence: float = 1.0
    fire_count: int = 0
    useful_count: int = 0
    origin: TriggerOrigin = TriggerOrigin.EXPLICIT
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def usefulness_ratio(self) -> float:
        """Fraction of firings that were marked useful by synthesis."""
        if self.fire_count == 0:
            return 0.0
        return self.useful_count / self.fire_count
