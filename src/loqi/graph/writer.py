"""Write-time memory processing.

When new content arrives, this module performs the first act of cognition:
splitting documents into section-level memory objects, computing embeddings,
creating containment edges, and discovering initial cross-section relationships.

This is NOT preprocessing. It is the system forming first impressions of
new information and connecting it to what it already knows.

The key difference from traditional RAG indexing: we don't just store the
document. We actively look at what's already in the graph and create
tentative connections between the new content and existing knowledge.
"""

from __future__ import annotations

import numpy as np

from loqi.graph.embeddings import EmbeddingModel, cosine_similarity_matrix
from loqi.graph.models import Edge, EdgeType, Node, NodeType
from loqi.graph.store import GraphStore
from loqi.triggers.extractor import _split_markdown_sections


class MemoryWriter:
    """Processes new content into the memory graph at write time.

    Handles:
    1. Document -> section splitting
    2. Section node creation with embeddings
    3. Containment edges (document -> section)
    4. Cross-section relationship discovery (new sections vs existing graph)
    5. Trigger extraction (delegated to the trigger system)
    """

    def __init__(
        self,
        store: GraphStore,
        embedding_model: EmbeddingModel,
        cross_section_threshold: float = 0.10,
    ):
        self._store = store
        self._model = embedding_model
        self._threshold = cross_section_threshold

    def ingest_document(self, doc_id: str, title: str, content: str) -> list[Node]:
        """Process a new document into section-level memory objects.

        Returns the list of section nodes created (for downstream use
        by trigger extraction, retrieval indexing, etc.)
        """
        # Split into sections
        sections = _split_markdown_sections(content)

        if not sections:
            # No structure — treat whole document as one section
            sections = [(title, content)]

        # Create the document container node (no embedding — it's a container)
        doc_node = Node(
            id=doc_id,
            title=title,
            content="",  # Container has no content of its own
            node_type=NodeType.DOCUMENT,
            parent_id=None,
        )
        self._store.add_node(doc_node)

        # Create section nodes
        section_nodes = []
        section_texts = []

        for i, (heading, body) in enumerate(sections):
            section_id = f"{doc_id}::s{i}"
            full_text = f"{heading}\n{body}" if heading else body

            section_node = Node(
                id=section_id,
                title=heading or title,
                content=body,
                node_type=NodeType.SECTION,
                parent_id=doc_id,
            )
            section_nodes.append(section_node)
            section_texts.append(full_text)

        # Compute embeddings for all sections at once (batch efficiency)
        if section_texts:
            embeddings = self._model.encode(section_texts)
            for node, emb in zip(section_nodes, embeddings):
                node.embedding = emb

        # Store section nodes
        for node in section_nodes:
            self._store.add_node(node)

        # Create containment edges (document -> section)
        for node in section_nodes:
            self._store.add_edge(Edge(
                source_id=doc_id,
                target_id=node.id,
                weight=1.0,
                edge_type=EdgeType.HARD,
            ))

        # Discover cross-section relationships within this document
        self._link_sibling_sections(section_nodes, embeddings if section_texts else None)

        # Discover relationships with existing sections in the graph
        self._link_to_existing_sections(section_nodes)

        return section_nodes

    def _link_sibling_sections(
        self, sections: list[Node], embeddings: np.ndarray | None
    ) -> None:
        """Create edges between sections within the same document.

        Sibling sections always get at least a weak DIFFUSE edge —
        the author put them in the same file for a reason (shared provenance).
        Similarity determines edge weight and type.
        """
        if len(sections) < 2:
            return

        if embeddings is not None:
            n = len(sections)
            norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            sim_matrix = norms @ norms.T
        else:
            sim_matrix = None

        for i in range(len(sections)):
            for j in range(i + 1, len(sections)):
                # Siblings always link — shared provenance
                sim = float(sim_matrix[i, j]) if sim_matrix is not None else 0.15
                weight = max(sim, 0.15)  # Minimum weight for siblings
                edge_type = EdgeType.SOFT if sim >= 0.4 else EdgeType.DIFFUSE

                self._store.add_edge(Edge(
                    source_id=sections[i].id,
                    target_id=sections[j].id,
                    weight=weight,
                    edge_type=edge_type,
                ))
                self._store.add_edge(Edge(
                    source_id=sections[j].id,
                    target_id=sections[i].id,
                    weight=weight,
                    edge_type=edge_type,
                ))

    def _link_to_existing_sections(self, new_sections: list[Node]) -> None:
        """Compare new sections against ALL existing sections in the graph.

        This is the "first impression" step — when new content arrives,
        the system immediately notices what it's related to in existing
        knowledge. These tentative edges can later be strengthened by
        Hebbian learning or weakened by decay.
        """
        if not new_sections:
            return

        # Get all existing section nodes (exclude the ones we just added)
        new_ids = {s.id for s in new_sections}
        existing_nodes = [
            n for n in self._store.get_all_nodes()
            if n.node_type == NodeType.SECTION
            and n.id not in new_ids
            and n.embedding is not None
        ]

        if not existing_nodes:
            return

        # Build embedding matrix for existing sections
        existing_embeddings = np.array(
            [n.embedding for n in existing_nodes], dtype=np.float32
        )

        # Compare each new section against all existing sections
        for new_node in new_sections:
            if new_node.embedding is None:
                continue

            similarities = cosine_similarity_matrix(
                new_node.embedding, existing_embeddings
            )

            for idx, sim in enumerate(similarities):
                sim_val = float(sim)
                if sim_val >= self._threshold:
                    existing_id = existing_nodes[idx].id
                    edge_type = EdgeType.SOFT if sim_val >= 0.5 else EdgeType.DIFFUSE

                    self._store.add_edge(Edge(
                        source_id=new_node.id,
                        target_id=existing_id,
                        weight=sim_val,
                        edge_type=edge_type,
                    ))
                    self._store.add_edge(Edge(
                        source_id=existing_id,
                        target_id=new_node.id,
                        weight=sim_val,
                        edge_type=edge_type,
                    ))
