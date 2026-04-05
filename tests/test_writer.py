"""Tests for write-time memory processing.

Verifies that documents are split into section-level memory objects
with proper containment edges and cross-section relationships.
"""

import pytest

from loqi.graph.embeddings import EmbeddingModel
from loqi.graph.models import EdgeType, NodeType
from loqi.graph.store import GraphStore
from loqi.graph.writer import MemoryWriter


@pytest.fixture
def store():
    s = GraphStore(":memory:")
    yield s
    s.close()


@pytest.fixture(scope="module")
def model():
    return EmbeddingModel()


CODING_STANDARDS = """\
# Coding Standards

## UI Component Rules
Always use COC (Component-on-Component) pattern in all UI components.
Never use FCOC (Functional Component-on-Component).

## API Conventions
All REST endpoints must use snake_case for field names.
The mobile team depends on this.

## Error Handling
Never swallow exceptions silently. Every catch block must either
re-raise, log at WARNING or above, or return an explicit error response.
"""

DEPLOYMENT_RULES = """\
# Deployment Rules

## Merge Freeze
No non-critical merges to main after Thursday each sprint.

## Database Migrations
All migrations must be backwards-compatible. Never drop a column
in the same release that stops using it.

## Feature Flags
New features touching payments must be behind a feature flag.
"""


class TestMemoryWriter:
    def test_splits_document_into_sections(self, store, model):
        writer = MemoryWriter(store, model)
        sections = writer.ingest_document(
            "coding_standards.md", "Coding Standards", CODING_STANDARDS
        )

        assert len(sections) == 3  # UI, API, Error Handling

    def test_creates_document_container_node(self, store, model):
        writer = MemoryWriter(store, model)
        writer.ingest_document(
            "coding_standards.md", "Coding Standards", CODING_STANDARDS
        )

        doc_node = store.get_node("coding_standards.md")
        assert doc_node is not None
        assert doc_node.node_type == NodeType.DOCUMENT
        assert doc_node.parent_id is None

    def test_creates_section_nodes(self, store, model):
        writer = MemoryWriter(store, model)
        sections = writer.ingest_document(
            "coding_standards.md", "Coding Standards", CODING_STANDARDS
        )

        for section in sections:
            node = store.get_node(section.id)
            assert node is not None
            assert node.node_type == NodeType.SECTION
            assert node.parent_id == "coding_standards.md"
            assert node.embedding is not None

    def test_section_ids_follow_convention(self, store, model):
        writer = MemoryWriter(store, model)
        sections = writer.ingest_document(
            "coding_standards.md", "Coding Standards", CODING_STANDARDS
        )

        assert sections[0].id == "coding_standards.md::s0"
        assert sections[1].id == "coding_standards.md::s1"
        assert sections[2].id == "coding_standards.md::s2"

    def test_creates_containment_edges(self, store, model):
        writer = MemoryWriter(store, model)
        sections = writer.ingest_document(
            "coding_standards.md", "Coding Standards", CODING_STANDARDS
        )

        for section in sections:
            edge = store.get_edge("coding_standards.md", section.id)
            assert edge is not None
            assert edge.weight == 1.0
            assert edge.edge_type == EdgeType.HARD

    def test_creates_sibling_edges(self, store, model):
        writer = MemoryWriter(store, model)
        sections = writer.ingest_document(
            "coding_standards.md", "Coding Standards", CODING_STANDARDS
        )

        # At least some sibling sections should be linked
        linked = False
        for i, s1 in enumerate(sections):
            for s2 in sections[i + 1:]:
                edge = store.get_edge(s1.id, s2.id)
                if edge is not None:
                    linked = True
                    assert edge.edge_type in (EdgeType.SOFT, EdgeType.DIFFUSE)
        assert linked, "At least some sibling sections should be linked"

    def test_cross_document_section_linking(self, store, model):
        writer = MemoryWriter(store, model)

        # Ingest first document
        writer.ingest_document(
            "coding_standards.md", "Coding Standards", CODING_STANDARDS
        )

        # Ingest second document — should create cross-document section edges
        writer.ingest_document(
            "deployment_rules.md", "Deployment Rules", DEPLOYMENT_RULES
        )

        # Check for cross-document edges between related sections
        # (e.g., "API Conventions" in coding_standards might link to
        # sections in deployment_rules)
        all_nodes = store.get_all_nodes()
        cs_sections = [n for n in all_nodes if n.parent_id == "coding_standards.md"]
        dr_sections = [n for n in all_nodes if n.parent_id == "deployment_rules.md"]

        cross_links = 0
        for cs in cs_sections:
            for dr in dr_sections:
                if store.get_edge(cs.id, dr.id) is not None:
                    cross_links += 1

        assert cross_links > 0, (
            "Related sections across documents should be linked"
        )

    def test_no_structure_fallback(self, store, model):
        """Document without ## headings becomes one section."""
        writer = MemoryWriter(store, model)
        sections = writer.ingest_document(
            "note.md", "Quick Note", "Just a plain text note without headings."
        )

        assert len(sections) == 1
        assert sections[0].node_type == NodeType.SECTION
        assert sections[0].parent_id == "note.md"

    def test_section_titles_from_headings(self, store, model):
        writer = MemoryWriter(store, model)
        sections = writer.ingest_document(
            "coding_standards.md", "Coding Standards", CODING_STANDARDS
        )

        assert sections[0].title == "UI Component Rules"
        assert sections[1].title == "API Conventions"
        assert sections[2].title == "Error Handling"
