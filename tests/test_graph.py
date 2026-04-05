"""Tests for graph storage and data models."""

import numpy as np
import pytest

from loqi.graph.embeddings import cosine_similarity, cosine_similarity_matrix
from loqi.graph.models import Edge, EdgeType, Node, Trigger, TriggerOrigin
from loqi.graph.store import GraphStore


@pytest.fixture
def store():
    """Create a fresh in-memory graph store for each test."""
    s = GraphStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def sample_node():
    embedding = np.random.randn(384).astype(np.float32)
    return Node(id="node_1", title="Test Node", content="Some content", embedding=embedding)


# ---------------------------------------------------------------------------
# Node CRUD
# ---------------------------------------------------------------------------

class TestNodeOperations:
    def test_add_and_get_node(self, store, sample_node):
        store.add_node(sample_node)
        retrieved = store.get_node("node_1")

        assert retrieved is not None
        assert retrieved.id == "node_1"
        assert retrieved.title == "Test Node"
        assert retrieved.content == "Some content"
        assert retrieved.access_count == 0

    def test_embedding_roundtrip(self, store, sample_node):
        store.add_node(sample_node)
        retrieved = store.get_node("node_1")

        assert retrieved.embedding is not None
        np.testing.assert_array_almost_equal(retrieved.embedding, sample_node.embedding)

    def test_get_nonexistent_node(self, store):
        assert store.get_node("nonexistent") is None

    def test_update_access(self, store, sample_node):
        store.add_node(sample_node)
        store.update_node_access("node_1")
        store.update_node_access("node_1")

        retrieved = store.get_node("node_1")
        assert retrieved.access_count == 2

    def test_get_all_nodes(self, store):
        for i in range(5):
            store.add_node(Node(id=f"n{i}", title=f"Node {i}"))
        assert len(store.get_all_nodes()) == 5

    def test_node_count(self, store):
        assert store.get_node_count() == 0
        store.add_node(Node(id="a"))
        store.add_node(Node(id="b"))
        assert store.get_node_count() == 2


# ---------------------------------------------------------------------------
# Edge CRUD
# ---------------------------------------------------------------------------

class TestEdgeOperations:
    def test_add_and_get_edge(self, store):
        store.add_node(Node(id="a"))
        store.add_node(Node(id="b"))

        edge = Edge(source_id="a", target_id="b", weight=0.8, edge_type=EdgeType.HARD)
        store.add_edge(edge)

        retrieved = store.get_edge("a", "b")
        assert retrieved is not None
        assert retrieved.weight == 0.8
        assert retrieved.edge_type == EdgeType.HARD

    def test_get_neighbors(self, store):
        store.add_node(Node(id="a"))
        store.add_node(Node(id="b"))
        store.add_node(Node(id="c"))

        store.add_edge(Edge(source_id="a", target_id="b", weight=0.9))
        store.add_edge(Edge(source_id="a", target_id="c", weight=0.3))

        neighbors = store.get_neighbors("a")
        assert len(neighbors) == 2
        # Should be sorted by weight descending
        assert neighbors[0][0] == "b"
        assert neighbors[1][0] == "c"

    def test_get_neighbors_by_type(self, store):
        store.add_node(Node(id="a"))
        store.add_node(Node(id="b"))
        store.add_node(Node(id="c"))

        store.add_edge(Edge(source_id="a", target_id="b", edge_type=EdgeType.HARD))
        store.add_edge(Edge(source_id="a", target_id="c", edge_type=EdgeType.DIFFUSE))

        hard_neighbors = store.get_neighbors("a", edge_type=EdgeType.HARD)
        assert len(hard_neighbors) == 1
        assert hard_neighbors[0][0] == "b"

    def test_strengthen_edge(self, store):
        store.add_node(Node(id="a"))
        store.add_node(Node(id="b"))
        store.add_edge(Edge(source_id="a", target_id="b", weight=0.5))

        store.strengthen_edge("a", "b", 0.1)
        edge = store.get_edge("a", "b")
        assert abs(edge.weight - 0.6) < 1e-6
        assert edge.co_activation_count == 1

    def test_strengthen_capped_at_1(self, store):
        store.add_node(Node(id="a"))
        store.add_node(Node(id="b"))
        store.add_edge(Edge(source_id="a", target_id="b", weight=0.95))

        store.strengthen_edge("a", "b", 0.2)
        edge = store.get_edge("a", "b")
        assert edge.weight == 1.0

    def test_decay_edge(self, store):
        store.add_node(Node(id="a"))
        store.add_node(Node(id="b"))
        store.add_edge(Edge(source_id="a", target_id="b", weight=0.5))

        store.decay_edge("a", "b", 0.1)
        edge = store.get_edge("a", "b")
        assert abs(edge.weight - 0.4) < 1e-6

    def test_decay_floored_at_0(self, store):
        store.add_node(Node(id="a"))
        store.add_node(Node(id="b"))
        store.add_edge(Edge(source_id="a", target_id="b", weight=0.05))

        store.decay_edge("a", "b", 0.2)
        edge = store.get_edge("a", "b")
        assert edge.weight == 0.0

    def test_promote_edge(self, store):
        store.add_node(Node(id="a"))
        store.add_node(Node(id="b"))
        store.add_edge(Edge(source_id="a", target_id="b", edge_type=EdgeType.DIFFUSE))

        store.promote_edge("a", "b", EdgeType.SOFT)
        edge = store.get_edge("a", "b")
        assert edge.edge_type == EdgeType.SOFT


# ---------------------------------------------------------------------------
# Trigger CRUD
# ---------------------------------------------------------------------------

class TestTriggerOperations:
    def test_add_and_get_trigger(self, store):
        store.add_node(Node(id="n1"))
        trigger = Trigger(
            id="t1",
            pattern=["frontend", "UI", "component"],
            associated_node_id="n1",
            origin=TriggerOrigin.EXPLICIT,
        )
        store.add_trigger(trigger)

        triggers = store.get_all_triggers()
        assert len(triggers) == 1
        assert triggers[0].id == "t1"
        assert triggers[0].pattern == ["frontend", "UI", "component"]

    def test_update_trigger_fire_useful(self, store):
        store.add_node(Node(id="n1"))
        store.add_trigger(Trigger(id="t1", pattern=["x"], associated_node_id="n1"))

        store.update_trigger_fire("t1", was_useful=True)
        t = store.get_all_triggers()[0]
        assert t.fire_count == 1
        assert t.useful_count == 1

    def test_update_trigger_fire_not_useful(self, store):
        store.add_node(Node(id="n1"))
        store.add_trigger(Trigger(id="t1", pattern=["x"], associated_node_id="n1"))

        store.update_trigger_fire("t1", was_useful=False)
        t = store.get_all_triggers()[0]
        assert t.fire_count == 1
        assert t.useful_count == 0

    def test_decay_trigger(self, store):
        store.add_node(Node(id="n1"))
        store.add_trigger(Trigger(id="t1", pattern=["x"], associated_node_id="n1", confidence=0.8))

        store.decay_trigger("t1", 0.3)
        t = store.get_all_triggers()[0]
        assert abs(t.confidence - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Bulk operations
# ---------------------------------------------------------------------------

class TestBulkOperations:
    def test_clear(self, store):
        store.add_node(Node(id="a"))
        store.add_node(Node(id="b"))
        store.add_edge(Edge(source_id="a", target_id="b"))
        store.add_trigger(Trigger(id="t1", pattern=["x"], associated_node_id="a"))

        store.clear()
        assert store.get_node_count() == 0
        assert store.get_edge_count() == 0
        assert store.get_trigger_count() == 0


# ---------------------------------------------------------------------------
# Embedding utilities
# ---------------------------------------------------------------------------

class TestEmbeddingUtils:
    def test_cosine_identical(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_cosine_opposite(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_cosine_matrix(self):
        query = np.array([1.0, 0.0, 0.0])
        corpus = np.array([
            [1.0, 0.0, 0.0],  # identical
            [0.0, 1.0, 0.0],  # orthogonal
            [0.5, 0.5, 0.0],  # partial
        ])
        sims = cosine_similarity_matrix(query, corpus)
        assert abs(sims[0] - 1.0) < 1e-6
        assert abs(sims[1]) < 1e-6
        assert sims[2] > 0

    def test_cosine_matrix_empty(self):
        query = np.array([1.0, 0.0])
        corpus = np.zeros((0, 2), dtype=np.float32)
        sims = cosine_similarity_matrix(query, corpus)
        assert len(sims) == 0
