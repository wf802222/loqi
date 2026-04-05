"""Tests for the consolidation engine.

Verifies that offline consolidation correctly replays episodes,
promotes edges, discovers bridges, and mines trigger candidates.
"""

import numpy as np
import pytest

from loqi.graph.models import Edge, EdgeType, Node, NodeType, TriggerOrigin
from loqi.graph.store import GraphStore
from loqi.hebbian.consolidator import Consolidator
from loqi.hebbian.episode import Episode, EpisodeLog
from loqi.pipeline.config import PipelineConfig


@pytest.fixture
def store():
    s = GraphStore(":memory:")
    # Create section nodes for testing
    for name in ["s_a", "s_b", "s_c", "s_d", "s_e"]:
        s.add_node(Node(id=name, title=name, node_type=NodeType.SECTION))
    yield s
    s.close()


@pytest.fixture
def episode_log():
    return EpisodeLog()


@pytest.fixture
def config():
    return PipelineConfig(
        hebbian_strengthen_rate=0.1,
        hebbian_decay_rate=0.02,
        hebbian_promotion_threshold_soft=2,
        hebbian_promotion_threshold_hard=4,
        hebbian_promotion_threshold_trigger=6,
    )


class TestReplay:
    def test_replays_strengthen_useful_edges(self, store, episode_log, config):
        store.add_edge(Edge(source_id="s_a", target_id="s_b", weight=0.3))

        # Record episodes with useful co-activations
        for i in range(3):
            episode_log.record(Episode(
                context=f"query {i}",
                retrieved_ids=["s_a", "s_b"],
                useful_ids={"s_a", "s_b"},
            ))

        consolidator = Consolidator(store, episode_log, config)
        report = consolidator.consolidate()

        edge = store.get_edge("s_a", "s_b")
        assert edge.weight > 0.3
        assert report.episodes_replayed > 0

    def test_only_replays_new_episodes(self, store, episode_log, config):
        store.add_edge(Edge(source_id="s_a", target_id="s_b", weight=0.3))

        # First batch
        episode_log.record(Episode(
            context="q1", retrieved_ids=["s_a", "s_b"], useful_ids={"s_a", "s_b"},
        ))

        consolidator = Consolidator(store, episode_log, config)
        consolidator.consolidate()
        weight_after_first = store.get_edge("s_a", "s_b").weight

        # Second batch — only new episodes should be replayed
        episode_log.record(Episode(
            context="q2", retrieved_ids=["s_a", "s_b"], useful_ids={"s_a", "s_b"},
        ))
        report = consolidator.consolidate()
        weight_after_second = store.get_edge("s_a", "s_b").weight

        assert weight_after_second > weight_after_first
        assert report.episodes_replayed == 1  # Only the new episode


class TestBridgeDiscovery:
    def test_discovers_bridge_between_strong_neighbors(self, store, episode_log, config):
        # A->B strong, B->C strong, but no A->C
        store.add_edge(Edge(source_id="s_b", target_id="s_a", weight=0.6))
        store.add_edge(Edge(source_id="s_b", target_id="s_c", weight=0.5))

        consolidator = Consolidator(store, episode_log, config)
        report = consolidator.consolidate()

        # A->C bridge should be created
        bridge = store.get_edge("s_a", "s_c")
        assert bridge is not None
        assert bridge.edge_type == EdgeType.DIFFUSE
        assert report.bridges_created >= 1

    def test_does_not_duplicate_existing_edges(self, store, episode_log, config):
        store.add_edge(Edge(source_id="s_b", target_id="s_a", weight=0.6))
        store.add_edge(Edge(source_id="s_b", target_id="s_c", weight=0.5))
        store.add_edge(Edge(source_id="s_a", target_id="s_c", weight=0.3))

        consolidator = Consolidator(store, episode_log, config)
        report = consolidator.consolidate()

        # No new bridge needed — A->C already exists
        assert report.bridges_created == 0

    def test_weak_edges_dont_create_bridges(self, store, episode_log, config):
        # Weak edges below min_weight (0.4) shouldn't produce bridges
        store.add_edge(Edge(source_id="s_b", target_id="s_a", weight=0.2))
        store.add_edge(Edge(source_id="s_b", target_id="s_c", weight=0.2))

        consolidator = Consolidator(store, episode_log, config)
        report = consolidator.consolidate()

        assert report.bridges_created == 0


class TestTriggerMining:
    def test_mines_trigger_from_frequently_useful_node(self, store, episode_log, config):
        from loqi.graph.embeddings import EmbeddingModel
        model = EmbeddingModel()

        # Record many episodes where s_a is useful
        for i in range(5):
            episode_log.record(Episode(
                context=f"deploy the webhook handler {i}",
                context_embedding=np.random.randn(384).astype(np.float32),
                retrieved_ids=["s_a", "s_b"],
                useful_ids={"s_a"},
            ))

        consolidator = Consolidator(store, episode_log, config, model)
        report = consolidator.consolidate()

        assert report.trigger_candidates >= 1

        # Check the trigger was created
        triggers = store.get_all_triggers()
        hebbian = [t for t in triggers if t.origin == TriggerOrigin.HEBBIAN]
        assert len(hebbian) >= 1
        assert hebbian[0].associated_node_id == "s_a"
        assert hebbian[0].confidence == 0.5  # Starts low

    def test_does_not_mine_if_trigger_already_exists(self, store, episode_log, config):
        from loqi.graph.embeddings import EmbeddingModel
        from loqi.graph.models import Trigger
        model = EmbeddingModel()

        # Pre-existing trigger for s_a
        store.add_trigger(Trigger(
            id="existing", pattern=["x"], associated_node_id="s_a",
        ))

        for i in range(5):
            episode_log.record(Episode(
                context=f"query {i}",
                context_embedding=np.random.randn(384).astype(np.float32),
                retrieved_ids=["s_a"], useful_ids={"s_a"},
            ))

        consolidator = Consolidator(store, episode_log, config, model)
        report = consolidator.consolidate()

        assert report.trigger_candidates == 0


class TestFullConsolidation:
    def test_consolidation_report_is_complete(self, store, episode_log, config):
        store.add_edge(Edge(source_id="s_a", target_id="s_b", weight=0.3))

        episode_log.record(Episode(
            context="test", retrieved_ids=["s_a", "s_b"], useful_ids={"s_a", "s_b"},
        ))

        consolidator = Consolidator(store, episode_log, config)
        report = consolidator.consolidate()

        # Report should have all fields populated
        assert report.episodes_replayed >= 0
        assert isinstance(report.promotions, list)
        assert report.bridges_created >= 0
        assert report.trigger_candidates >= 0
        assert isinstance(report.decay_summary, dict)

        # Summary should be printable
        summary = report.summary()
        assert "Consolidation Report" in summary
