"""Tests for the Hebbian learning loop.

Tests the full learning pipeline: episode logging, usefulness-gated
strengthening, edge promotion, trigger creation, and decay/pruning.
"""

import numpy as np
import pytest

from loqi.graph.models import Edge, EdgeType, Node, Trigger, TriggerOrigin
from loqi.graph.store import GraphStore
from loqi.hebbian.decay import DecayManager
from loqi.hebbian.episode import Episode, EpisodeLog
from loqi.hebbian.promoter import EdgePromoter
from loqi.hebbian.updater import HebbianUpdater
from loqi.pipeline.config import LOQI_FULL, PipelineConfig


@pytest.fixture
def store():
    s = GraphStore(":memory:")
    # Add some test nodes
    for name in ["a", "b", "c", "d"]:
        s.add_node(Node(id=name, title=f"Node {name}"))
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
        hebbian_promotion_threshold_soft=3,
        hebbian_promotion_threshold_hard=6,
        hebbian_promotion_threshold_trigger=10,
    )


# ---------------------------------------------------------------------------
# Episode logging
# ---------------------------------------------------------------------------

class TestEpisodeLog:
    def test_record_and_retrieve(self, episode_log):
        ep = Episode(context="test query", retrieved_ids=["a", "b"], useful_ids={"a"})
        episode_log.record(ep)
        assert len(episode_log) == 1

    def test_useless_ids(self):
        ep = Episode(retrieved_ids=["a", "b", "c"], useful_ids={"a"}, context="x")
        assert ep.useless_ids == {"b", "c"}

    def test_useful_trigger_ids(self):
        ep = Episode(
            triggered_ids={"a", "b"}, useful_ids={"a", "c"},
            retrieved_ids=["a", "b", "c"], context="x",
        )
        assert ep.useful_trigger_ids == {"a"}
        assert ep.useless_trigger_ids == {"b"}

    def test_episodes_with_node(self, episode_log):
        episode_log.record(Episode(context="q1", retrieved_ids=["a", "b"], useful_ids={"a"}))
        episode_log.record(Episode(context="q2", retrieved_ids=["c", "d"], useful_ids={"c"}))
        episode_log.record(Episode(context="q3", retrieved_ids=["a", "c"], useful_ids={"a", "c"}))

        assert len(episode_log.episodes_with_node("a")) == 2
        assert len(episode_log.episodes_with_node("d")) == 1
        assert len(episode_log.episodes_with_node("z")) == 0

    def test_useful_episodes_with_edge(self, episode_log):
        episode_log.record(Episode(context="q1", retrieved_ids=["a", "b"], useful_ids={"a", "b"}))
        episode_log.record(Episode(context="q2", retrieved_ids=["a", "b"], useful_ids={"a"}))  # b not useful
        episode_log.record(Episode(context="q3", retrieved_ids=["a", "b"], useful_ids={"a", "b"}))

        # Both useful in 2 out of 3 episodes
        assert len(episode_log.useful_episodes_with_edge("a", "b")) == 2

    def test_max_capacity(self):
        log = EpisodeLog(max_episodes=5)
        for i in range(10):
            log.record(Episode(context=f"q{i}", retrieved_ids=[], useful_ids=set()))
        assert len(log) == 5

    def test_context_embeddings_for_edge(self, episode_log):
        emb1 = np.array([1.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0], dtype=np.float32)

        episode_log.record(Episode(
            context="q1", context_embedding=emb1,
            retrieved_ids=["a", "b"], useful_ids={"a", "b"},
        ))
        episode_log.record(Episode(
            context="q2", context_embedding=emb2,
            retrieved_ids=["a", "b"], useful_ids={"a", "b"},
        ))

        embeddings = episode_log.context_embeddings_for_edge("a", "b")
        assert len(embeddings) == 2


# ---------------------------------------------------------------------------
# Hebbian updater
# ---------------------------------------------------------------------------

class TestHebbianUpdater:
    def test_strengthens_useful_pairs(self, store, episode_log, config):
        store.add_edge(Edge(source_id="a", target_id="b", weight=0.5))
        updater = HebbianUpdater(store, episode_log, config)

        episode = Episode(
            context="test", retrieved_ids=["a", "b"], useful_ids={"a", "b"},
        )
        updater.update(episode)

        edge = store.get_edge("a", "b")
        assert edge.weight > 0.5  # Strengthened
        assert edge.co_activation_count == 1

    def test_does_not_strengthen_useless(self, store, episode_log, config):
        store.add_edge(Edge(source_id="a", target_id="b", weight=0.5))
        updater = HebbianUpdater(store, episode_log, config)

        # a is useful but b is not
        episode = Episode(
            context="test", retrieved_ids=["a", "b"], useful_ids={"a"},
        )
        updater.update(episode)

        edge = store.get_edge("a", "b")
        # Should be slightly decayed, not strengthened
        assert edge.weight < 0.5

    def test_creates_new_edge_for_useful_pair(self, store, episode_log, config):
        # No edge between a and b initially
        assert store.get_edge("a", "b") is None

        updater = HebbianUpdater(store, episode_log, config)
        episode = Episode(
            context="test", retrieved_ids=["a", "b"], useful_ids={"a", "b"},
        )
        updater.update(episode)

        # New DIFFUSE edge should exist
        edge = store.get_edge("a", "b")
        assert edge is not None
        assert edge.edge_type == EdgeType.DIFFUSE
        assert edge.weight == 0.2

    def test_records_episode(self, store, episode_log, config):
        updater = HebbianUpdater(store, episode_log, config)
        episode = Episode(context="test", retrieved_ids=["a"], useful_ids={"a"})
        updater.update(episode)
        assert len(episode_log) == 1

    def test_trigger_feedback_useful(self, store, episode_log, config):
        store.add_trigger(Trigger(
            id="t1", pattern=["x"], associated_node_id="a", confidence=1.0,
        ))
        updater = HebbianUpdater(store, episode_log, config)

        episode = Episode(
            context="test", retrieved_ids=["a"],
            triggered_ids={"a"}, useful_ids={"a"},
        )
        updater.update(episode)

        trigger = store.get_all_triggers()[0]
        assert trigger.fire_count == 1
        assert trigger.useful_count == 1
        assert trigger.confidence == 1.0  # No decay — it was useful

    def test_trigger_feedback_not_useful(self, store, episode_log, config):
        store.add_trigger(Trigger(
            id="t1", pattern=["x"], associated_node_id="a", confidence=1.0,
        ))
        updater = HebbianUpdater(store, episode_log, config)

        episode = Episode(
            context="test", retrieved_ids=["a"],
            triggered_ids={"a"}, useful_ids=set(),  # Not useful
        )
        updater.update(episode)

        trigger = store.get_all_triggers()[0]
        assert trigger.fire_count == 1
        assert trigger.useful_count == 0
        assert trigger.confidence < 1.0  # Decayed


# ---------------------------------------------------------------------------
# Edge promoter
# ---------------------------------------------------------------------------

class TestEdgePromoter:
    def test_promote_diffuse_to_soft(self, store, episode_log, config):
        store.add_edge(Edge(
            source_id="a", target_id="b",
            edge_type=EdgeType.DIFFUSE, co_activation_count=3,
        ))
        promoter = EdgePromoter(store, episode_log, config)

        result = promoter.check_and_promote("a", "b")
        assert result == "soft"

        edge = store.get_edge("a", "b")
        assert edge.edge_type == EdgeType.SOFT

    def test_promote_soft_to_hard(self, store, episode_log, config):
        store.add_edge(Edge(
            source_id="a", target_id="b",
            edge_type=EdgeType.SOFT, co_activation_count=6,
        ))
        promoter = EdgePromoter(store, episode_log, config)

        result = promoter.check_and_promote("a", "b")
        assert result == "hard"

    def test_no_premature_promotion(self, store, episode_log, config):
        store.add_edge(Edge(
            source_id="a", target_id="b",
            edge_type=EdgeType.DIFFUSE, co_activation_count=2,  # Below threshold of 3
        ))
        promoter = EdgePromoter(store, episode_log, config)

        result = promoter.check_and_promote("a", "b")
        assert result is None

    def test_promote_hard_to_trigger(self, store, episode_log, config):
        store.add_edge(Edge(
            source_id="a", target_id="b",
            edge_type=EdgeType.HARD, co_activation_count=10,
        ))

        # Add some useful episodes so the trigger gets a pattern
        for i in range(3):
            episode_log.record(Episode(
                context=f"test context about webhooks {i}",
                context_embedding=np.random.randn(384).astype(np.float32),
                retrieved_ids=["a", "b"],
                useful_ids={"a", "b"},
            ))

        promoter = EdgePromoter(store, episode_log, config)
        result = promoter.check_and_promote("a", "b")

        assert result == "trigger"

        # A new Hebbian trigger should exist
        triggers = store.get_all_triggers()
        assert len(triggers) == 1
        assert triggers[0].origin == TriggerOrigin.HEBBIAN
        assert triggers[0].confidence == 0.7  # Starts lower than explicit
        assert triggers[0].associated_node_id == "b"
        assert len(triggers[0].pattern) > 0  # Keywords extracted from contexts


# ---------------------------------------------------------------------------
# Decay manager
# ---------------------------------------------------------------------------

class TestDecayManager:
    def test_decay_edges(self, store, config):
        store.add_edge(Edge(source_id="a", target_id="b", weight=0.5))
        decay = DecayManager(store, config)

        decay.run_decay_cycle()

        edge = store.get_edge("a", "b")
        assert edge.weight < 0.5

    def test_prune_weak_edges(self, store, config):
        store.add_edge(Edge(source_id="a", target_id="b", weight=0.03))
        decay = DecayManager(store, config)

        result = decay.run_decay_cycle()

        edge = store.get_edge("a", "b")
        assert edge.weight == 0.0  # Pruned to zero
        assert result["edges_pruned"] >= 1

    def test_crystallization_slows_decay(self, store, config):
        # Edge with many co-activations — should be crystallized
        store.add_edge(Edge(
            source_id="a", target_id="b", weight=0.8, co_activation_count=20,
        ))
        # Edge with few co-activations — normal decay
        store.add_edge(Edge(
            source_id="c", target_id="d", weight=0.8, co_activation_count=1,
        ))
        decay = DecayManager(store, config)

        decay.run_decay_cycle()

        crystal_edge = store.get_edge("a", "b")
        normal_edge = store.get_edge("c", "d")

        # Crystallized edge should have decayed less
        assert crystal_edge.weight > normal_edge.weight

    def test_out_degree_cap(self, store, config):
        # Give node "a" 25 outgoing edges (above cap of 20)
        for i in range(25):
            node_id = f"n{i}"
            store.add_node(Node(id=node_id))
            store.add_edge(Edge(
                source_id="a", target_id=node_id, weight=0.1 + i * 0.01,
            ))

        decay = DecayManager(store, config)
        result = decay.run_decay_cycle()

        # Should have pruned the weakest 5
        assert result["edges_capped"] >= 5

    def test_tick_only_runs_periodically(self, store, config):
        store.add_edge(Edge(source_id="a", target_id="b", weight=0.5))
        decay = DecayManager(store, config)

        # First 9 ticks should do nothing
        for _ in range(9):
            result = decay.tick()
            assert result == {}

        # 10th tick should run decay
        result = decay.tick()
        assert result != {}


# ---------------------------------------------------------------------------
# Integration: full learning pipeline
# ---------------------------------------------------------------------------

class TestHebbianIntegration:
    def test_edge_strengthens_over_multiple_episodes(self, store, episode_log, config):
        store.add_edge(Edge(source_id="a", target_id="b", weight=0.3))
        updater = HebbianUpdater(store, episode_log, config)

        initial_weight = 0.3
        for i in range(5):
            episode = Episode(
                context=f"query {i}",
                retrieved_ids=["a", "b"],
                useful_ids={"a", "b"},
            )
            updater.update(episode)

        edge = store.get_edge("a", "b")
        assert edge.weight > initial_weight
        assert edge.co_activation_count == 5

    def test_full_promotion_lifecycle(self, store, episode_log):
        """Test the complete path: no edge → diffuse → soft → hard → trigger.

        Note: the first useful co-activation CREATES a diffuse edge
        (co_activation_count=0). Subsequent episodes STRENGTHEN it
        (incrementing co_activation_count). So reaching threshold=2
        requires 3 total episodes (1 create + 2 strengthens).
        """
        config = PipelineConfig(
            hebbian_strengthen_rate=0.1,
            hebbian_promotion_threshold_soft=2,
            hebbian_promotion_threshold_hard=4,
            hebbian_promotion_threshold_trigger=6,
        )
        updater = HebbianUpdater(store, episode_log, config)
        promoter = EdgePromoter(store, episode_log, config)

        def episode(ctx: str) -> None:
            updater.update(Episode(
                context=ctx,
                context_embedding=np.random.randn(384).astype(np.float32),
                retrieved_ids=["a", "b"], useful_ids={"a", "b"},
            ))

        # Episode 1: creates DIFFUSE edge (co_activation_count=0)
        episode("webhook retry logic")
        edge = store.get_edge("a", "b")
        assert edge is not None
        assert edge.edge_type == EdgeType.DIFFUSE

        # Episodes 2-3: strengthen to co_activation_count=2 → SOFT
        episode("webhook error handling")
        episode("webhook timeout recovery")
        promoter.check_and_promote("a", "b")
        edge = store.get_edge("a", "b")
        assert edge.edge_type == EdgeType.SOFT

        # Episodes 4-5: strengthen to co_activation_count=4 → HARD
        episode("webhook circuit breaker")
        episode("webhook dead letter queue")
        promoter.check_and_promote("a", "b")
        edge = store.get_edge("a", "b")
        assert edge.edge_type == EdgeType.HARD

        # Episodes 6-7: strengthen to co_activation_count=6 → TRIGGER
        episode("webhook monitoring alerts")
        episode("webhook delivery failures")
        promoter.check_and_promote("a", "b")

        # A Hebbian trigger should now exist
        triggers = store.get_all_triggers()
        assert len(triggers) == 1
        assert triggers[0].origin == TriggerOrigin.HEBBIAN
        assert "webhook" in triggers[0].pattern  # Learned from context
