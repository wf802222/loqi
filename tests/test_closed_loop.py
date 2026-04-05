"""Integration tests for the closed feedback loop.

These tests verify the end-to-end claim: Hebbian learning creates
triggers that fire on future queries, improving retrieval over time.

This is the most important test in the project — it proves the thesis
that the three layers (triggers + retrieval + Hebbian) work together
as a closed feedback loop.
"""

import numpy as np
import pytest

from loqi.benchmarks.schema import Document
from loqi.eval.protocol import RetrievalResult
from loqi.graph.embeddings import EmbeddingModel
from loqi.graph.models import TriggerOrigin
from loqi.pipeline.config import PipelineConfig
from loqi.retrieval.graph_rag import GraphRAG
from loqi.retrieval.trigger_rag import TriggerRAG


@pytest.fixture(scope="module")
def model():
    return EmbeddingModel()


def _make_docs():
    """Create a small fixed corpus of memory documents."""
    return [
        Document(
            id="coding_standards",
            title="Coding Standards",
            text="Always use COC pattern in UI components. Never FCOC. "
            "All REST endpoints must use snake_case for field names.",
        ),
        Document(
            id="deployment_rules",
            title="Deployment Rules",
            text="No merges to main after Thursday. Database migrations "
            "must be backwards-compatible. Payment features need feature flags.",
        ),
        Document(
            id="api_limits",
            title="API Rate Limits",
            text="Stripe API limit is 100 requests per second. "
            "Webhook delivery limit is 50 per second. "
            "SendGrid limit is 600 per minute for transactional email.",
        ),
    ]


class TestClosedLoop:
    """Test that Hebbian learning creates triggers that fire on future queries."""

    def test_hebbian_trigger_is_created_and_fires(self, model):
        """End-to-end: repeated useful co-activation creates a trigger that fires."""
        # Use aggressive thresholds so promotion happens quickly
        config = PipelineConfig(
            enable_graph=True,
            enable_triggers=True,
            enable_diffuse=False,
            enable_hebbian=True,
            hebbian_strengthen_rate=0.15,
            hebbian_promotion_threshold_soft=2,
            hebbian_promotion_threshold_hard=3,
            hebbian_promotion_threshold_trigger=5,
            trigger_confidence_threshold=0.10,
        )

        base = GraphRAG(config=config, embedding_model=model)
        system = TriggerRAG(base_system=base, config=config, embedding_model=model)

        # Index the corpus once
        docs = _make_docs()
        system.index(docs)

        # Phase 1: Repeated queries where api_limits and deployment_rules
        # are both useful together (Stripe + deployment context)
        stripe_contexts = [
            "Deploy the new Stripe payment integration to production",
            "Roll out updated Stripe webhook handling to all servers",
            "Ship the Stripe subscription billing feature this sprint",
            "Release the Stripe coupon redemption flow to staging",
            "Push the Stripe invoice generation update to production",
            "Deploy Stripe payment retry logic to the live environment",
        ]

        for ctx in stripe_contexts:
            result = system.retrieve(ctx, top_k=3)
            # Simulate feedback: both api_limits and deployment_rules were useful
            system.update(
                ctx, result,
                useful_ids={"api_limits", "deployment_rules"},
            )

        # Check: a Hebbian trigger should have been created
        store_triggers = base._store.get_all_triggers()
        hebbian_triggers = [
            t for t in store_triggers if t.origin == TriggerOrigin.HEBBIAN
        ]
        assert len(hebbian_triggers) >= 1, (
            f"Expected at least 1 Hebbian trigger after 6 co-activations, "
            f"got {len(hebbian_triggers)}. "
            f"Store has {len(store_triggers)} total triggers."
        )

        # Phase 2: A NEW query in a similar context should now fire the
        # Hebbian trigger, surfacing both documents without explicit triggers
        new_query = "Integrate Stripe checkout for the mobile app release"
        result2 = system.retrieve(new_query, top_k=3)

        # The Hebbian trigger should have fired
        assert result2.metadata.get("triggers_hebbian", 0) >= 1, (
            f"Hebbian trigger should fire on Stripe+deployment context. "
            f"Metadata: {result2.metadata}"
        )

    def test_documents_persist_across_index_calls(self, model):
        """Verify that index() accumulates documents, not replaces."""
        config = PipelineConfig(enable_graph=True, enable_triggers=True)
        base = GraphRAG(config=config, embedding_model=model)
        system = TriggerRAG(base_system=base, config=config, embedding_model=model)

        # Index batch 1
        batch1 = [Document(id="doc_a", title="A", text="First document about cats")]
        system.index(batch1)

        # Index batch 2
        batch2 = [Document(id="doc_b", title="B", text="Second document about dogs")]
        system.index(batch2)

        # Both documents should be retrievable
        result = system.retrieve("animals", top_k=10)
        retrieved_ids = set(result.retrieved_ids)
        assert "doc_a" in retrieved_ids, "doc_a should persist after second index()"
        assert "doc_b" in retrieved_ids, "doc_b should be present"

    def test_trigger_fires_for_document_from_earlier_index(self, model):
        """Trigger should inject a document indexed earlier, not just current batch."""
        config = PipelineConfig(
            enable_graph=True,
            enable_triggers=True,
            trigger_confidence_threshold=0.10,
        )
        base = GraphRAG(config=config, embedding_model=model)
        system = TriggerRAG(base_system=base, config=config, embedding_model=model)

        # Index memory files in two separate calls
        system.index([Document(
            id="coding_rules",
            title="Coding Rules",
            text="Always use COC Component-on-Component pattern in all "
            "React UI frontend components. Never use FCOC.",
        )])
        system.index([Document(
            id="unrelated",
            title="Unrelated",
            text="The weather forecast for tomorrow is sunny with clouds.",
        )])

        # Query about UI work — should trigger coding_rules from first index
        result = system.retrieve("Build a new React dropdown component", top_k=5)
        assert "coding_rules" in result.triggered_memories or "coding_rules" in set(result.retrieved_ids), (
            f"coding_rules should be surfaced. Got: {result.retrieved_ids}, triggered: {result.triggered_memories}"
        )
