"""Run the trigger promotion scenarios against the fixed memory corpus.

Tests the Hebbian learning closed loop: repeated useful co-activation
should promote edges through DIFFUSE -> SOFT -> HARD -> Trigger.

Usage: python scripts/run_promotion.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loqi.benchmarks.custom_loader import load_memories, load_promotion_scenarios
from loqi.benchmarks.schema import Document
from loqi.graph.embeddings import EmbeddingModel
from loqi.graph.models import EdgeType, TriggerOrigin
from loqi.pipeline.config import PipelineConfig
from loqi.retrieval.graph_rag import GraphRAG
from loqi.retrieval.trigger_rag import TriggerRAG


def run_promotion_scenario(scenario, docs, model):
    """Run a single promotion scenario and report results."""
    # Aggressive thresholds for promotion testing
    config = PipelineConfig(
        enable_graph=True,
        enable_triggers=True,
        enable_diffuse=True,
        enable_hebbian=True,
        hebbian_strengthen_rate=0.15,
        hebbian_promotion_threshold_soft=2,
        hebbian_promotion_threshold_hard=4,
        hebbian_promotion_threshold_trigger=6,
        trigger_confidence_threshold=0.10,
    )

    base = GraphRAG(config=config, embedding_model=model)
    system = TriggerRAG(base_system=base, config=config, embedding_model=model)
    system.index(docs)

    print(f"\n{'='*60}")
    print(f"  {scenario.id}: {scenario.name}")
    print(f"  Category: {scenario.category}")
    if scenario.description:
        print(f"  {scenario.description.strip()}")
    print(f"{'='*60}")

    for i, step in enumerate(scenario.sequence):
        context = step.context
        expectations = step.expectations

        # Determine useful_ids from expectations
        useful_ids = set()
        for key, val in expectations.items():
            if key in ("expect_diffuse_useful", "mark_useful", "co_retrieved_useful"):
                if isinstance(val, list):
                    useful_ids.update(val)
            if key == "marked_useful" and val is False:
                useful_ids = set()  # Explicitly not useful

        # Run retrieval
        result = system.retrieve(context, top_k=len(docs))

        # Run Hebbian update with the expected useful feedback
        if useful_ids:
            system.update(context, result, useful_ids)
        elif "marked_useful" in expectations and expectations["marked_useful"] is False:
            # Fire but not useful — update with empty useful set
            system.update(context, result, set())

        # Report state
        store = base._store
        edge_count = store.get_edge_count()
        triggers = store.get_all_triggers()
        hebbian_triggers = [t for t in triggers if t.origin == TriggerOrigin.HEBBIAN]

        print(f"\n  Step {i+1}: \"{context[:60]}...\"")
        print(f"    Retrieved: {result.retrieved_ids[:3]}")
        print(f"    Triggered: {sorted(result.triggered_memories) if result.triggered_memories else '(none)'}")
        print(f"    Useful: {sorted(useful_ids) if useful_ids else '(none)'}")
        print(f"    Graph: {store.get_node_count()} nodes, {edge_count} edges, {len(hebbian_triggers)} Hebbian triggers")

        # Check edge expectations
        if "expect_edge_type" in expectations:
            exp = expectations["expect_edge_type"]
            edge = store.get_edge(exp["from"], exp["to"])
            actual_type = edge.edge_type.value if edge else "none"
            expected_type = exp["type"]
            match = actual_type == expected_type
            status = "PASS" if match else "FAIL"
            print(f"    Edge {exp['from']} -> {exp['to']}: expected={expected_type}, actual={actual_type} [{status}]")
            if edge:
                print(f"      weight={edge.weight:.3f}, co_activations={edge.co_activation_count}")

        # Check trigger expectations
        if "expect_trigger_now" in expectations:
            expected_triggers = set(expectations["expect_trigger_now"])
            fired = result.triggered_memories
            match = expected_triggers <= fired
            status = "PASS" if match else "FAIL"
            print(f"    Expected triggers: {sorted(expected_triggers)}, fired: {sorted(fired)} [{status}]")

        if "expect_trigger_decayed" in expectations:
            for doc_id in expectations["expect_trigger_decayed"]:
                relevant = [t for t in triggers if t.associated_node_id == doc_id]
                if relevant:
                    t = relevant[0]
                    decayed = t.confidence < 0.9
                    status = "PASS" if decayed else "FAIL"
                    print(f"    Trigger for {doc_id}: confidence={t.confidence:.3f}, fire_count={t.fire_count} [{status}]")

        if "expect_co_retrieved" in expectations:
            expected = set(expectations["expect_co_retrieved"])
            retrieved = set(result.retrieved_ids)
            match = expected <= retrieved
            status = "PASS" if match else "PARTIAL"
            print(f"    Expected co-retrieved: {sorted(expected)}, got: {sorted(retrieved & expected)} [{status}]")

    # Final summary
    print(f"\n  --- Final state ---")
    print(f"  Nodes: {store.get_node_count()}")
    print(f"  Edges: {store.get_edge_count()}")
    print(f"  Hebbian triggers: {len(hebbian_triggers)}")
    for t in hebbian_triggers:
        print(f"    {t.id}: confidence={t.confidence:.2f}, pattern={t.pattern[:5]}...")
    print(f"  Episodes logged: {len(base._episode_log)}")


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    model = EmbeddingModel()

    # Load memories and promotion scenarios
    memories = load_memories(data_dir / "custom_benchmark" / "memories")
    scenarios = load_promotion_scenarios(data_dir / "custom_benchmark" / "scenarios")

    print(f"Loaded {len(scenarios)} promotion scenarios, {len(memories)} memory files")

    # Create Document objects from memories
    docs = [
        Document(id=name, title=name, text=content)
        for name, content in memories.items()
    ]

    for scenario in scenarios:
        run_promotion_scenario(scenario, docs, model)

    print(f"\n{'='*60}")
    print("All promotion scenarios complete.")


if __name__ == "__main__":
    main()
