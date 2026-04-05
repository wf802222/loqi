"""Persistent-corpus benchmark for Hebbian learning gain.

Proves that the closed loop produces QUANTITATIVE improvement,
not just MECHANICAL function. Uses the fixed memory corpus with
training queries (where the system learns) and test queries
(where we measure if learning helped).

Design:
  Phase 1 (Training): Run N queries, provide usefulness feedback.
    The system learns which memories go together.
  Phase 2 (Test): Run M NEW queries. Measure trigger recall.
    Compare against a system that had no training.

If the trained system fires more correct triggers on test queries,
we've proven empirical advantage from Hebbian learning.

Usage: python scripts/run_persistent_benchmark.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loqi.benchmarks.schema import Document
from loqi.graph.embeddings import EmbeddingModel
from loqi.graph.models import EdgeType, TriggerOrigin
from loqi.pipeline.config import PipelineConfig
from loqi.retrieval.graph_rag import GraphRAG
from loqi.retrieval.trigger_rag import TriggerRAG


# --- Training queries with expected useful documents ---
# These simulate a developer working on a project over several weeks.
# Each query has a context and the documents that SHOULD be useful.

TRAINING_QUERIES = [
    # Week 1: Frontend sprint
    ("Refactor the header component to match the new design system", {"coding_standards.md"}),
    ("Build a new settings page with user profile editing", {"coding_standards.md", "user_preferences.md"}),
    ("Fix the dropdown menu that breaks on mobile", {"coding_standards.md"}),
    ("Add dark mode toggle to the sidebar navigation", {"coding_standards.md"}),

    # Week 2: API + payments work (heavy on api_rate_limits + deployment_rules pair)
    ("Create REST endpoints for the new billing dashboard", {"coding_standards.md", "api_rate_limits.md"}),
    ("Implement Stripe subscription upgrade flow", {"api_rate_limits.md", "deployment_rules.md"}),
    ("Add webhook handlers for Stripe payment events", {"api_rate_limits.md", "deployment_rules.md"}),
    ("Build the invoice generation API endpoint", {"coding_standards.md", "api_rate_limits.md"}),
    ("Deploy Stripe payment retry logic to staging", {"api_rate_limits.md", "deployment_rules.md"}),
    ("Test the Stripe webhook signature verification", {"api_rate_limits.md", "deployment_rules.md"}),

    # Week 3: Auth + deployment (heavy on auth + deployment pair)
    ("Migrate the session middleware to JWT tokens", {"project_auth_redesign.md", "deployment_rules.md"}),
    ("Set up feature flags for the new auth flow", {"deployment_rules.md", "project_auth_redesign.md"}),
    ("Plan the production rollout of the auth changes", {"project_auth_redesign.md", "deployment_rules.md"}),
    ("Write database migration for the new token schema", {"deployment_rules.md"}),
    ("Test the auth token rotation in staging", {"project_auth_redesign.md", "deployment_rules.md"}),
    ("Configure the auth middleware for backwards compatibility", {"project_auth_redesign.md", "deployment_rules.md"}),

    # Week 4: Mixed work (reinforcing key pairs)
    ("Add error handling to the payment processing pipeline", {"coding_standards.md", "api_rate_limits.md"}),
    ("Review the Stripe integration before Thursday deploy", {"api_rate_limits.md", "deployment_rules.md"}),
    ("Update the team onboarding docs with the new auth flow", {"team_context.md", "project_auth_redesign.md"}),
    ("Fix the API rate limiter that's drifting under load", {"api_rate_limits.md"}),
    ("Roll out the updated Stripe webhooks to production", {"api_rate_limits.md", "deployment_rules.md"}),
    ("Prepare the auth cutover runbook for the ops team", {"project_auth_redesign.md", "deployment_rules.md"}),
    ("Add coding standards linting to the API test suite", {"coding_standards.md", "api_rate_limits.md"}),
    ("Review deployment checklist for the payment feature", {"deployment_rules.md", "api_rate_limits.md"}),
]

# --- Test queries (NOT seen during training) ---
# These are new queries in similar domains. If the system learned,
# it should fire Hebbian triggers that surface the right memories.

TEST_QUERIES = [
    # --- EASY: Explicit triggers should handle these ---
    # (Both systems should score well — baseline sanity check)
    {
        "context": "Create a new modal component for the checkout confirmation",
        "expected_useful": {"coding_standards.md"},
        "domain": "easy",
    },
    {
        "context": "Add OAuth2 social login with Google and GitHub",
        "expected_useful": {"project_auth_redesign.md"},
        "domain": "easy",
    },

    # --- HARD: Hebbian should help, explicit triggers probably won't ---
    # These queries are about COMBINATIONS of memories that the system
    # should have learned go together during training, but that aren't
    # explicitly mentioned in any single memory file.
    {
        # Training taught: Stripe work needs both rate limits AND deploy rules
        # But the query doesn't mention Stripe, rate limits, or deployment
        "context": "Set up the new vendor payment processing pipeline",
        "expected_useful": {"api_rate_limits.md", "deployment_rules.md"},
        "domain": "learned_pair",
    },
    {
        # Training taught: auth changes need deployment rules
        # This query is about access control, not explicitly auth redesign
        "context": "Implement role-based access control for the admin panel",
        "expected_useful": {"project_auth_redesign.md", "deployment_rules.md"},
        "domain": "learned_pair",
    },
    {
        # Training taught: API work needs both coding standards AND rate limits
        # This query is about a new service, not explicitly API conventions
        "context": "Build the microservice for real-time order tracking",
        "expected_useful": {"coding_standards.md", "api_rate_limits.md"},
        "domain": "learned_pair",
    },
    {
        # Training taught: frontend + user prefs go together
        "context": "Customize the dashboard layout based on user role",
        "expected_useful": {"coding_standards.md", "user_preferences.md"},
        "domain": "learned_pair",
    },
    {
        # Training taught: team context + auth go together for onboarding
        "context": "Prepare the security review checklist for the new hire",
        "expected_useful": {"team_context.md", "project_auth_redesign.md"},
        "domain": "learned_pair",
    },
    {
        # Training taught: deploy + rate limits for production releases
        "context": "Run the load test before the production cutover",
        "expected_useful": {"api_rate_limits.md", "deployment_rules.md"},
        "domain": "learned_pair",
    },
    {
        # Training taught: coding standards + deploy for migration work
        "context": "Write the backward-compatible schema change for the new feature",
        "expected_useful": {"coding_standards.md", "deployment_rules.md"},
        "domain": "learned_pair",
    },
    {
        # Training taught: Stripe + deploy + rate limits all together
        "context": "Ship the payment reconciliation batch job to production",
        "expected_useful": {"api_rate_limits.md", "deployment_rules.md"},
        "domain": "learned_pair",
    },
]


def load_memory_docs(data_dir):
    """Load the 6 memory files as Document objects."""
    memories_dir = data_dir / "custom_benchmark" / "memories"
    docs = []
    for md_file in sorted(memories_dir.glob("*.md")):
        docs.append(Document(
            id=md_file.name,
            title=md_file.name,
            text=md_file.read_text(encoding="utf-8"),
        ))
    return docs


def build_system(config, model):
    """Build a TriggerRAG system with the given config."""
    base = GraphRAG(config=config, embedding_model=model)
    return TriggerRAG(base_system=base, config=config, embedding_model=model)


def evaluate_test_queries(system, test_queries):
    """Run test queries and compute per-query trigger recall."""
    results = []
    for tq in test_queries:
        result = system.retrieve(tq["context"], top_k=6)

        # What memories were surfaced (triggered or retrieved)?
        surfaced = set(result.retrieved_ids) | result.triggered_memories
        expected = tq["expected_useful"]
        hit = surfaced & expected
        recall = len(hit) / len(expected) if expected else 1.0

        # Distinguish explicit vs Hebbian trigger contributions
        explicit_triggered = result.triggered_memories - {
            t.associated_node_id
            for t in (system._get_all_triggers() if hasattr(system, '_get_all_triggers') else [])
            if t.origin == TriggerOrigin.HEBBIAN
        }
        hebbian_triggered = result.triggered_memories - explicit_triggered

        results.append({
            "context": tq["context"][:80],
            "domain": tq["domain"],
            "expected": sorted(expected),
            "surfaced": sorted(surfaced),
            "hit": sorted(hit),
            "miss": sorted(expected - surfaced),
            "triggered_explicit": sorted(explicit_triggered),
            "triggered_hebbian": sorted(hebbian_triggered),
            "retrieved_rank": result.retrieved_ids[:6],
            "recall": recall,
            "hebbian_trigger_count": result.metadata.get("triggers_hebbian", 0),
        })
    return results


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    model = EmbeddingModel()

    # Aggressive learning thresholds
    config_learning = PipelineConfig(
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

    # Same config but Hebbian OFF
    config_no_learning = PipelineConfig(
        enable_graph=True,
        enable_triggers=True,
        enable_diffuse=True,
        enable_hebbian=False,
        trigger_confidence_threshold=0.10,
    )

    docs = load_memory_docs(data_dir)
    print(f"Loaded {len(docs)} memory documents")
    print(f"Training queries: {len(TRAINING_QUERIES)}")
    print(f"Test queries: {len(TEST_QUERIES)}")

    # === SYSTEM A: With Hebbian learning (trained) ===
    print(f"\n{'='*60}")
    print("SYSTEM A: Explicit triggers + Hebbian learning (trained)")
    print(f"{'='*60}")

    system_a = build_system(config_learning, model)
    system_a.index(docs)

    # Training phase
    print("\n  Training phase:")
    for i, (ctx, useful) in enumerate(TRAINING_QUERIES):
        result = system_a.retrieve(ctx, top_k=6)
        system_a.update(ctx, result, useful)
        if (i + 1) % 4 == 0:
            store = system_a._base._store
            hebbian_count = sum(
                1 for t in store.get_all_triggers()
                if t.origin == TriggerOrigin.HEBBIAN
            )
            print(f"    After {i+1} queries: {store.get_edge_count()} edges, {hebbian_count} Hebbian triggers")

    # Test phase
    print("\n  Test phase:")
    results_a = evaluate_test_queries(system_a, TEST_QUERIES)

    # === SYSTEM B: No Hebbian learning (untrained) ===
    print(f"\n{'='*60}")
    print("SYSTEM B: Explicit triggers only (no Hebbian learning)")
    print(f"{'='*60}")

    system_b = build_system(config_no_learning, model)
    system_b.index(docs)
    # No training phase — go straight to test
    results_b = evaluate_test_queries(system_b, TEST_QUERIES)

    # === COMPARISON ===
    print(f"\n{'='*60}")
    print("COMPARISON: Trained (A) vs Untrained (B)")
    print(f"{'='*60}")

    print(f"\n{'Query':<62} {'Domain':<8} {'A':>6} {'B':>6} {'Delta':>6}")
    print("-" * 90)
    for ra, rb in zip(results_a, results_b):
        delta = ra["recall"] - rb["recall"]
        marker = "+" if delta > 0 else " "
        print(f"{ra['context']:<62} {ra['domain']:<8} {ra['recall']:>5.2f} {rb['recall']:>5.2f} {marker}{delta:>5.2f}")

    # Aggregates
    mean_a = sum(r["recall"] for r in results_a) / len(results_a)
    mean_b = sum(r["recall"] for r in results_b) / len(results_b)
    delta = mean_a - mean_b

    print(f"\n{'OVERALL':<62} {'':8} {mean_a:>5.3f} {mean_b:>5.3f} {'+' if delta > 0 else ''}{delta:>5.3f}")

    # Per-domain breakdown
    domains = sorted(set(r["domain"] for r in results_a))
    print(f"\n{'Domain':<12} {'Trained':>10} {'Untrained':>10} {'Delta':>10}")
    print("-" * 44)
    for domain in domains:
        a_vals = [r["recall"] for r in results_a if r["domain"] == domain]
        b_vals = [r["recall"] for r in results_b if r["domain"] == domain]
        a_mean = sum(a_vals) / len(a_vals)
        b_mean = sum(b_vals) / len(b_vals)
        d = a_mean - b_mean
        print(f"{domain:<12} {a_mean:>10.3f} {b_mean:>10.3f} {'+' if d > 0 else ''}{d:>9.3f}")

    # Hebbian trigger analysis
    hebbian_fires_a = sum(1 for r in results_a if r["hebbian_trigger_count"] > 0)
    print(f"\nHebbian triggers fired on test queries: {hebbian_fires_a}/{len(results_a)}")

    # Per-query diagnostic detail
    print(f"\n{'='*60}")
    print("DIAGNOSTIC: Per-query trigger attribution")
    print(f"{'='*60}")
    for ra, rb in zip(results_a, results_b):
        improved = ra["recall"] > rb["recall"]
        if improved or ra.get("triggered_hebbian"):
            print(f"\n  Query: {ra['context']}")
            print(f"    Expected: {ra['expected']}")
            print(f"    Trained:  hit={ra['hit']}, miss={ra['miss']}")
            print(f"      explicit triggers: {ra['triggered_explicit']}")
            print(f"      hebbian triggers:  {ra['triggered_hebbian']}")
            print(f"    Untrained: hit={rb['hit']}, miss={rb['miss']}")
            if improved:
                print(f"    ** IMPROVED by Hebbian: {ra['recall']:.2f} vs {rb['recall']:.2f}")

    # Final graph state
    store = system_a._base._store
    all_triggers = store.get_all_triggers()
    hebbian = [t for t in all_triggers if t.origin == TriggerOrigin.HEBBIAN]
    print(f"\nFinal graph state (System A):")
    print(f"  Nodes: {store.get_node_count()}")
    print(f"  Edges: {store.get_edge_count()}")
    print(f"  Hebbian triggers: {len(hebbian)}")
    for t in hebbian:
        print(f"    {t.id}:")
        print(f"      target: {t.associated_node_id}")
        print(f"      confidence: {t.confidence:.2f}")
        print(f"      pattern: {t.pattern[:8]}")

    # Edge type distribution
    edge_types = {"hard": 0, "soft": 0, "diffuse": 0}
    for node in store.get_all_nodes():
        for _, edge in store.get_neighbors(node.id):
            edge_types[edge.edge_type.value] = edge_types.get(edge.edge_type.value, 0) + 1
    print(f"  Edge types: {edge_types}")

    # Save results with full diagnostics
    output_dir = data_dir.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "persistent_benchmark.json"
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "system_a_mean_recall": mean_a,
                "system_b_mean_recall": mean_b,
                "delta": delta,
                "delta_learned_pair": (
                    sum(r["recall"] for r in results_a if r["domain"] == "learned_pair")
                    - sum(r["recall"] for r in results_b if r["domain"] == "learned_pair")
                ) / max(1, sum(1 for r in results_a if r["domain"] == "learned_pair")),
                "hebbian_triggers_created": len(hebbian),
                "hebbian_triggers_fired_on_test": hebbian_fires_a,
                "training_queries": len(TRAINING_QUERIES),
                "test_queries": len(TEST_QUERIES),
            },
            "hebbian_triggers": [
                {
                    "id": t.id,
                    "target": t.associated_node_id,
                    "confidence": t.confidence,
                    "pattern": t.pattern[:10],
                    "origin": t.origin.value,
                }
                for t in hebbian
            ],
            "per_query_trained": results_a,
            "per_query_untrained": results_b,
            "graph_state": {
                "nodes": store.get_node_count(),
                "edges": store.get_edge_count(),
                "edge_types": edge_types,
            },
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
