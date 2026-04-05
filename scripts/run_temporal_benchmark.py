"""Temporal benchmark — tests Loqi as a continuous memory formation system.

Simulates a developer working on a project over time:
1. Documents arrive incrementally (not all at once)
2. Downtime consolidation runs between work sessions
3. Queries are interleaved with document creation
4. The system should get smarter over time

Key metric: connection lead time — did the system form the useful
link before the query arrived?

Usage: python scripts/run_temporal_benchmark.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loqi.benchmarks.schema import Document
from loqi.graph.embeddings import EmbeddingModel
from loqi.graph.models import EdgeType, NodeType, TriggerOrigin
from loqi.pipeline.config import PipelineConfig
from loqi.retrieval.flat_rag import FlatRAG
from loqi.retrieval.section_retrieval import SectionRetrieval


# --- The memory corpus, introduced incrementally ---

DOCUMENTS = [
    {
        "id": "coding_standards.md",
        "title": "Coding Standards",
        "content": (
            "# Coding Standards\n\n"
            "## UI Component Rules\n"
            "Always use COC (Component-on-Component) pattern in all UI components. "
            "Never use FCOC (Functional Component-on-Component). "
            "This was decided after the Q2 refactor broke three production pages.\n\n"
            "## API Conventions\n"
            "All REST endpoints must use snake_case for field names. "
            "The mobile team depends on this — they auto-generate Swift models from the API schema.\n\n"
            "## Error Handling\n"
            "Never swallow exceptions silently. Every catch block must either re-raise, "
            "log at WARNING or above, or return an explicit error response.\n"
        ),
        "day": 1,
    },
    {
        "id": "deployment_rules.md",
        "title": "Deployment Rules",
        "content": (
            "# Deployment Rules\n\n"
            "## Merge Freeze\n"
            "No non-critical merges to main after Thursday each sprint. "
            "The mobile team cuts their release branch on Friday morning.\n\n"
            "## Database Migrations\n"
            "All migrations must be backwards-compatible. Never drop a column "
            "in the same release that stops using it.\n\n"
            "## Feature Flags\n"
            "New features touching payments must be behind a feature flag. "
            "No exceptions. The compliance team requires a 24-hour bake period.\n"
        ),
        "day": 2,
    },
    {
        "id": "api_rate_limits.md",
        "title": "API Rate Limits",
        "content": (
            "# API Rate Limits\n\n"
            "## Current Limits\n"
            "Public endpoints: 100 requests/minute per IP. "
            "Authenticated endpoints: 1000 requests/minute per user.\n\n"
            "## Third-Party API Limits\n"
            "Stripe: 100 requests/second (we are at ~40 in peak). "
            "SendGrid: 600 requests/minute for transactional email. "
            "Twilio: 1 request/second for SMS (bottleneck for OTP flows).\n"
        ),
        "day": 3,
    },
    {
        "id": "project_auth_redesign.md",
        "title": "Auth System Redesign",
        "content": (
            "# Auth System Redesign\n\n"
            "## Background\n"
            "We are replacing the old session-token auth middleware because legal "
            "flagged it for storing tokens in a way that does not meet SOC2 compliance.\n\n"
            "## Architecture Decision\n"
            "Using JWT with short-lived access tokens (15 min) and rotating refresh tokens. "
            "The refresh token store is PostgreSQL, not Redis, because we need audit trails.\n"
        ),
        "day": 5,
    },
]

# --- Work queries that arrive after documents ---

WORK_QUERIES = [
    # Day 4: after coding_standards and deployment_rules, before api_rate_limits
    {
        "day": 4,
        "context": "Build a new checkout form component for the payment page",
        "expected_sections": ["coding_standards.md::s0"],  # UI Component Rules
        "domain": "frontend",
    },
    # Day 6: after all docs are ingested
    {
        "day": 6,
        "context": "Deploy the Stripe subscription billing feature to production",
        "expected_sections": ["api_rate_limits.md::s1", "deployment_rules.md::s2"],
        "domain": "stripe_deploy",
    },
    {
        "day": 6,
        "context": "Add error handling to the payment webhook retry logic",
        "expected_sections": ["coding_standards.md::s2", "api_rate_limits.md::s1"],
        "domain": "error_handling",
    },
    # Day 7: interleaved work
    {
        "day": 7,
        "context": "Migrate the auth session middleware to use the new JWT tokens",
        "expected_sections": ["project_auth_redesign.md::s1", "deployment_rules.md::s1"],
        "domain": "auth",
    },
    {
        "day": 7,
        "context": "Write a backwards-compatible migration for the new token schema",
        "expected_sections": ["deployment_rules.md::s1"],
        "domain": "migration",
    },
    # Day 8: novel combination the system should predict
    {
        "day": 8,
        "context": "Ship the new payment retry service with Stripe webhooks before Thursday",
        "expected_sections": [
            "api_rate_limits.md::s1",
            "deployment_rules.md::s0",
            "deployment_rules.md::s2",
        ],
        "domain": "complex",
    },
]


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    model = EmbeddingModel()

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

    # System A: architecture v2 (section-level retrieval)
    system_a = SectionRetrieval(config=config, embedding_model=model)

    # System B: flat RAG baseline (document-level)
    flat_b = FlatRAG(embedding_model=model)

    print("=" * 65)
    print("TEMPORAL BENCHMARK -- Continuous Memory Formation")
    print("=" * 65)

    all_events = []
    for doc in DOCUMENTS:
        all_events.append(("doc", doc["day"], doc))
    for query in WORK_QUERIES:
        all_events.append(("query", query["day"], query))
    all_events.sort(key=lambda x: (x[1], 0 if x[0] == "doc" else 1))

    query_results = []
    current_day = 0
    all_docs_for_b = []

    for event_type, day, data in all_events:
        if day > current_day:
            if current_day > 0:
                report = system_a.consolidate()
                print(f"\n  [Consolidation after day {current_day}]")
                print(f"    Edges strengthened: {report.edges_strengthened}")
                print(f"    Bridges discovered: {report.bridges_created}")
                print(f"    Promotions: {len(report.promotions)}")
                print(f"    Trigger candidates: {report.trigger_candidates}")
            current_day = day

        if event_type == "doc":
            print(f"\n  Day {day}: INGEST {data['id']}")
            doc_obj = Document(id=data["id"], title=data["title"], text=data["content"])
            system_a.index([doc_obj])
            all_docs_for_b.append(doc_obj)

            store = system_a._store
            print(f"    Graph: {store.get_node_count()} nodes, {store.get_edge_count()} edges")

        elif event_type == "query":
            print(f"\n  Day {day}: QUERY \"{data['context'][:55]}...\"")

            # System A: section-level retrieval
            result_a = system_a.retrieve(data["context"], top_k=6)
            triggered_a = result_a.triggered_memories
            surfaced_a = set(result_a.retrieved_ids[:6]) | triggered_a

            expected = set(data["expected_sections"])
            hit_a = surfaced_a & expected
            recall_a = len(hit_a) / len(expected) if expected else 1.0

            # Connection lead time
            store = system_a._store
            lead_edges = 0
            for exp_id in expected:
                neighbors = store.get_neighbors(exp_id, min_weight=0.1)
                lead_edges += len(neighbors)

            # Hebbian feedback
            system_a.update(data["context"], result_a, expected)

            # System B: flat document-level baseline
            flat_b.reset()
            flat_b.index(all_docs_for_b)
            result_b = flat_b.retrieve(data["context"], top_k=6)
            surfaced_b_docs = set(result_b.retrieved_ids[:6])
            hit_b = set()
            for exp_id in expected:
                parent_doc = exp_id.split("::")[0]
                if parent_doc in surfaced_b_docs:
                    hit_b.add(exp_id)
            recall_b = len(hit_b) / len(expected) if expected else 1.0

            # Channel attribution
            meta = result_a.metadata
            semantic = set(meta.get("semantic_section_ids", []))
            triggered = set(meta.get("triggered_section_ids", []))
            graph_disc = set(meta.get("graph_discovered_section_ids", []))

            print(f"    Loqi v2 (sections): recall={recall_a:.2f}")
            print(f"      semantic:  {sorted(semantic & expected) or '(none)'}")
            print(f"      triggered: {sorted(triggered & expected) or '(none)'}")
            print(f"      graph:     {sorted(graph_disc & expected) or '(none)'}")
            print(f"    Flat RAG (docs):    recall={recall_b:.2f}")
            print(f"    Lead edges: {lead_edges}")

            query_results.append({
                "day": day,
                "context": data["context"][:80],
                "domain": data["domain"],
                "expected": sorted(expected),
                "recall_v2": recall_a,
                "recall_flat": recall_b,
                "hit_v2": sorted(hit_a),
                "semantic_hits": sorted(semantic & expected),
                "trigger_hits": sorted(triggered & expected),
                "graph_hits": sorted(graph_disc & expected),
                "lead_edges": lead_edges,
            })

    # Final consolidation
    report = system_a.consolidate()
    print(f"\n  [Final consolidation]")
    print(f"    {report.summary()}")

    # Results
    print(f"\n{'=' * 65}")
    print("RESULTS")
    print(f"{'=' * 65}")

    print(f"\n{'Query':<55} {'Day':>4} {'v2':>5} {'Flat':>5} {'Lead':>5}")
    print("-" * 76)
    for r in query_results:
        d = r["recall_v2"] - r["recall_flat"]
        m = "+" if d > 0 else ("-" if d < 0 else " ")
        print(f"{r['context']:<55} {r['day']:>4} {r['recall_v2']:>4.2f} {r['recall_flat']:>4.2f}{m} {r['lead_edges']:>4}")

    mean_a = sum(r["recall_v2"] for r in query_results) / len(query_results)
    mean_b = sum(r["recall_flat"] for r in query_results) / len(query_results)
    mean_lead = sum(r["lead_edges"] for r in query_results) / len(query_results)

    print(f"\n{'OVERALL':<55} {'':>4} {mean_a:>.3f} {mean_b:>.3f}  {mean_lead:>4.1f}")
    print(f"  Delta: {mean_a - mean_b:+.3f}")

    # Final graph state
    store = system_a._store
    all_triggers = store.get_all_triggers()
    hebbian = [t for t in all_triggers if t.origin == TriggerOrigin.HEBBIAN]
    print(f"\nFinal graph state:")
    print(f"  Nodes: {store.get_node_count()}, Edges: {store.get_edge_count()}")
    print(f"  Triggers: {len(all_triggers)} total, {len(hebbian)} Hebbian")
    print(f"  Episodes: {len(system_a._episode_log)}")

    # Save
    output_dir = data_dir.parent / "results"
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "temporal_benchmark.json", "w") as f:
        json.dump({
            "summary": {
                "mean_recall_v2": mean_a,
                "mean_recall_flat": mean_b,
                "delta": mean_a - mean_b,
                "mean_lead_edges": mean_lead,
            },
            "per_query": query_results,
        }, f, indent=2)
    print(f"\nResults saved to {output_dir / 'temporal_benchmark.json'}")


if __name__ == "__main__":
    main()
