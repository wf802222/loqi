# Loqi

*Memory that speaks when it matters.*

**Loqi** (from *loci* — the Latin root of the [memory palace technique](https://en.wikipedia.org/wiki/Method_of_loci)) is an experimental memory system for AI agents that preserves institutional knowledge across context loss.

## The Problem

Every AI coding assistant forgets. When the context window fills up, the system compacts — summarizing old messages, dropping details. Standing instructions like "never deploy on Thursday" or "payments need feature flags" get lost. The next time the agent works on a related task, it doesn't know the rules.

## What Loqi Does

Loqi stores standing instructions, project policies, and learned patterns in a persistent memory graph *outside* the context window. When context is compacted and the LLM forgets, Loqi re-injects the relevant rules before the next task.

**Compaction experiment (5 policy domains, 20 tasks, 3 models):**

| | With Loqi | Without Loqi |
|--|-----------|-------------|
| Before compaction (rules in context) | 63% compliance | 75% compliance |
| After compaction (rules erased) | **42-50%** | **15-28%** |
| Average advantage after compaction | | **+24pp** |

Without Loqi, models lose most institutional knowledge after compaction. With Loqi, policies are re-injected from external memory. The advantage holds across all three models tested.

**Tested across models:**

| Acting Model | Post-Compaction (Loqi) | Post-Compaction (No Loqi) | Delta |
|-------------|----------------------|-------------------------|-------|
| qwen2.5-coder:14b | 42% | 15% | +27pp |
| phi4 | 47% | 28% | +19pp |
| mistral-nemo:12b | 50% | 24% | +26pp |

## How It Works

Loqi runs as an external memory layer between the user and the acting model:

```
Task arrives
  -> Loqi retrieves relevant standing instructions (triggers + graph + semantic)
  -> Instructions injected into the acting model's prompt
  -> Model completes the task following the rules
  -> Episode logged: what was retrieved, what was useful
  -> Connections strengthened through Hebbian learning
  -> Next task benefits from accumulated knowledge
```

Three retrieval channels work together:

- **Triggers** — pattern-based pre-retrieval that fires on context match, not query mention. This is the primary contributor (+11pp over flat retrieval in ablation).
- **Graph traversal** — follows learned edges between related memory sections
- **Semantic search** — embedding similarity as the baseline

The system also learns. Connections between memories that are repeatedly useful together get strengthened. Eventually, frequently-useful patterns promote into new triggers — the system grows its own associative memory from usage.

## Current Evidence

| Experiment | Result |
|-----------|--------|
| Compaction resistance (expanded) | **+24pp average** across 3 models, 5 policy domains |
| Ablation: triggers | Primary contributor (+11pp over flat retrieval) |
| Proactive resurfacing | 80% precision (4/5 helpful, 4/5 correct silence) |
| Semantic-confound retrieval | Loqi 1.000 vs flat RAG 0.833 on ambiguous queries |
| LongMemEval (ICLR 2025) | 0.900 vs 0.867 on preference recall (n=30) |
| Closed loop | Proven — Hebbian learning creates triggers that fire on new queries |

## Architecture

Loqi is built around three phases of continuous memory formation:

**1. Write-time processing** — When a document arrives, Loqi splits it into section-level memory objects, computes embeddings, and discovers cross-section relationships with existing knowledge.

**2. Downtime consolidation** — Between work sessions, Loqi runs a consolidation cycle: decay stale edges, replay useful episodes, promote strong connections, discover bridge edges, and mine trigger candidates.

**3. Query-time orchestration** — Three independent channels (semantic, triggers, graph) retrieve and rank relevant sections. A local SmolLM2 model acts as a trigger suppression gate to prevent false positives.

## What This Is Not

- Not a production memory system
- Not a proven replacement for strong baseline RAG in all scenarios
- Not validated at large scale or long time horizons
- All benchmark data is synthetic (fictional policies, not real company data)

Loqi is a research prototype exploring whether AI memory should be proactive rather than purely query-driven.

## Quickstart

```bash
# Install
uv venv && uv pip install -e ".[dev,benchmarks]"

# Run tests
pytest

# Run benchmarks
python scripts/download_benchmarks.py
python scripts/run_hard_benchmark.py
```

### Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) recommended for reproducible installs (lockfile tracked)
- [Ollama](https://ollama.ai) with `smollm2:1.7b` for the LLM trigger gate (optional)
- No GPU required

## Repository Layout

```
src/loqi/
  graph/        — Node/Edge/Trigger models, SQLite store, MemoryWriter
  triggers/     — trigger extraction and matching
  retrieval/    — FlatRAG, GraphRAG, SectionRetrieval (v2)
  hebbian/      — episode log, updater, promoter, decay, consolidator
  llm/          — Ollama client, SmolLM trigger gate
  eval/         — metrics, protocol, evaluation runner
  benchmarks/   — data loaders (MuSiQue, HotpotQA, LongMemEval, MemoryAgentBench)
  pipeline/     — PipelineConfig with ablation toggles

scripts/        — benchmark and experiment runners
tests/          — unit and integration tests
data/           — custom benchmark scenarios
```

## Contributors

- **Wyn Fox** — architecture design, experiment design, project direction
- **Claude** (Anthropic, Opus 4.6) — implementation, evaluation harness, benchmarks, tests
- **GPT** (OpenAI, GPT-5.4) — strategic review, benchmark critique, architecture feedback

## License

[MIT](./LICENSE) — Copyright 2026 Wyn Fox

See [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md) for dependency and data licenses.
