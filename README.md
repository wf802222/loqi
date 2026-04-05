# Loqi

*Memory that speaks when it matters.*

**Loqi** (from *loci* — the Latin root of the [memory palace technique](https://en.wikipedia.org/wiki/Method_of_loci)) is an experimental memory system for AI agents that preserves institutional knowledge across context loss.

## The Problem

Every AI coding assistant forgets. When the context window fills up, the system compacts — summarizing old messages, dropping details. Standing instructions like "never deploy on Thursday" or "payments need feature flags" get lost. The next time the agent works on a related task, it doesn't know the rules.

## What Loqi Does

Loqi stores standing instructions, project policies, and learned patterns in a persistent memory graph *outside* the context window. When context is compacted and the LLM forgets, Loqi re-injects the relevant rules before the next task.

**Compaction experiment result:**

| | With Loqi | Without Loqi |
|--|-----------|-------------|
| Before compaction (rules in context) | 45% policy compliance | 53% policy compliance |
| After compaction (rules erased) | **63%** | **17%** |
| Compaction drop | **+18% (improved)** | **-36% (collapsed)** |

Without Loqi, the model drops from 53% to 17% — it loses almost all institutional knowledge. With Loqi, the model actually *improves* to 63% because Loqi surfaces only the relevant policies per task instead of a full policy dump.

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
- **Triggers** — pattern-based pre-retrieval that fires on context match, not query mention
- **Graph traversal** — follows learned edges between related memory sections
- **Semantic search** — embedding similarity as the baseline

The system also learns. Connections between memories that are repeatedly useful together get strengthened. Eventually, frequently-useful patterns promote into new triggers — the system grows its own associative memory from usage.

## Current Evidence

| Experiment | Result |
|-----------|--------|
| **Compaction resistance** | **+46pp** — Loqi 63% vs no-memory 17% after context loss |
| Semantic-confound retrieval | Loqi 1.000 vs flat RAG 0.833 on ambiguous queries |
| Trigger recall | 66.7% vs 20% baseline on standing instruction surfacing |
| LongMemEval (ICLR 2025) | 0.900 vs 0.867 on preference recall (n=30) |
| Closed loop | Proven — Hebbian learning creates triggers that fire on new queries |
| Agent loop (10 tasks) | 100% rule compliance with Loqi memory across all tasks |

## Architecture

Loqi is built around three phases of continuous memory formation:

**1. Write-time processing** — When a document arrives, Loqi splits it into section-level memory objects, computes embeddings, and discovers cross-section relationships with existing knowledge. This is not indexing — it's the first act of cognition.

**2. Downtime consolidation** — Between work sessions, Loqi runs a consolidation cycle: decay stale edges, replay useful episodes, promote strong connections, discover bridge edges, and mine trigger candidates. The brain analogy is sleep consolidation.

**3. Query-time orchestration** — Three independent channels (semantic, triggers, graph) retrieve and rank relevant sections. A local SmolLM2 model acts as a trigger suppression gate to prevent false positives.

## What This Is Not

- Not a production memory system
- Not a proven replacement for strong baseline RAG
- Not validated at large scale or long time horizons
- Not a general-purpose knowledge graph

Loqi is a research prototype exploring whether AI memory should be proactive rather than purely query-driven.

## Quickstart

```bash
# Install
uv venv && uv pip install -e ".[dev,benchmarks]"

# Run tests (166 passing)
pytest

# Run the compaction experiment (requires Ollama with qwen2.5-coder:14b)
python sandbox/compaction_experiment.py

# Run the agent loop (10-task coding assistant with memory)
python sandbox/agent_loop.py --mode loqi_memory

# Run benchmarks
python scripts/download_benchmarks.py
python scripts/run_hard_benchmark.py
```

### Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) recommended for reproducible installs (lockfile tracked). pip works but resolves dependencies fresh.
- [Ollama](https://ollama.ai) with `qwen2.5-coder:14b` and `smollm2:1.7b` (for agent loop and LLM gate)
- No GPU required — all models run on CPU

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

sandbox/        — agent loop and compaction experiment
scripts/        — benchmark runners
tests/          — 166 unit and integration tests
```

## Contributors

- **Wyn Fox** — architecture design, experiment design, project direction
- **Claude** (Anthropic, Opus 4.6) — implementation, evaluation harness, benchmarks, tests
- **GPT** (OpenAI, GPT-5.4) — strategic review, benchmark critique, architecture feedback

## License

[MIT](./LICENSE) — Copyright 2026 Wyn Fox

See [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md) for dependency and data licenses.
