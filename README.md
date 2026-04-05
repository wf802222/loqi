# Loqi

*Memory that speaks when it matters.*

**Loqi** (a phonetic play on *loci*, the plural of *locus* — the Latin root of the [method of loci](https://en.wikipedia.org/wiki/Method_of_loci), the ancient memory palace technique) is an experimental memory architecture for AI agents. Where the classical memory palace anchors memories to specific locations, Loqi anchors them to *contexts* — surfacing what matters before you ask for it.

---

## What Loqi Does

Loqi is a continuous memory formation system. Instead of treating memory as a static document store queried after the fact, Loqi:

- Forms memory as work happens (write-time processing)
- Builds connections during downtime (consolidation / "dreaming")
- Retrieves through three independent channels (semantic, triggers, graph)
- Learns from feedback and grows new triggers from usage patterns (Hebbian closed loop)
- Uses a local LLM gate to suppress false triggers (v2.5)

## Current Evidence

| Claim | Result | Benchmark |
| --- | --- | --- |
| Triggers surface standing instructions | 66.7% recall vs 20% baseline | Custom trigger benchmark (25 scenarios) |
| Triggers work on published data | 0.900 vs 0.867 hit rate | LongMemEval preferences (ICLR 2025, n=30) |
| Semantic-confound retrieval | **Loqi 1.000 vs flat RAG 0.833** | Hard temporal benchmark |
| Co-activation learning | 0.833 (matches flat RAG after training) | Hard temporal benchmark |
| Trigger suppression | 1.000 with SmolLM gate | Hard temporal benchmark |
| Closed loop mechanism | Proven end-to-end | Integration test: co-activation -> promotion -> trigger creation -> trigger fires on new query |
| Overall hard benchmark | 0.879 vs 0.924 flat RAG | 11 queries across 5 difficulty categories |

The strongest result: Loqi beats flat RAG on **semantic-confound** queries — where multiple sections share vocabulary but differ in operational meaning. This is the environment the architecture was designed for.

## How It Works

### 1. Write-Time Processing

When a document arrives, Loqi splits it into **section-level memory objects** (by `##` headings), computes embeddings, creates containment edges, and discovers cross-section relationships with existing knowledge. This is not preprocessing — it is the first act of cognition.

### 2. Downtime Consolidation

Between work sessions, Loqi runs a consolidation cycle: decay stale edges, replay recent useful episodes, promote strong connections, discover bridge edges (A->B strong, B->C strong, propose A->C), and mine trigger candidates from repeated useful patterns.

### 3. Query-Time Retrieval

Three channels, each tracked separately:
- **Semantic** — cosine similarity between query and section embeddings
- **Triggers** — pre-retrieval pattern matching that fires on context, not query mention
- **Graph** — 2-hop BFS from entry sections along learned edges

### 4. Hebbian Closed Loop

The system learns from feedback: edges between co-activated useful sections strengthen, edges between useful and non-useful sections weaken, and frequently useful connections promote through DIFFUSE -> SOFT -> HARD -> **new Trigger**. The system literally grows its own trigger layer from usage patterns.

### 5. LLM Trigger Gate (v2.5)

A local SmolLM2 model (1.7B, via Ollama) acts as a cheap relevance filter on trigger candidates. It suppresses false positives that pass deterministic guards but are contextually irrelevant. The LLM does not create, modify, or delete memory — it only recommends suppress or allow.

## What This Repo Is Not

- A production memory system
- A demonstrated win over strong external baselines (HippoRAG, Microsoft GraphRAG)
- A proven large-scale long-horizon memory architecture
- A replacement for standard RAG in most use cases

The evidence is promising in some areas and limited in others.

## Repository Layout

```
src/loqi/
  graph/        — Node/Edge/Trigger models, SQLite store, embeddings, MemoryWriter
  triggers/     — trigger extraction (markdown + conversational) and matching
  retrieval/    — FlatRAG, GraphRAG, TriggerRAG, SectionRetrieval (v2)
  hebbian/      — episode log, updater, promoter, decay manager, consolidator
  llm/          — Ollama client, SmolLM trigger gate, Phi-4 arbitrator
  eval/         — metrics, protocol, evaluation runner
  benchmarks/   — data loaders for MuSiQue, HotpotQA, LongMemEval, MemoryAgentBench
  pipeline/     — PipelineConfig with ablation toggles

scripts/        — benchmark and experiment runners
tests/          — 166 unit and integration tests
data/           — custom benchmark corpus and temporal benchmark scenarios
results/        — saved experiment outputs
```

## Quickstart

### Install

With `uv` (recommended):

```bash
uv venv
uv pip install -e ".[dev,benchmarks]"
```

With `pip`:

```bash
python -m venv .venv
.venv/Scripts/activate   # Windows
pip install -e ".[dev,benchmarks]"
```

### Run Tests

```bash
pytest
```

### Run Benchmarks

```bash
python scripts/download_benchmarks.py    # fetch benchmark datasets
python scripts/run_hard_benchmark.py     # hard temporal benchmark (Loqi vs flat RAG)
python scripts/run_temporal_benchmark.py # temporal architecture benchmark
```

### Additional Documentation

- [BENCHMARKS.md](./BENCHMARKS.md) — benchmark research and dataset catalog
- [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md) — dependency and data licenses

## License

[MIT](./LICENSE) — Copyright 2026 Wyn Fox

## Short Version

Loqi is a research prototype that asks: *can a memory system become more useful over time by living alongside work, rather than only retrieving from a frozen corpus?*

The early answer is yes — in environments where semantic similarity is ambiguous and useful connections had to be formed before the query arrived, Loqi's temporal architecture produces measurable advantage over flat search.

That is enough to make the repo worth shipping.
