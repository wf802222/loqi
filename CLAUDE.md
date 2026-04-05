# Loqi

> Version: 2.5
> Last updated: 2026-04-05

## What this project does

Loqi is a continuous memory formation system for AI agents. It combines associative triggers, section-level graph retrieval, Hebbian learning, and local LLM gating into a self-improving memory architecture.

## Architecture

```
document arrives
  -> MemoryWriter splits into sections, creates nodes + edges
    -> Consolidator runs during downtime (decay, replay, bridges, trigger mining)
      -> SectionRetrieval queries via 3 channels (semantic + triggers + graph)
        -> SmolLM gate suppresses false triggers (optional)
          -> Episode logged for Hebbian learning
```

## Key components

- `src/loqi/graph/` — Node/Edge/Trigger models, SQLite store, embeddings, MemoryWriter
- `src/loqi/triggers/` — trigger extraction (markdown + conversational) and matching
- `src/loqi/retrieval/` — FlatRAG, GraphRAG, TriggerRAG, SectionRetrieval (v2)
- `src/loqi/hebbian/` — episode log, usefulness-gated updater, promoter, decay, consolidator
- `src/loqi/llm/` — Ollama client, SmolLM trigger gate, Phi-4 arbitrator
- `src/loqi/eval/` — metrics, protocol, evaluation runner
- `src/loqi/benchmarks/` — data loaders for MuSiQue, HotpotQA, LongMemEval, MemoryAgentBench

## Code conventions

- Language: Python 3.12+
- Package manager: uv
- Data models: Pydantic
- Graph storage: SQLite with adjacency tables
- Embedding model: all-MiniLM-L6-v2 (local, CPU)
- Tests: pytest (166 passing)

## Running

```bash
uv venv && uv pip install -e ".[dev,benchmarks]"
pytest
python scripts/download_benchmarks.py
python scripts/run_hard_benchmark.py
```
