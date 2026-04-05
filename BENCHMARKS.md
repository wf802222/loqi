# Loqi -- Benchmark & Dataset Research

> Last updated: 2026-04-04
> Purpose: Identify publicly available benchmarks and datasets for evaluating Loqi against baselines (flat RAG, GraphRAG, etc.)

---

## Summary

This document catalogs benchmarks and datasets relevant to evaluating Loqi's three differentiating claims:
1. **Associative Triggers** -- pre-retrieval pattern matching that surfaces standing instructions
2. **Focused/Diffuse Dual-Mode Retrieval** -- graph traversal that connects information across multiple documents
3. **Hebbian Edge Strengthening** -- co-activation learning that promotes frequently useful connections

No single existing benchmark tests all three. A Loqi evaluation suite will likely combine datasets from multiple categories below.

---

## 1. Standard RAG Benchmarks

These are the workhorses used by most RAG papers. Good for establishing baseline comparisons.

### 1.1 CRAG (Comprehensive RAG Benchmark)
- **What it tests:** Factual QA across 5 domains, 8 question types, varied entity popularity (popular to long-tail), temporal dynamism (years to seconds)
- **Size:** 4,409 question-answer pairs + mock web/KG search APIs
- **Origin:** Meta (Facebook Research), KDD Cup 2024 challenge, NeurIPS 2024 Datasets Track
- **Where to get it:**
  - HuggingFace: `Quivr/CRAG`
  - GitHub: https://github.com/facebookresearch/CRAG/
- **Paper:** https://arxiv.org/abs/2406.04744
- **Loqi relevance:** MEDIUM. Good baseline comparison for factual retrieval quality, but does not test triggers, graph traversal, or memory persistence.

### 1.2 RAGBench
- **What it tests:** End-to-end RAG across industry-specific domains with explainable metrics (TRACe: uTilization, Relevance, Adherence, Completeness)
- **Size:** 100,000 examples across train/val/test splits, 5 industry domains
- **Origin:** Galileo AI, 2024
- **Where to get it:**
  - HuggingFace: `galileo-ai/ragbench`
  - GitHub: https://github.com/rungalileo/ragbench
- **Paper:** https://arxiv.org/abs/2407.11005
- **Loqi relevance:** LOW-MEDIUM. Large-scale baseline comparison, but tests standard single-hop retrieval. Useful for proving Loqi doesn't regress on simple cases.

### 1.3 RGB (Retrieval-Augmented Generation Benchmark)
- **What it tests:** Four fundamental RAG abilities: noise robustness, negative rejection, information integration, counterfactual robustness. Bilingual (English + Chinese).
- **Size:** ~600 questions (English + Chinese combined)
- **Origin:** Chinese Academy of Sciences, AAAI 2024
- **Where to get it:**
  - GitHub: https://github.com/chen700564/RGB
- **Paper:** https://arxiv.org/abs/2309.01431
- **Loqi relevance:** MEDIUM-HIGH. The "information integration" dimension (synthesizing answers from multiple documents) maps well to Loqi's diffuse retrieval. The "counterfactual robustness" dimension is relevant to trigger confidence decay.

### 1.4 FRAMES (Factuality, Retrieval, And reasoning MEasurement Set)
- **What it tests:** Multi-step factual reasoning requiring integration of 2-15 Wikipedia articles per question. Tests factuality, retrieval accuracy, and reasoning.
- **Size:** 824 multi-hop questions
- **Origin:** Google + Harvard, 2024. Published at NAACL 2025.
- **Where to get it:**
  - HuggingFace: `google/frames-benchmark` (12.6K downloads)
- **Paper:** https://arxiv.org/abs/2409.12941
- **Loqi relevance:** HIGH. Multi-hop questions requiring 2-15 sources directly tests whether graph traversal (focused + diffuse) outperforms flat retrieval. 36% of questions involve multi-constraint reasoning.

### 1.5 CRUD-RAG
- **What it tests:** Four RAG application modes: Create (generate original content), Read (knowledge-intensive QA), Update (correct inaccuracies), Delete (summarize/compress). Chinese-focused.
- **Size:** Large-scale across four task types (exact counts TBD)
- **Origin:** 2024
- **Where to get it:**
  - GitHub: referenced in paper
- **Paper:** https://arxiv.org/abs/2401.17043 (HuggingFace paper page)
- **Loqi relevance:** MEDIUM. The "Update" task (correcting outdated information) maps to Hebbian decay. The "Create" task tests generative quality from retrieval.

---

## 2. Multi-Hop Reasoning Benchmarks

Critical for testing Loqi's graph traversal and cross-document reasoning claims.

### 2.1 HotpotQA
- **What it tests:** Multi-hop QA requiring reasoning over 2 Wikipedia paragraphs. Provides supporting facts for explainability.
- **Size:** 113K question-answer pairs (train: 90K, dev: 7.4K, test: 7.4K)
- **Origin:** Carnegie Mellon + Stanford, 2018
- **Where to get it:**
  - HuggingFace: `hotpotqa/hotpot_qa` (72.4K downloads, most popular)
  - Also in MTEB and BEIR collections
- **Paper:** https://arxiv.org/abs/1809.09600
- **Loqi relevance:** HIGH. The canonical multi-hop benchmark. Used by HippoRAG, GraphRAG, and nearly every graph-based retrieval paper. Essential baseline.

### 2.2 MuSiQue (Multihop Questions via Single-hop Question Composition)
- **What it tests:** 2-4 hop questions intentionally designed to resist shortcut reasoning. More difficult than HotpotQA (3x increase in human-machine gap). Includes unanswerable contrast questions.
- **Size:** ~25K questions (answerable: MuSiQue-Ans; full with unanswerable: MuSiQue-Full). HF version has ~22K rows.
- **Origin:** UNC Chapel Hill, TACL 2022
- **Where to get it:**
  - HuggingFace: `dgslibisey/MuSiQue` (5.7K downloads, 271 MB) or `bdsaglam/musique` (3.6K downloads, 808 MB)
  - Original: https://github.com/StonyBrookNLP/musique
- **Paper:** https://arxiv.org/abs/2108.00573
- **Loqi relevance:** VERY HIGH. The hardest standard multi-hop benchmark. If Loqi's graph traversal can outperform flat RAG here, it's a strong signal. HippoRAG uses this as a primary eval.

### 2.3 2WikiMultiHopQA
- **What it tests:** Multi-hop QA with comprehensive reasoning chain explanations, using logical rules from Wikidata to create natural multi-hop questions.
- **Size:** 192K+ total (train: 167K, dev: 12.5K, test: 12.5K). Based on 5.95M Wikidata entities.
- **Origin:** ALAB-NII, COLING 2020
- **Where to get it:**
  - HuggingFace: `xanhho/2WikiMultihopQA`
  - GitHub: https://github.com/Alab-NII/2wikimultihop
- **Paper:** https://aclanthology.org/2020.coling-main.580/
- **Loqi relevance:** HIGH. Large scale and structured reasoning chains make it ideal for testing whether Hebbian strengthening improves retrieval paths over time.

### 2.4 MultiHop-RAG
- **What it tests:** Multi-hop queries specifically designed for RAG systems. Evidence distributed across 2-4 news articles per query. Includes document metadata.
- **Size:** 2,556 queries over a news article knowledge base
- **Origin:** COLM 2024
- **Where to get it:**
  - HuggingFace: `yixuantt/MultiHopRAG`
  - GitHub: https://github.com/yixuantt/MultiHop-RAG
- **Paper:** https://arxiv.org/abs/2401.15391
- **Loqi relevance:** HIGH. Purpose-built for RAG multi-hop evaluation. Smaller but more targeted than HotpotQA.

### 2.5 MINTQA
- **What it tests:** Multi-hop reasoning on new and long-tail knowledge -- tests whether systems can handle information not well-represented in training data.
- **Size:** Dataset details in paper
- **Origin:** 2024
- **Paper:** https://arxiv.org/abs/2412.17032
- **Loqi relevance:** MEDIUM-HIGH. Long-tail knowledge is relevant to Loqi's trigger system, which should surface less popular but pattern-matched information.

---

## 3. Knowledge Graph / GraphRAG Benchmarks

Directly comparable to Loqi's graph-based architecture.

### 3.1 Microsoft GraphRAG Benchmarking Datasets
- **What it tests:** Open-ended and thematic questions over real document corpora. Tests comprehensiveness, diversity, empowerment, and relevance of answers.
- **Datasets included:**
  - HotPotQA subset
  - Kevin Scott Podcasts (125 thematic questions from tech leader interviews)
  - AP Health News (1,397 health news articles)
  - Behind the Tech podcast transcripts (70 episodes)
- **Where to get it:**
  - GitHub: https://github.com/microsoft/graphrag-benchmarking-datasets
  - BenchmarkQED tool: https://github.com/microsoft/benchmark-qed
- **Loqi relevance:** HIGH. Direct comparison target. These are the exact datasets Microsoft uses to evaluate their GraphRAG system.

### 3.2 GraphRAG-Bench (ICLR 2026)
- **What it tests:** Domain-specific reasoning across 16 academic disciplines. College-level questions requiring multi-hop reasoning beyond simple content retrieval. Task types: fact retrieval, complex reasoning, contextual summarization, creative generation.
- **Size:** 1,018 college-level questions across 20 core textbooks
- **Origin:** "When to use Graphs in RAG" (ICLR 2026)
- **Where to get it:**
  - HuggingFace: `GraphRAG-Bench/GraphRAG-Bench`
  - GitHub: https://github.com/GraphRAG-Bench/GraphRAG-Benchmark
- **Paper:** https://arxiv.org/abs/2506.05690
- **Loqi relevance:** HIGH. Purpose-built for evaluating graph-based RAG. Tests whether graph structure actually helps vs. flat retrieval.

### 3.3 BEIR (Benchmarking Information Retrieval)
- **What it tests:** Zero-shot retrieval quality across 17+ heterogeneous datasets spanning QA, fact-checking, biomedical retrieval, and more. Tests embedding/retrieval model generalization.
- **Size:** 17+ datasets, millions of documents total
- **Origin:** 2021, widely adopted standard
- **Where to get it:**
  - HuggingFace: `BeIR/*` namespace (multiple datasets)
  - GitHub: https://github.com/beir-cellar/beir
- **Paper:** https://arxiv.org/abs/2104.08663
- **Loqi relevance:** MEDIUM. Tests the retrieval layer only (no generation). Useful for evaluating Loqi's embedding + graph traversal retrieval quality independent of the LLM.

### 3.4 MTEB (Massive Text Embedding Benchmark)
- **What it tests:** Embedding quality across 8 tasks (retrieval, clustering, classification, reranking, etc.) on 58+ datasets in 112 languages.
- **Size:** 58+ datasets (includes HotpotQA, NQ, and others from BEIR)
- **Origin:** HuggingFace, 2022
- **Where to get it:**
  - HuggingFace: `mteb/*` namespace
  - GitHub: https://github.com/embeddings-benchmark/mteb
  - Leaderboard: https://huggingface.co/spaces/mteb/leaderboard
- **Paper:** https://arxiv.org/abs/2210.07316
- **Loqi relevance:** LOW-MEDIUM. Useful for selecting Loqi's embedding model (nomic-embed vs. alternatives), but not for evaluating the full system.

---

## 4. Long-Term Memory Benchmarks

Most relevant to Loqi's core differentiator: remembering standing instructions and preferences over time.

### 4.1 LongMemEval (ICLR 2025) -- TOP PRIORITY
- **What it tests:** Five core long-term memory abilities: Information Extraction, Multi-Session Reasoning, Temporal Reasoning, Knowledge Updates, and Abstention. Tests implicit user preference extraction.
- **Size:** 500 curated questions embedded within scalable chat histories
- **Origin:** UCLA, ICLR 2025
- **Where to get it:**
  - HuggingFace: dataset released on HF (see GitHub for link)
  - GitHub: https://github.com/xiaowu0162/LongMemEval
- **Paper:** https://arxiv.org/abs/2410.10813
- **Loqi relevance:** VERY HIGH. Tests exactly what Loqi's trigger system is designed for: surfacing standing instructions, preferences, and knowledge updates across sessions. The "Knowledge Updates" dimension tests Hebbian-like learning. Commercial chat assistants show 30% accuracy drops here.

### 4.2 MemoryAgentBench (ICLR 2026)
- **What it tests:** Four core competencies for memory agents: Accurate Retrieval, Test-Time Learning, Long-Range Understanding, and Selective Forgetting. Multi-turn format simulating incremental information processing.
- **Size:** Multiple datasets including two novel ones (EventQA, FactConsolidation)
- **Origin:** Huazhong University of Science and Technology, ICLR 2026
- **Where to get it:**
  - HuggingFace: `ai-hyz/MemoryAgentBench`
  - GitHub: https://github.com/HUST-AI-HYZ/MemoryAgentBench
- **Paper:** https://arxiv.org/abs/2507.05257
- **Loqi relevance:** VERY HIGH. "Selective Forgetting" maps directly to Hebbian decay. "Test-Time Learning" maps to Hebbian strengthening. "Accurate Retrieval" maps to trigger + focused retrieval. Current methods fail to master all four competencies -- this is exactly where Loqi claims to excel.

### 4.3 LoCoMo (Long-Term Conversational Memory)
- **What it tests:** Very long-term conversational memory: QA, event summarization, and multi-modal dialogue generation over conversations spanning 300 turns, 9K tokens avg, up to 35 sessions.
- **Size:** 10 long conversations with comprehensive annotations (subset of original 50)
- **Origin:** Snap Research, ACL 2024
- **Where to get it:**
  - GitHub: https://github.com/snap-research/locomo (data in `./data/locomo10.json`)
  - Project page: https://snap-research.github.io/locomo/
- **Paper:** https://arxiv.org/abs/2402.17753
- **Loqi relevance:** HIGH. Tests memory over extended multi-session interactions. Small dataset but high quality. Good for qualitative evaluation of trigger persistence.

### 4.4 MemoryBench
- **What it tests:** Memory and continual learning across 11 datasets spanning open-domain, legal, and academic data. Tests both declarative and procedural memory with interactive feedback cycles.
- **Size:** 20,000 cases across 11 public benchmarks
- **Origin:** 2025
- **Where to get it:**
  - Paper/GitHub: https://arxiv.org/abs/2510.17281
- **Paper:** https://arxiv.org/abs/2510.17281
- **Loqi relevance:** HIGH. Tests whether systems learn from feedback (procedural memory) without forgetting -- directly tests the Hebbian learning loop.

### 4.5 AMA-Bench (Agent Memory with Any Length)
- **What it tests:** Long-horizon memory for agentic applications using real-world agentic trajectories (not just dialogue). Tests causality reasoning and objective information retention.
- **Size:** Real-world trajectories with expert-curated QA + synthetic scalable trajectories
- **Origin:** 2026 (February)
- **Where to get it:**
  - Paper: https://arxiv.org/abs/2602.22769
- **Loqi relevance:** HIGH. Tests agent memory on machine-generated interaction streams (not just human dialogue). Existing memory systems fail because they lack causality and use lossy similarity retrieval -- Loqi's trigger + graph approach directly addresses these weaknesses.

---

## 5. Long-Document / Reading Comprehension Benchmarks

Used by RAPTOR and similar systems. Tests ability to synthesize across long documents.

### 5.1 NarrativeQA
- **What it tests:** Reading comprehension requiring understanding of entire books or movie scripts. Questions based on summaries, requiring narrative comprehension (not shallow fact recall).
- **Size:** ~47K QA pairs over ~1,500 documents
- **Origin:** DeepMind, 2017
- **Where to get it:**
  - HuggingFace: `deepmind/narrativeqa` (7.5K downloads)
- **Paper:** https://arxiv.org/abs/1712.07040
- **Loqi relevance:** MEDIUM. Tests long-range comprehension. Relevant to diffuse retrieval's ability to find non-obvious connections.

### 5.2 QuALITY
- **What it tests:** Multiple-choice QA on long passages (~5,000 tokens avg). Questions are written by annotators who read the full text. Only half answerable by skimming.
- **Size:** Multiple-choice questions over long English passages
- **Origin:** NYU, 2021
- **Where to get it:**
  - GitHub: https://github.com/nyu-mll/quality
  - Project page: https://nyu-mll.github.io/quality/
- **Paper:** https://arxiv.org/abs/2112.08608
- **Loqi relevance:** MEDIUM. Tests deep comprehension vs. shallow retrieval -- relevant to focused vs. diffuse distinction.

### 5.3 QASPER
- **What it tests:** QA on NLP research papers. Questions from abstracts, answers from full papers. Evaluates both answer accuracy (Answer-F1) and evidence selection (Evidence-F1).
- **Size:** 5,049 questions over 1,585 NLP papers
- **Origin:** Allen AI
- **Where to get it:**
  - HuggingFace: `allenai/qasper`
- **Paper:** referenced in RAPTOR and multiple RAG papers
- **Loqi relevance:** MEDIUM. Tests retrieval + reasoning over technical documents. Could test whether Loqi's graph captures relationships between concepts in scientific papers.

### 5.4 LV-Eval
- **What it tests:** Long-context evaluation at 5 length levels (16K-256K words). Single-hop and multi-hop QA with confusing facts insertion to prevent knowledge leakage. Bilingual.
- **Size:** 11 datasets, contexts averaging ~102K words
- **Origin:** 2024
- **Where to get it:**
  - GitHub: https://github.com/infinigence/LVEval
- **Paper:** https://arxiv.org/abs/2402.05136
- **Loqi relevance:** MEDIUM. Tests performance at extreme context lengths. Relevant to showing that Loqi's graph structure scales better than brute-force long context.

---

## 6. Single-Hop Factual QA (Baseline / Regression Tests)

Important for proving Loqi doesn't sacrifice simple retrieval quality.

### 6.1 Natural Questions (NQ)
- **What it tests:** Real Google search queries with Wikipedia answers. Single-hop factual QA.
- **Size:** 307K+ training examples, ~7.8K dev, ~7.8K test. Download: ~45 GB.
- **Origin:** Google, 2019
- **Where to get it:**
  - HuggingFace: `google-research-datasets/natural_questions`
- **Loqi relevance:** LOW (baseline only). Used by HippoRAG to show it doesn't regress on simple queries.

### 6.2 PopQA
- **What it tests:** Entity-centric QA with popularity annotations (from Wikipedia page views). Tests whether systems handle popular vs. long-tail entities.
- **Size:** 14K entity-centric QA pairs
- **Origin:** 2023
- **Where to get it:**
  - HuggingFace: `akariasai/PopQA`
- **Loqi relevance:** LOW-MEDIUM. The popularity dimension is interesting -- Loqi's trigger system should perform well on long-tail entities that flat RAG misses.

---

## 7. RAG Evaluation Frameworks (Tools, Not Datasets)

### 7.1 RAGAS
- Automated evaluation metrics for RAG: answer relevancy, faithfulness, context precision, context recall.
- GitHub: https://github.com/explodinggradients/ragas
- Paper: https://arxiv.org/abs/2309.15217

### 7.2 BenchmarkQED (Microsoft)
- Automated benchmarking of RAG systems. Complements GraphRAG.
- GitHub: https://github.com/microsoft/benchmark-qed

---

## Recommended Evaluation Strategy for Loqi

### Tier 1: Must-Have (Core Claims)
| Benchmark | Tests | Loqi Claim Tested |
|-----------|-------|-------------------|
| MuSiQue | Hard multi-hop (2-4 hops) | Focused/Diffuse retrieval |
| HotpotQA | Standard multi-hop (2 hops) | Focused/Diffuse retrieval |
| LongMemEval | Standing instructions, preferences, knowledge updates | Associative Triggers |
| MemoryAgentBench | Selective forgetting, test-time learning | Hebbian Learning |
| FRAMES | Multi-source integration (2-15 articles) | All three layers |

### Tier 2: Strong Comparisons
| Benchmark | Tests | Loqi Claim Tested |
|-----------|-------|-------------------|
| GraphRAG-Bench | Graph vs. flat RAG | Graph traversal value |
| MS GraphRAG datasets | Direct GraphRAG comparison | Architecture comparison |
| 2WikiMultiHopQA | Large-scale multi-hop | Hebbian learning over time |
| MemoryBench | Continual learning from feedback | Hebbian loop |

### Tier 3: Baseline / Regression
| Benchmark | Tests | Purpose |
|-----------|-------|---------|
| Natural Questions | Single-hop factual | Prove no regression |
| PopQA | Entity QA (popular + long-tail) | Trigger value for long-tail |
| CRAG | Comprehensive factual QA | Broad baseline |

### Custom Benchmark Needed
None of the existing benchmarks specifically tests the combination of:
- Write-time trigger creation from standing instructions
- Pre-retrieval trigger firing on context match
- Edge strengthening from repeated co-activation
- Promotion from diffuse hit to trigger

Recommendation: Create a **Loqi-specific micro-benchmark** (~100-200 examples) that tests standing instruction persistence across sessions, with ground truth for which instructions should fire given which contexts. This would complement the standardized benchmarks above.

---

## Key Papers Using These Benchmarks

| Paper | Year | Venue | Benchmarks Used |
|-------|------|-------|-----------------|
| HippoRAG | 2024 | NeurIPS | MuSiQue, 2WikiMultiHopQA, HotpotQA, PopQA, NQ |
| GraphRAG (Microsoft) | 2024 | -- | Custom podcast/news datasets |
| RAPTOR | 2024 | ICLR | QuALITY, QASPER, NarrativeQA |
| NodeRAG | 2025 | -- | Multi-hop benchmarks + open-ended evaluations |
| HippoRAG 2 | 2025 | -- | MuSiQue, 2WikiMultiHopQA, HotpotQA, NarrativeQA, NQ, PopQA, LV-Eval |

---

## Sources

- [Microsoft GraphRAG Benchmarking Datasets](https://github.com/microsoft/graphrag-benchmarking-datasets)
- [BenchmarkQED - Microsoft Research](https://www.microsoft.com/en-us/research/blog/benchmarkqed-automated-benchmarking-of-rag-systems/)
- [GraphRAG-Bench (ICLR 2026)](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark)
- [HippoRAG - OSU NLP Group](https://github.com/osu-nlp-group/hipporag)
- [MultiHop-RAG (COLM 2024)](https://github.com/yixuantt/MultiHop-RAG)
- [FRAMES Benchmark - Google](https://huggingface.co/datasets/google/frames-benchmark)
- [RAGBench - Galileo AI](https://huggingface.co/datasets/galileo-ai/ragbench)
- [CRAG - Facebook Research](https://github.com/facebookresearch/CRAG/)
- [LongMemEval (ICLR 2025)](https://github.com/xiaowu0162/LongMemEval)
- [MemoryAgentBench (ICLR 2026)](https://github.com/HUST-AI-HYZ/MemoryAgentBench)
- [LoCoMo - Snap Research](https://github.com/snap-research/locomo)
- [MemoryBench](https://arxiv.org/abs/2510.17281)
- [AMA-Bench](https://arxiv.org/abs/2602.22769)
- [BEIR Benchmark](https://github.com/beir-cellar/beir)
- [MTEB Benchmark](https://github.com/embeddings-benchmark/mteb)
- [RGB Benchmark](https://github.com/chen700564/RGB)
- [RAPTOR Paper](https://arxiv.org/abs/2401.18059)
- [Evaluation of RAG: A Survey](https://arxiv.org/abs/2405.07437)
- [RAG Evaluation Survey 2025](https://arxiv.org/abs/2504.14891)
