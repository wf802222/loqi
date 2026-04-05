# Third-Party Notices

Loqi uses the following third-party software and data. We gratefully
acknowledge the work of these projects and comply with their licenses.

## Software Dependencies

### sentence-transformers
- License: Apache-2.0
- Source: https://github.com/UKPLab/sentence-transformers
- Used for: text embedding (all-MiniLM-L6-v2 model)

### all-MiniLM-L6-v2 (embedding model)
- License: Apache-2.0
- Source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Used for: converting text to vector embeddings for similarity search

### PyTorch
- License: BSD-3-Clause
- Source: https://github.com/pytorch/pytorch
- Used for: tensor operations (via sentence-transformers)

### NumPy
- License: BSD-3-Clause
- Source: https://github.com/numpy/numpy
- Used for: array operations, cosine similarity

### Pydantic
- License: MIT
- Source: https://github.com/pydantic/pydantic
- Used for: data model validation

### PyYAML
- License: MIT
- Source: https://github.com/yaml/pyyaml
- Used for: loading benchmark scenario files

### HuggingFace Datasets
- License: Apache-2.0
- Source: https://github.com/huggingface/datasets
- Used for: downloading benchmark datasets

### HuggingFace Transformers
- License: Apache-2.0
- Source: https://github.com/huggingface/transformers
- Used for: model loading (via sentence-transformers)

## Benchmark Datasets

These datasets are downloaded by `scripts/download_benchmarks.py` and
are NOT distributed with this repository. Users download them directly
from their original sources.

### MuSiQue
- License: CC BY-SA 4.0
- Paper: Trivedi et al., "MuSiQue: Multihop Questions via Single Hop Question Composition" (TACL 2022)
- Source: https://github.com/StonyBrookNLP/musique
- HuggingFace: https://huggingface.co/datasets/bdsaglam/musique
- Used for: multi-hop retrieval evaluation

### HotpotQA
- License: CC BY-SA 4.0
- Paper: Yang et al., "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering" (EMNLP 2018)
- Source: https://hotpotqa.github.io/
- HuggingFace: https://huggingface.co/datasets/hotpotqa/hotpot_qa
- Used for: multi-hop retrieval evaluation

### LongMemEval
- License: MIT
- Paper: Wu et al., "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory" (ICLR 2025)
- Source: https://github.com/xiaowu0162/LongMemEval
- HuggingFace: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
- Used for: long-term memory and preference recall evaluation

### MemoryAgentBench
- License: MIT
- Paper: He et al., "MemoryAgentBench: Benchmarking LLM-based Agentic Memory Systems" (ICLR 2026)
- Source: https://github.com/HUST-AI-HYZ/MemoryAgentBench
- HuggingFace: https://huggingface.co/datasets/ai-hyz/MemoryAgentBench
- Used for: memory agent competency evaluation

## Acknowledged Inspiration

The self-learning and workflow discipline aspects of this project were
informed by ideas from:

- **gstack** by Garry Tan — workflow discipline, review agents, learnings persistence
  (https://github.com/garrytan/gstack)
- **HippoRAG** — graph memory with relevance propagation
- **Microsoft GraphRAG** — community structure and local/global retrieval
- **Barbara Oakley's "A Mind for Numbers"** — focused/diffuse thinking modes
  that inspired the dual retrieval pass design
