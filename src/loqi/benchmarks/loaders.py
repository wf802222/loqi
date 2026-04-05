"""Benchmark data loaders.

Each loader reads a raw benchmark file and yields normalized BenchmarkExample objects.
All loaders follow the same signature: load_X(path) -> list[BenchmarkExample].
"""

import json
from pathlib import Path

from loqi.benchmarks.schema import BenchmarkExample, Document, ReasoningStep


# ---------------------------------------------------------------------------
# MuSiQue
# ---------------------------------------------------------------------------

def load_musique(path: Path) -> list[BenchmarkExample]:
    """Load MuSiQue JSONL file into normalized examples.

    Each example has 20 paragraphs (some supporting, rest distractors)
    and a multi-hop question decomposition.
    """
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            documents = []
            for p in row["paragraphs"]:
                doc_id = f"musique_{row['id']}_p{p['idx']}"
                documents.append(Document(
                    id=doc_id,
                    title=p["title"],
                    text=p["paragraph_text"],
                    is_supporting=p["is_supporting"],
                ))

            supporting_ids = [
                f"musique_{row['id']}_p{p['idx']}"
                for p in row["paragraphs"]
                if p["is_supporting"]
            ]

            chain = []
            for step in row["question_decomposition"]:
                support_idx = step["paragraph_support_idx"]
                chain.append(ReasoningStep(
                    question=step["question"],
                    answer=step["answer"],
                    supporting_doc_id=(
                        f"musique_{row['id']}_p{support_idx}"
                        if support_idx is not None
                        else None
                    ),
                ))

            examples.append(BenchmarkExample(
                id=row["id"],
                benchmark="musique",
                query=row["question"],
                answer=row["answer"],
                answer_aliases=row.get("answer_aliases", []),
                documents=documents,
                supporting_doc_ids=supporting_ids,
                reasoning_chain=chain,
                category=row["id"].split("__")[0],
                metadata={"answerable": row.get("answerable", True)},
            ))

    return examples


# ---------------------------------------------------------------------------
# HotpotQA
# ---------------------------------------------------------------------------

def load_hotpotqa(path: Path) -> list[BenchmarkExample]:
    """Load HotpotQA JSONL file (distractor setting).

    HuggingFace serializes context as columnar:
      context = {title: [t1, t2, ...], sentences: [[s1, s2], [s3, s4], ...]}
    We zip these back into per-document objects.
    """
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ctx = row["context"]
            titles = ctx["title"]
            sentences_per_doc = ctx["sentences"]

            documents = []
            for i, (title, sents) in enumerate(zip(titles, sentences_per_doc)):
                documents.append(Document(
                    id=f"hotpot_{row['id']}_d{i}",
                    title=title,
                    text=" ".join(sents),
                ))

            # Supporting facts: {title: [t1, t2], sent_id: [0, 2]}
            sf = row["supporting_facts"]
            supporting_titles = set(sf["title"])

            # Mark documents as supporting and collect IDs
            supporting_ids = []
            final_docs = []
            for doc in documents:
                is_sup = doc.title in supporting_titles
                final_docs.append(Document(
                    id=doc.id,
                    title=doc.title,
                    text=doc.text,
                    is_supporting=is_sup,
                ))
                if is_sup:
                    supporting_ids.append(doc.id)

            examples.append(BenchmarkExample(
                id=row["id"],
                benchmark="hotpotqa",
                query=row["question"],
                answer=row["answer"],
                documents=final_docs,
                supporting_doc_ids=supporting_ids,
                category=row.get("type", ""),
                metadata={
                    "level": row.get("level", ""),
                    "supporting_facts_detail": sf,
                },
            ))

    return examples


# ---------------------------------------------------------------------------
# LongMemEval
# ---------------------------------------------------------------------------

def load_longmemeval(path: Path) -> list[BenchmarkExample]:
    """Load LongMemEval JSON file (oracle or small-scale).

    Each example contains chat sessions. Answer sessions contain turns with
    has_answer=True. We convert each session into a Document.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for row in data:
        documents = []
        supporting_ids = []
        answer_session_set = set(row["answer_session_ids"])

        for session_id, session_turns in zip(
            row["haystack_session_ids"], row["haystack_sessions"]
        ):
            # Flatten session turns into a single text
            text_parts = []
            for turn in session_turns:
                role = turn["role"]
                text_parts.append(f"[{role}]: {turn['content']}")

            is_answer = session_id in answer_session_set
            doc = Document(
                id=session_id,
                title=session_id,
                text="\n".join(text_parts),
                is_supporting=is_answer,
            )
            documents.append(doc)
            if is_answer:
                supporting_ids.append(session_id)

        examples.append(BenchmarkExample(
            id=row["question_id"],
            benchmark="longmemeval",
            query=row["question"],
            answer=row["answer"],
            documents=documents,
            supporting_doc_ids=supporting_ids,
            category=row["question_type"],
            metadata={
                "question_date": row.get("question_date", ""),
                "haystack_dates": row.get("haystack_dates", []),
            },
        ))

    return examples


# ---------------------------------------------------------------------------
# MemoryAgentBench
# ---------------------------------------------------------------------------

def load_memoryagentbench(path: Path) -> list[BenchmarkExample]:
    """Load a MemoryAgentBench JSONL split file.

    Each row contains a context (knowledge base) and multiple questions.
    We expand into one BenchmarkExample per question.
    """
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            source = row["metadata"]["source"]

            # The context is a single large text -- treat as one document
            context_doc = Document(
                id=f"mab_{source}_context",
                title=source,
                text=row["context"],
                is_supporting=True,
            )

            qa_pair_ids = row["metadata"].get("qa_pair_ids") or []
            question_types = row["metadata"].get("question_types") or []

            for i, (q, a) in enumerate(zip(row["questions"], row["answers"])):
                qa_id = qa_pair_ids[i] if i < len(qa_pair_ids) else f"{source}_q{i}"
                q_type = question_types[i] if i < len(question_types) else ""

                examples.append(BenchmarkExample(
                    id=qa_id,
                    benchmark="memoryagentbench",
                    query=q,
                    answer=a[0] if isinstance(a, list) and a else str(a),
                    answer_aliases=a if isinstance(a, list) else [str(a)],
                    documents=[context_doc],
                    supporting_doc_ids=[context_doc.id],
                    category=q_type,
                    metadata={
                        "source": source,
                        "split": path.stem,
                    },
                ))

    return examples


# ---------------------------------------------------------------------------
# Convenience: load all benchmarks
# ---------------------------------------------------------------------------

def load_all(data_dir: Path) -> dict[str, list[BenchmarkExample]]:
    """Load all available benchmarks from data/raw/."""
    raw = data_dir / "raw"
    result = {}

    musique_path = raw / "musique" / "validation.jsonl"
    if musique_path.exists():
        result["musique"] = load_musique(musique_path)

    hotpotqa_path = raw / "hotpotqa" / "validation.jsonl"
    if hotpotqa_path.exists():
        result["hotpotqa"] = load_hotpotqa(hotpotqa_path)

    longmemeval_path = raw / "longmemeval" / "longmemeval_oracle.json"
    if longmemeval_path.exists():
        result["longmemeval"] = load_longmemeval(longmemeval_path)

    # Load all MemoryAgentBench splits
    mab_dir = raw / "memoryagentbench"
    if mab_dir.exists():
        mab_examples = []
        for split_file in sorted(mab_dir.glob("*.jsonl")):
            mab_examples.extend(load_memoryagentbench(split_file))
        if mab_examples:
            result["memoryagentbench"] = mab_examples

    return result
