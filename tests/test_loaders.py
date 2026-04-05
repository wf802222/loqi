"""Tests for benchmark data loaders.

These tests run against the actual downloaded data in data/raw/.
Skip gracefully if data hasn't been downloaded yet.
"""

from pathlib import Path

import pytest

from loqi.benchmarks.loaders import (
    load_hotpotqa,
    load_longmemeval,
    load_memoryagentbench,
    load_musique,
)
from loqi.benchmarks.schema import BenchmarkExample

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


# ---------------------------------------------------------------------------
# MuSiQue
# ---------------------------------------------------------------------------

MUSIQUE_PATH = DATA_DIR / "musique" / "validation.jsonl"


@pytest.mark.skipif(not MUSIQUE_PATH.exists(), reason="MuSiQue data not downloaded")
class TestMuSiQue:
    @pytest.fixture(scope="class")
    def examples(self):
        return load_musique(MUSIQUE_PATH)

    def test_loads_examples(self, examples):
        assert len(examples) > 0

    def test_example_is_benchmark_example(self, examples):
        assert isinstance(examples[0], BenchmarkExample)

    def test_benchmark_name(self, examples):
        assert all(e.benchmark == "musique" for e in examples)

    def test_has_paragraphs(self, examples):
        # Most have 20 but some unanswerable variants have fewer (15-19)
        assert all(15 <= len(e.documents) <= 20 for e in examples)

    def test_answerable_have_supporting_docs(self, examples):
        # Only answerable examples have supporting docs; unanswerable have 0
        for e in examples[:100]:
            if e.metadata.get("answerable", True):
                supporting = [d for d in e.documents if d.is_supporting]
                assert len(supporting) >= 1, f"{e.id} answerable but 0 supporting docs"

    def test_reasoning_chain_matches_hops(self, examples):
        for e in examples[:50]:
            hop_count = int(e.category[0])  # "2hop" -> 2, "3hop1" -> 3
            assert len(e.reasoning_chain) == hop_count, (
                f"{e.id}: category={e.category}, chain={len(e.reasoning_chain)}"
            )

    def test_supporting_ids_match_documents(self, examples):
        for e in examples[:50]:
            doc_ids = {d.id for d in e.documents}
            for sid in e.supporting_doc_ids:
                assert sid in doc_ids, f"{e.id}: supporting ID {sid} not in documents"


# ---------------------------------------------------------------------------
# HotpotQA
# ---------------------------------------------------------------------------

HOTPOTQA_PATH = DATA_DIR / "hotpotqa" / "validation.jsonl"


@pytest.mark.skipif(not HOTPOTQA_PATH.exists(), reason="HotpotQA data not downloaded")
class TestHotpotQA:
    @pytest.fixture(scope="class")
    def examples(self):
        return load_hotpotqa(HOTPOTQA_PATH)

    def test_loads_examples(self, examples):
        assert len(examples) > 0

    def test_benchmark_name(self, examples):
        assert all(e.benchmark == "hotpotqa" for e in examples)

    def test_has_documents(self, examples):
        # Most have 10 but a small fraction have fewer (2-9)
        assert all(2 <= len(e.documents) <= 10 for e in examples)

    def test_has_supporting_docs(self, examples):
        for e in examples[:50]:
            supporting = [d for d in e.documents if d.is_supporting]
            assert len(supporting) >= 2, f"{e.id} has {len(supporting)} supporting docs"

    def test_documents_have_text(self, examples):
        for e in examples[:50]:
            for d in e.documents:
                assert len(d.text) > 0, f"{e.id}: doc {d.id} has empty text"


# ---------------------------------------------------------------------------
# LongMemEval
# ---------------------------------------------------------------------------

LONGMEMEVAL_PATH = DATA_DIR / "longmemeval" / "longmemeval_oracle.json"


@pytest.mark.skipif(not LONGMEMEVAL_PATH.exists(), reason="LongMemEval data not downloaded")
class TestLongMemEval:
    @pytest.fixture(scope="class")
    def examples(self):
        return load_longmemeval(LONGMEMEVAL_PATH)

    def test_loads_500_examples(self, examples):
        assert len(examples) == 500

    def test_benchmark_name(self, examples):
        assert all(e.benchmark == "longmemeval" for e in examples)

    def test_has_question_types(self, examples):
        types = {e.category for e in examples}
        assert "single-session-user" in types
        assert "temporal-reasoning" in types

    def test_has_supporting_sessions(self, examples):
        for e in examples[:50]:
            assert len(e.supporting_doc_ids) >= 1, f"{e.id} has no supporting sessions"

    def test_documents_contain_text(self, examples):
        for e in examples[:20]:
            for d in e.documents:
                assert len(d.text) > 0


# ---------------------------------------------------------------------------
# MemoryAgentBench
# ---------------------------------------------------------------------------

MAB_PATH = DATA_DIR / "memoryagentbench" / "Conflict_Resolution.jsonl"


@pytest.mark.skipif(not MAB_PATH.exists(), reason="MemoryAgentBench data not downloaded")
class TestMemoryAgentBench:
    @pytest.fixture(scope="class")
    def examples(self):
        return load_memoryagentbench(MAB_PATH)

    def test_loads_examples(self, examples):
        assert len(examples) > 0

    def test_benchmark_name(self, examples):
        assert all(e.benchmark == "memoryagentbench" for e in examples)

    def test_has_answer(self, examples):
        for e in examples[:20]:
            assert len(e.answer) > 0, f"{e.id} has empty answer"

    def test_has_context_document(self, examples):
        for e in examples[:20]:
            assert len(e.documents) >= 1
            assert len(e.documents[0].text) > 0
