"""Tests for the evaluation runner.

Uses a mock retrieval system to verify the harness orchestration
and metric computation work end-to-end.
"""

from loqi.benchmarks.schema import BenchmarkExample, Document
from loqi.eval.protocol import RetrievalResult, RetrievalSystem
from loqi.eval.runner import BenchmarkResult, evaluate_retrieval


class PerfectSystem(RetrievalSystem):
    """A mock system that always returns the supporting documents."""

    @property
    def name(self) -> str:
        return "perfect-oracle"

    def index(self, documents: list[Document]) -> None:
        self._docs = documents

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        supporting = [d for d in self._docs if d.is_supporting][:top_k]
        return RetrievalResult(
            retrieved_ids=[d.id for d in supporting],
            retrieved_docs=supporting,
        )


class RandomSystem(RetrievalSystem):
    """A mock system that returns documents in index order (no intelligence)."""

    @property
    def name(self) -> str:
        return "random-baseline"

    def index(self, documents: list[Document]) -> None:
        self._docs = documents

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        docs = self._docs[:top_k]
        return RetrievalResult(
            retrieved_ids=[d.id for d in docs],
            retrieved_docs=docs,
        )


def _make_example(num_docs=10, num_supporting=2) -> BenchmarkExample:
    """Create a synthetic example for testing."""
    docs = []
    supporting_ids = []
    for i in range(num_docs):
        is_sup = i < num_supporting
        doc = Document(
            id=f"doc_{i}",
            title=f"Document {i}",
            text=f"Content of document {i}",
            is_supporting=is_sup,
        )
        docs.append(doc)
        if is_sup:
            supporting_ids.append(doc.id)

    return BenchmarkExample(
        id="test_001",
        benchmark="test",
        query="What is the answer?",
        answer="42",
        documents=docs,
        supporting_doc_ids=supporting_ids,
    )


class TestEvaluateRetrieval:
    def test_perfect_system_gets_perfect_scores(self):
        examples = [_make_example()]
        result = evaluate_retrieval(PerfectSystem(), examples, top_k=10)

        assert isinstance(result, BenchmarkResult)
        assert result.system_name == "perfect-oracle"
        assert result.benchmark_name == "test"
        assert len(result.example_results) == 1

        metrics = result.example_results[0].metrics
        assert metrics["support_f1"] == 1.0
        assert metrics["recall@1"] == 0.5  # 2 gold, top-1 gets 1
        assert metrics["recall@3"] == 1.0  # top-3 gets both supporting docs

    def test_random_system_scores_lower(self):
        # Supporting docs are at index 0,1 -- random returns 0..9 in order
        # So random actually gets both supporting docs in top-2 here
        examples = [_make_example()]
        result = evaluate_retrieval(RandomSystem(), examples, top_k=10)

        metrics = result.example_results[0].metrics
        # Random returns all 10 docs, gold is 2 -> support_f1 < 1.0
        # because precision is 2/10
        assert metrics["support_f1"] < 1.0

    def test_multiple_examples(self):
        examples = [_make_example() for _ in range(5)]
        result = evaluate_retrieval(PerfectSystem(), examples, top_k=10)

        assert len(result.example_results) == 5
        assert "support_f1" in result.aggregate
        assert result.aggregate["support_f1"]["count"] == 5
        assert result.aggregate["support_f1"]["mean"] == 1.0

    def test_latency_is_tracked(self):
        examples = [_make_example()]
        result = evaluate_retrieval(PerfectSystem(), examples, top_k=10)

        assert result.example_results[0].latency_ms >= 0
        assert result.total_time_s >= 0

    def test_summary_output(self):
        examples = [_make_example()]
        result = evaluate_retrieval(PerfectSystem(), examples, top_k=10)

        summary = result.summary()
        assert "perfect-oracle" in summary
        assert "support_f1" in summary

    def test_reset_between_examples(self):
        """Verify reset is called between examples when reset_between=True."""

        class TrackingSystem(PerfectSystem):
            def __init__(self):
                self.reset_count = 0

            def reset(self):
                self.reset_count += 1

        system = TrackingSystem()
        examples = [_make_example() for _ in range(3)]
        evaluate_retrieval(system, examples, reset_between=True)
        assert system.reset_count == 3

    def test_no_reset_for_learning(self):
        """Verify reset is NOT called when reset_between=False (Hebbian tests)."""

        class TrackingSystem(PerfectSystem):
            def __init__(self):
                self.reset_count = 0

            def reset(self):
                self.reset_count += 1

        system = TrackingSystem()
        examples = [_make_example() for _ in range(3)]
        evaluate_retrieval(system, examples, reset_between=False)
        assert system.reset_count == 0
