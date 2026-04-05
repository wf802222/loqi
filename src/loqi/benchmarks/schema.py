"""Common schema for all benchmark examples.

Every benchmark loader normalizes its raw format into these dataclasses.
The evaluation harness operates entirely on this schema — it never sees
raw benchmark formats directly.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """A single document/paragraph in the corpus."""

    id: str
    title: str
    text: str
    is_supporting: bool = False


@dataclass(frozen=True)
class ReasoningStep:
    """A single hop in a multi-hop reasoning chain."""

    question: str
    answer: str
    supporting_doc_id: str | None = None


@dataclass(frozen=True)
class BenchmarkExample:
    """A single evaluation example, normalized across all benchmarks.

    Fields:
        id: Unique identifier for this example.
        benchmark: Source benchmark name (musique, hotpotqa, longmemeval, etc.).
        query: The question or context to evaluate against.
        answer: Ground truth answer string.
        answer_aliases: Acceptable alternative answers.
        documents: The corpus documents for this example.
        supporting_doc_ids: IDs of documents needed to answer correctly.
        reasoning_chain: Multi-hop decomposition (if available).
        category: Benchmark-specific category (e.g., question_type in LongMemEval).
        metadata: Any extra benchmark-specific data.
    """

    id: str
    benchmark: str
    query: str
    answer: str
    answer_aliases: list[str] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)
    supporting_doc_ids: list[str] = field(default_factory=list)
    reasoning_chain: list[ReasoningStep] = field(default_factory=list)
    category: str = ""
    metadata: dict = field(default_factory=dict)
