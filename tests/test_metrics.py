"""Tests for evaluation metrics.

These are pure function tests — no data dependencies.
"""

import pytest

from loqi.eval.metrics import (
    aggregate_metrics,
    answer_exact_match,
    answer_f1,
    normalize_answer,
    retrieval_precision_at_k,
    retrieval_recall_at_k,
    support_f1,
    trigger_precision,
    trigger_recall,
)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_lowercase(self):
        assert normalize_answer("Hello World") == "hello world"

    def test_remove_articles(self):
        assert normalize_answer("the cat and a dog") == "cat and dog"

    def test_remove_punctuation(self):
        assert normalize_answer("hello, world!") == "hello world"

    def test_collapse_whitespace(self):
        assert normalize_answer("  hello   world  ") == "hello world"

    def test_combined(self):
        assert normalize_answer("The Answer is: 42!") == "answer is 42"


# ---------------------------------------------------------------------------
# Answer Exact Match
# ---------------------------------------------------------------------------

class TestAnswerEM:
    def test_exact_match(self):
        assert answer_exact_match("Paris", "Paris") == 1.0

    def test_case_insensitive(self):
        assert answer_exact_match("paris", "Paris") == 1.0

    def test_no_match(self):
        assert answer_exact_match("London", "Paris") == 0.0

    def test_alias_match(self):
        assert answer_exact_match("NYC", "New York City", aliases=["NYC", "New York"]) == 1.0

    def test_article_removal(self):
        assert answer_exact_match("the United States", "United States") == 1.0


# ---------------------------------------------------------------------------
# Answer F1
# ---------------------------------------------------------------------------

class TestAnswerF1:
    def test_perfect_match(self):
        assert answer_f1("Miller County", "Miller County") == 1.0

    def test_partial_match(self):
        # "Miller County Missouri" vs "Miller County" -> precision=2/3, recall=2/2
        f1 = answer_f1("Miller County Missouri", "Miller County")
        expected = 2 * (2 / 3) * 1.0 / ((2 / 3) + 1.0)
        assert abs(f1 - expected) < 1e-6

    def test_no_overlap(self):
        assert answer_f1("London", "Paris") == 0.0

    def test_empty_prediction(self):
        assert answer_f1("", "Paris") == 0.0

    def test_alias_takes_best(self):
        # Should use whichever alias gives highest F1
        f1 = answer_f1("NYC", "New York City", aliases=["NYC"])
        assert f1 == 1.0

    def test_symmetric_for_identical(self):
        f1 = answer_f1("hello world", "hello world")
        assert f1 == 1.0


# ---------------------------------------------------------------------------
# Support F1
# ---------------------------------------------------------------------------

class TestSupportF1:
    def test_perfect(self):
        assert support_f1({"a", "b"}, {"a", "b"}) == 1.0

    def test_no_overlap(self):
        assert support_f1({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial(self):
        # predicted={a,b,c}, gold={a,b} -> precision=2/3, recall=2/2=1
        f1 = support_f1({"a", "b", "c"}, {"a", "b"})
        expected = 2 * (2 / 3) * 1.0 / ((2 / 3) + 1.0)
        assert abs(f1 - expected) < 1e-6

    def test_both_empty(self):
        assert support_f1(set(), set()) == 1.0

    def test_predicted_empty(self):
        assert support_f1(set(), {"a"}) == 0.0

    def test_gold_empty(self):
        assert support_f1({"a"}, set()) == 0.0


# ---------------------------------------------------------------------------
# Retrieval Recall@k and Precision@k
# ---------------------------------------------------------------------------

class TestRetrievalMetrics:
    def test_recall_at_1_hit(self):
        assert retrieval_recall_at_k(["a", "b", "c"], {"a"}, k=1) == 1.0

    def test_recall_at_1_miss(self):
        assert retrieval_recall_at_k(["b", "a", "c"], {"a"}, k=1) == 0.0

    def test_recall_at_3_partial(self):
        # gold={a,d}, top-3=[a,b,c] -> found a, missed d -> 0.5
        assert retrieval_recall_at_k(["a", "b", "c"], {"a", "d"}, k=3) == 0.5

    def test_precision_at_3(self):
        # top-3=[a,b,c], gold={a,c} -> 2/3
        p = retrieval_precision_at_k(["a", "b", "c"], {"a", "c"}, k=3)
        assert abs(p - 2 / 3) < 1e-6

    def test_precision_empty_retrieved(self):
        assert retrieval_precision_at_k([], {"a"}, k=5) == 0.0

    def test_recall_empty_gold(self):
        assert retrieval_recall_at_k(["a", "b"], set(), k=2) == 1.0


# ---------------------------------------------------------------------------
# Trigger Recall and Precision
# ---------------------------------------------------------------------------

class TestTriggerMetrics:
    def test_recall_all_fired(self):
        assert trigger_recall({"a", "b"}, {"a", "b"}) == 1.0

    def test_recall_partial(self):
        assert trigger_recall({"a"}, {"a", "b"}) == 0.5

    def test_recall_none_fired(self):
        assert trigger_recall(set(), {"a", "b"}) == 0.0

    def test_recall_no_expected(self):
        assert trigger_recall({"a"}, set()) == 1.0

    def test_precision_no_false_positives(self):
        assert trigger_precision({"a", "b"}, {"a", "b"}, {"c"}) == 1.0

    def test_precision_one_false_positive(self):
        # fired={a,c}, expected={a}, non_expected={c} -> 1 FP out of 2 fired
        assert trigger_precision({"a", "c"}, {"a"}, {"c"}) == 0.5

    def test_precision_nothing_fired(self):
        assert trigger_precision(set(), {"a"}, {"c"}) == 1.0

    def test_precision_neutral_triggers_not_penalized(self):
        # fired={a,x}, expected={a}, non_expected={c}
        # x is not in non_expected, so it's neutral, not penalized
        assert trigger_precision({"a", "x"}, {"a"}, {"c"}) == 1.0


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

class TestAggregate:
    def test_basic(self):
        result = aggregate_metrics([0.5, 0.75, 1.0])
        assert result["mean"] == 0.75
        assert result["min"] == 0.5
        assert result["max"] == 1.0
        assert result["count"] == 3

    def test_empty(self):
        result = aggregate_metrics([])
        assert result["count"] == 0
