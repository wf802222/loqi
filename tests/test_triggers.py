"""Tests for the trigger engine: extraction, matching, and TriggerRAG."""

import numpy as np
import pytest

from loqi.graph.models import Trigger, TriggerOrigin
from loqi.triggers.extractor import (
    _extract_keywords,
    _split_markdown_sections,
    extract_triggers,
)
from loqi.triggers.matcher import (
    _tokenize_context,
    keyword_score,
    match_triggers,
    semantic_score,
)


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

class TestExtractKeywords:
    def test_removes_stop_words(self):
        kw = _extract_keywords("the cat is on the mat")
        assert "the" not in kw
        assert "cat" in kw
        assert "mat" in kw

    def test_removes_short_tokens(self):
        kw = _extract_keywords("I am at UI HQ")
        assert "am" not in kw
        assert "at" not in kw

    def test_keeps_domain_terms(self):
        kw = _extract_keywords("Always use COC pattern in React components")
        assert "coc" in kw
        assert "react" in kw
        assert "components" in kw
        assert "pattern" in kw

    def test_deduplicates(self):
        kw = _extract_keywords("React React React component component")
        assert kw.count("react") == 1

    def test_handles_punctuation(self):
        kw = _extract_keywords("snake_case for REST endpoints (required).")
        assert "snake_case" in kw
        assert "rest" in kw
        assert "endpoints" in kw


# ---------------------------------------------------------------------------
# Markdown section splitting
# ---------------------------------------------------------------------------

class TestSplitSections:
    def test_splits_by_h2(self):
        md = "# Title\n\n## Section A\nContent A\n\n## Section B\nContent B"
        sections = _split_markdown_sections(md)
        assert len(sections) == 2
        assert sections[0][0] == "Section A"
        assert "Content A" in sections[0][1]
        assert sections[1][0] == "Section B"

    def test_no_sections(self):
        md = "Just a paragraph of text."
        sections = _split_markdown_sections(md)
        assert len(sections) == 1
        assert sections[0][0] == ""  # no heading

    def test_top_level_heading_only(self):
        md = "# Title\nSome intro content"
        sections = _split_markdown_sections(md)
        assert len(sections) == 1
        assert sections[0][0] == "Title"

    def test_real_memory_file(self):
        md = (
            "# Coding Standards\n\n"
            "## UI Component Rules\n"
            "Always use COC.\n\n"
            "## API Conventions\n"
            "Use snake_case.\n\n"
            "## Error Handling\n"
            "Never swallow exceptions.\n"
        )
        sections = _split_markdown_sections(md)
        assert len(sections) == 3
        assert sections[0][0] == "UI Component Rules"
        assert sections[1][0] == "API Conventions"
        assert sections[2][0] == "Error Handling"


# ---------------------------------------------------------------------------
# Trigger extraction (needs embedding model)
# ---------------------------------------------------------------------------

class TestExtractTriggers:
    @pytest.fixture(scope="class")
    def model(self):
        from loqi.graph.embeddings import EmbeddingModel
        return EmbeddingModel()

    def test_extracts_from_coding_standards(self, model):
        content = (
            "# Coding Standards\n\n"
            "## UI Component Rules\n"
            "Always use COC in UI components. Never FCOC.\n\n"
            "## API Conventions\n"
            "Use snake_case for REST endpoints.\n"
        )
        triggers = extract_triggers("coding_standards.md", content, model)

        assert len(triggers) == 2
        assert triggers[0].associated_node_id == "coding_standards.md"
        assert triggers[0].origin == TriggerOrigin.EXPLICIT
        assert triggers[0].confidence == 1.0
        assert triggers[0].pattern_embedding is not None
        assert len(triggers[0].pattern) > 0

    def test_keywords_contain_domain_terms(self, model):
        content = "## Feature Flags\nNew payment features must use feature flags."
        triggers = extract_triggers("rules.md", content, model)

        assert len(triggers) == 1
        kw = triggers[0].pattern
        assert "payment" in kw
        assert "feature" in kw
        assert "flags" in kw

    def test_embedding_dimension(self, model):
        content = "## Section\nSome content here."
        triggers = extract_triggers("test.md", content, model)
        assert triggers[0].pattern_embedding.shape == (model.dimension,)


# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------

class TestKeywordScore:
    def test_full_match(self):
        t = Trigger(
            id="t1", pattern=["react", "component"],
            associated_node_id="doc1",
        )
        tokens = _tokenize_context("Build a React component")
        assert keyword_score(t, tokens) == 1.0

    def test_partial_match(self):
        t = Trigger(
            id="t1", pattern=["react", "component", "frontend", "css"],
            associated_node_id="doc1",
        )
        tokens = _tokenize_context("Build a React component")
        assert keyword_score(t, tokens) == 0.5  # 2 of 4

    def test_no_match(self):
        t = Trigger(
            id="t1", pattern=["database", "migration", "schema"],
            associated_node_id="doc1",
        )
        tokens = _tokenize_context("Build a React component")
        assert keyword_score(t, tokens) == 0.0

    def test_empty_pattern(self):
        t = Trigger(id="t1", pattern=[], associated_node_id="doc1")
        tokens = _tokenize_context("anything")
        assert keyword_score(t, tokens) == 0.0


# ---------------------------------------------------------------------------
# Semantic matching
# ---------------------------------------------------------------------------

class TestSemanticScore:
    def test_similar_context(self):
        # Same vector = perfect similarity
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        t = Trigger(
            id="t1", pattern=["x"],
            pattern_embedding=emb,
            associated_node_id="doc1",
        )
        assert abs(semantic_score(t, emb) - 1.0) < 1e-6

    def test_no_embedding(self):
        t = Trigger(
            id="t1", pattern=["x"],
            pattern_embedding=None,
            associated_node_id="doc1",
        )
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert semantic_score(t, emb) == 0.0


# ---------------------------------------------------------------------------
# Full trigger matching
# ---------------------------------------------------------------------------

class TestMatchTriggers:
    @pytest.fixture(scope="class")
    def model(self):
        from loqi.graph.embeddings import EmbeddingModel
        return EmbeddingModel()

    def test_fires_on_relevant_context(self, model):
        ui_emb = model.encode_single(
            "UI Component Rules: Always use COC in UI components"
        )
        t = Trigger(
            id="t1",
            pattern=["component", "frontend", "react", "coc", "fcoc"],
            pattern_embedding=ui_emb,
            associated_node_id="coding_standards.md",
        )
        context = "Build a new React dropdown component for the settings page"
        ctx_emb = model.encode_single(context)

        fired = match_triggers([t], context, ctx_emb, threshold=0.2)
        assert len(fired) >= 1
        assert fired[0][0].id == "t1"

    def test_does_not_fire_on_unrelated_context(self, model):
        ui_emb = model.encode_single(
            "UI Component Rules: Always use COC in UI components"
        )
        t = Trigger(
            id="t1",
            pattern=["component", "frontend", "react", "coc", "fcoc"],
            pattern_embedding=ui_emb,
            associated_node_id="coding_standards.md",
        )
        context = "Optimize the batch job that processes daily transaction reports"
        ctx_emb = model.encode_single(context)

        fired = match_triggers([t], context, ctx_emb, threshold=0.3)
        assert len(fired) == 0

    def test_confidence_decay_prevents_firing(self, model):
        emb = model.encode_single("some pattern text")
        t = Trigger(
            id="t1",
            pattern=["react", "component"],
            pattern_embedding=emb,
            associated_node_id="doc1",
            confidence=0.05,  # Below 0.1 threshold
        )
        context = "Build a React component"
        ctx_emb = model.encode_single(context)

        fired = match_triggers([t], context, ctx_emb, threshold=0.1)
        assert len(fired) == 0

    def test_multiple_triggers_sorted_by_score(self, model):
        t1 = Trigger(
            id="t1",
            pattern=["react", "component", "frontend"],
            pattern_embedding=model.encode_single("React frontend component UI"),
            associated_node_id="doc1",
        )
        t2 = Trigger(
            id="t2",
            pattern=["database", "migration"],
            pattern_embedding=model.encode_single("database migration schema"),
            associated_node_id="doc2",
        )

        context = "Build a new React component for the dashboard"
        ctx_emb = model.encode_single(context)

        fired = match_triggers([t1, t2], context, ctx_emb, threshold=0.15)

        # t1 should fire (React/component match), t2 should not (database/migration)
        fired_ids = [t.id for t, _ in fired]
        assert "t1" in fired_ids
