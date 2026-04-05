"""Tests for PipelineConfig and ablation variants."""

from loqi.pipeline.config import (
    ALL_VARIANTS,
    FLAT_RAG,
    GRAPH_ONLY,
    LOQI_FULL,
    LOQI_NO_DIFFUSE,
    LOQI_NO_HEBBIAN,
    LOQI_NO_TRIGGERS,
    PipelineConfig,
)


class TestVariantNames:
    def test_flat_rag(self):
        assert FLAT_RAG.variant_name == "flat-rag"

    def test_graph_only(self):
        assert GRAPH_ONLY.variant_name == "graph-only"

    def test_no_triggers(self):
        assert LOQI_NO_TRIGGERS.variant_name == "loqi-no-triggers"

    def test_no_diffuse(self):
        assert LOQI_NO_DIFFUSE.variant_name == "loqi-no-diffuse"

    def test_no_hebbian(self):
        assert LOQI_NO_HEBBIAN.variant_name == "loqi-no-hebbian"

    def test_full(self):
        assert LOQI_FULL.variant_name == "loqi-full"


class TestAllVariants:
    def test_six_variants(self):
        assert len(ALL_VARIANTS) == 6

    def test_names_match_configs(self):
        for name, config in ALL_VARIANTS.items():
            assert config.variant_name == name

    def test_all_have_same_seed(self):
        seeds = {c.random_seed for c in ALL_VARIANTS.values()}
        assert len(seeds) == 1  # All share seed=42 for reproducibility


class TestConfigImmutable:
    def test_frozen(self):
        import pytest
        with pytest.raises(AttributeError):
            LOQI_FULL.enable_triggers = False
