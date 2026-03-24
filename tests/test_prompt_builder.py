"""Tests for PromptBuilder."""

from generation.prompt_builder import PromptBuilder


def test_basic_build(sample_query, sample_contexts):
    builder = PromptBuilder()
    prompt = builder.build(sample_query, sample_contexts)
    assert sample_query in prompt
    assert "[1]" in prompt
    assert "[2]" in prompt


def test_empty_contexts(sample_query):
    builder = PromptBuilder()
    prompt = builder.build(sample_query, [])
    assert sample_query in prompt


def test_healing_prompt(sample_query, sample_contexts):
    builder = PromptBuilder()
    prompt = builder.build_healing_prompt(sample_query, sample_contexts)
    assert "context" in prompt.lower()
    assert sample_query in prompt
