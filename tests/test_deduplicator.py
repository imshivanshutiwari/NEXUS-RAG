"""Tests for Deduplicator."""
import pytest
from ingestion.deduplicator import Deduplicator


def test_no_duplicates(sample_docs):
    dedup = Deduplicator()
    result = dedup.deduplicate(sample_docs)
    assert len(result) == len(sample_docs)


def test_exact_duplicates():
    dedup = Deduplicator(threshold=0.9)
    docs = [
        {"doc_id": "a", "content": "The quick brown fox jumps over the lazy dog."},
        {"doc_id": "b", "content": "The quick brown fox jumps over the lazy dog."},
    ]
    result = dedup.deduplicate(docs)
    assert len(result) == 1


def test_empty_list():
    dedup = Deduplicator()
    assert dedup.deduplicate([]) == []
