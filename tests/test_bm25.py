"""Tests for BM25Indexer."""

import pytest
from ingestion.bm25_indexer import BM25Indexer


def test_build_and_search(sample_docs, tmp_path):
    indexer = BM25Indexer(index_path=tmp_path / "bm25.pkl")
    indexer.build(sample_docs)
    results = indexer.search("retrieval augmented generation", k=3)
    assert len(results) == 3
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    assert results[0][1] >= results[-1][1]


def test_empty_query(sample_docs, tmp_path):
    indexer = BM25Indexer(index_path=tmp_path / "bm25.pkl")
    indexer.build(sample_docs)
    results = indexer.search("", k=5)
    assert isinstance(results, list)


def test_len(sample_docs, tmp_path):
    indexer = BM25Indexer(index_path=tmp_path / "bm25.pkl")
    indexer.build(sample_docs)
    assert len(indexer) == len(sample_docs)


def test_save_load(sample_docs, tmp_path):
    path = tmp_path / "bm25.pkl"
    indexer = BM25Indexer(index_path=path)
    indexer.build(sample_docs)
    indexer2 = BM25Indexer(index_path=path)
    results = indexer2.search("PostgreSQL HNSW", k=3)
    assert len(results) == 3
