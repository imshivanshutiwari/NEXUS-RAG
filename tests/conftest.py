"""Shared pytest fixtures."""
import pytest


@pytest.fixture
def sample_docs():
    return [
        {"doc_id": "doc1", "content": "Retrieval-Augmented Generation combines retrieval with language models.", "source": "test"},
        {"doc_id": "doc2", "content": "pgvector enables ANN search inside PostgreSQL using HNSW indexing.", "source": "test"},
        {"doc_id": "doc3", "content": "BM25 is a probabilistic ranking function used in information retrieval.", "source": "test"},
        {"doc_id": "doc4", "content": "ColBERT uses late interaction for efficient reranking.", "source": "test"},
        {"doc_id": "doc5", "content": "LangGraph builds stateful multi-agent pipelines as graphs.", "source": "test"},
    ]


@pytest.fixture
def sample_query():
    return "What is Retrieval-Augmented Generation?"


@pytest.fixture
def sample_answer():
    return "RAG combines a retrieval system with a language model to generate grounded answers [1]."


@pytest.fixture
def sample_contexts(sample_docs):
    return [d["content"] for d in sample_docs[:3]]
