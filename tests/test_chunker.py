"""Tests for TextChunker."""
import pytest
from data.processors.text_chunker import TextChunker


def test_basic_chunking():
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    text = " ".join(["word"] * 200)
    chunks = chunker.chunk(text, doc_id="test_doc")
    assert len(chunks) > 1
    for c in chunks:
        assert c.content
        assert c.doc_id == "test_doc"


def test_empty_text():
    chunker = TextChunker()
    assert chunker.chunk("") == []


def test_chunk_overlap():
    chunker = TextChunker(chunk_size=30, chunk_overlap=10)
    text = "Hello world. This is a test. " * 20
    chunks = chunker.chunk(text)
    assert all(len(c.content.split()) <= 40 for c in chunks)


def test_chunk_metadata():
    chunker = TextChunker(chunk_size=50)
    meta = {"source": "wiki", "title": "Test"}
    chunks = chunker.chunk("Word " * 100, doc_id="d1", metadata=meta)
    for c in chunks:
        assert c.metadata == meta
