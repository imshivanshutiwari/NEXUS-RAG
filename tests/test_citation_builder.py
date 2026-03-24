"""Tests for CitationBuilder."""

from generation.citation_builder import CitationBuilder


def test_extract_citations(sample_contexts):
    builder = CitationBuilder()
    docs = [
        {
            "doc_id": f"d{i}",
            "content": c,
            "source": "test",
            "chunk_index": i,
            "metadata": {},
        }
        for i, c in enumerate(sample_contexts)
    ]
    answer = "According to the first passage [1] and the second [2], RAG is effective."
    citations = builder.extract(answer, docs)
    assert len(citations) == 2
    assert citations[0]["citation_number"] == 1
    assert citations[1]["citation_number"] == 2


def test_no_citations(sample_contexts):
    builder = CitationBuilder()
    docs = [
        {
            "doc_id": "d0",
            "content": "text",
            "source": "test",
            "chunk_index": 0,
            "metadata": {},
        }
    ]
    citations = builder.extract("No citations here.", docs)
    assert citations == []


def test_bibliography(sample_contexts):
    builder = CitationBuilder()
    citations = [
        {
            "citation_number": 1,
            "title": "Wikipedia",
            "url": "https://en.wikipedia.org",
            "source": "wiki",
        }
    ]
    bib = builder.format_bibliography(citations)
    assert "[1]" in bib
    assert "Wikipedia" in bib


def test_out_of_range_citation():
    builder = CitationBuilder()
    docs = [
        {
            "doc_id": "d0",
            "content": "text",
            "source": "test",
            "chunk_index": 0,
            "metadata": {},
        }
    ]
    citations = builder.extract("See [5] for details.", docs)
    assert citations == []
