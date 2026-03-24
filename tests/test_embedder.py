"""Tests for DocumentEmbedder (local fallback path)."""

import numpy as np
from unittest.mock import patch, MagicMock


def test_embed_query_shape():
    with patch.dict("os.environ", {"COHERE_API_KEY": ""}):
        from ingestion.embedder import DocumentEmbedder

        embedder = DocumentEmbedder()
        # Mock local model
        embedder._local_model = MagicMock()
        embedder._local_model.encode.return_value = np.random.rand(1, 768).astype(
            "float32"
        )
        result = embedder.embed_query("test query")
        assert result.shape == (1024,)


def test_embed_documents_shape():
    with patch.dict("os.environ", {"COHERE_API_KEY": ""}):
        from ingestion.embedder import DocumentEmbedder

        embedder = DocumentEmbedder()
        embedder._local_model = MagicMock()
        embedder._local_model.encode.return_value = np.random.rand(3, 768).astype(
            "float32"
        )
        result = embedder.embed_documents(["doc1", "doc2", "doc3"])
        assert result.shape == (3, 1024)


def test_compute_similarity():
    with patch.dict("os.environ", {"COHERE_API_KEY": ""}):
        from ingestion.embedder import DocumentEmbedder

        embedder = DocumentEmbedder()
        a = np.ones(1024, dtype="float32")
        b = np.ones(1024, dtype="float32")
        sim = embedder.compute_similarity(a, b)
        assert abs(sim - 1.0) < 1e-4
