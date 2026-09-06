"""Unit tests for KGK AI RAG components.

Tests cover:
- DocumentLoader: file loading, format support, error handling
- TextChunker: chunking logic, overlap, metadata
- FaissVectorStore: add, search, save/load, clear, size
- RAGPipeline: initialization, ingestion, retrieval, context formatting
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_faiss
from unittest.mock import MagicMock, patch

from app.rag.loader import DocumentLoader, LoadedDocument
from app.rag.chunker import TextChunker
from app.rag.vector_store import DocumentChunk, RetrievalResult
from app.rag.embedder import EmbeddingProvider
from app.rag.faiss_store import FaissVectorStore
from app.rag.pipeline import RAGPipeline, RAGResponse


class TestDocumentLoader:
    """Tests for DocumentLoader."""

    def test_load_txt_file(self, tmp_path):
        """Load a plain text file."""
        f = tmp_path / "test.txt"
        f.write_text("Hello, KGK AI!", encoding="utf-8")

        loader = DocumentLoader()
        doc = loader.load_file(f)

        assert doc is not None
        assert doc.content == "Hello, KGK AI!"
        assert doc.filename == "test.txt"
        assert doc.file_type == ".txt"
        assert doc.document_id == "test"

    def test_load_md_file(self, tmp_path):
        """Load a Markdown file."""
        f = tmp_path / "readme.md"
        f.write_text("# Title\n\nContent here.", encoding="utf-8")

        loader = DocumentLoader()
        doc = loader.load_file(f)

        assert doc is not None
        assert "# Title" in doc.content
        assert doc.file_type == ".md"

    def test_load_json_file(self, tmp_path):
        """Load a JSON file with text field."""
        f = tmp_path / "data.json"
        f.write_text(json.dumps({"text": "JSON content"}), encoding="utf-8")

        loader = DocumentLoader()
        doc = loader.load_file(f)

        assert doc is not None
        assert doc.content == "JSON content"

    def test_load_json_array(self, tmp_path):
        """Load a JSON array of text objects."""
        f = tmp_path / "array.json"
        f.write_text(json.dumps([
            {"text": "First item"},
            {"text": "Second item"},
        ]), encoding="utf-8")

        loader = DocumentLoader()
        doc = loader.load_file(f)

        assert doc is not None
        assert "First item" in doc.content
        assert "Second item" in doc.content

    def test_load_nonexistent_file(self):
        """Loading a nonexistent file returns None."""
        loader = DocumentLoader()
        doc = loader.load_file("nonexistent.txt")
        assert doc is None

    def test_load_unsupported_format(self, tmp_path):
        """Loading an unsupported format returns None."""
        f = tmp_path / "test.xyz"
        f.write_text("content", encoding="utf-8")

        loader = DocumentLoader()
        doc = loader.load_file(f)
        assert doc is None

    def test_load_empty_file(self, tmp_path):
        """Loading an empty file returns None."""
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")

        loader = DocumentLoader()
        doc = loader.load_file(f)
        assert doc is None

    def test_load_directory(self, tmp_path):
        """Load all supported files from a directory."""
        (tmp_path / "a.txt").write_text("Content A", encoding="utf-8")
        (tmp_path / "b.md").write_text("Content B", encoding="utf-8")
        (tmp_path / "c.json").write_text(json.dumps({"text": "C"}), encoding="utf-8")
        (tmp_path / "d.xyz").write_text("Unsupported", encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load_directory(tmp_path)

        assert len(docs) == 3  # txt, md, json

    def test_load_nonexistent_directory(self):
        """Loading from a nonexistent directory returns empty list."""
        loader = DocumentLoader()
        docs = loader.load_directory("nonexistent_dir")
        assert docs == []


class TestTextChunker:
    """Tests for TextChunker."""

    def test_short_text_single_chunk(self):
        """Short text produces a single chunk."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        doc = LoadedDocument(
            content="This is a short text.",
            filename="test.txt",
            document_id="test",
        )
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0].content == "This is a short text."
        assert chunks[0].filename == "test.txt"
        assert chunks[0].document_id == "test"

    def test_long_text_multiple_chunks(self):
        """Long text produces multiple chunks with overlap."""
        chunker = TextChunker(chunk_size=10, chunk_overlap=3)
        words = [f"word{i}" for i in range(30)]
        doc = LoadedDocument(
            content=" ".join(words),
            filename="long.txt",
            document_id="long",
        )
        chunks = chunker.chunk_document(doc)

        assert len(chunks) > 1
        # Verify chunk IDs are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"long_{i:04d}"

    def test_chunk_metadata(self):
        """Chunks have correct metadata."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        doc = LoadedDocument(
            content="Some text content here.",
            source="/path/to/file.txt",
            filename="file.txt",
            document_id="file",
            file_type=".txt",
        )
        chunks = chunker.chunk_document(doc)

        assert chunks[0].source == "/path/to/file.txt"
        assert chunks[0].filename == "file.txt"
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[0].metadata["word_count"] > 0
        assert chunks[0].metadata["file_type"] == ".txt"

    def test_chunk_empty_content(self):
        """Empty content produces no chunks."""
        chunker = TextChunker()
        doc = LoadedDocument(content="", filename="empty.txt", document_id="empty")
        chunks = chunker.chunk_document(doc)
        assert chunks == []

    def test_chunk_text_method(self):
        """chunk_text works without a LoadedDocument."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_text("Hello world.", source="inline", filename="test.txt")

        assert len(chunks) == 1
        assert chunks[0].content == "Hello world."

    def test_chunk_multiple_documents(self):
        """chunk_documents processes multiple documents."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        docs = [
            LoadedDocument(content="Doc one content.", filename="d1.txt", document_id="d1"),
            LoadedDocument(content="Doc two content.", filename="d2.txt", document_id="d2"),
        ]
        chunks = chunker.chunk_documents(docs)

        assert len(chunks) == 2
        assert chunks[0].document_id == "d1"
        assert chunks[1].document_id == "d2"


class TestFaissVectorStore:
    """Tests for FaissVectorStore."""

    def _make_chunks(self, n: int = 3) -> list[DocumentChunk]:
        """Create test chunks."""
        return [
            DocumentChunk(
                content=f"Content {i}",
                filename=f"file{i}.txt",
                chunk_id=f"file{i}_{i:04d}",
                document_id=f"file{i}",
            )
            for i in range(n)
        ]

    def _make_embeddings(self, n: int = 3, dim: int = 4) -> list[list[float]]:
        """Create test embeddings."""
        return [[float(i) for i in range(dim)] for _ in range(n)]

    def test_add_and_size(self):
        """Add chunks and verify size."""
        store = FaissVectorStore(dimension=4)
        chunks = self._make_chunks(3)
        embeddings = self._make_embeddings(3, 4)

        store.add(chunks, embeddings)
        assert store.size() == 3

    def test_add_empty(self):
        """Adding empty lists does nothing."""
        store = FaissVectorStore(dimension=4)
        store.add([], [])
        assert store.size() == 0

    def test_add_mismatch_raises(self):
        """Mismatched chunks and embeddings raises ValueError."""
        store = FaissVectorStore(dimension=4)
        with pytest.raises(ValueError):
            store.add(self._make_chunks(3), self._make_embeddings(2, 4))

    def test_search_empty_store(self):
        """Search on empty store returns empty list."""
        store = FaissVectorStore(dimension=4)
        results = store.search([1.0, 2.0, 3.0, 4.0])
        assert results == []

    def test_search_returns_results(self):
        """Search returns relevant results."""
        store = FaissVectorStore(dimension=4)
        chunks = self._make_chunks(3)
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]

        store.add(chunks, embeddings)
        results = store.search([1.0, 0.0, 0.0, 0.0], top_k=2)

        assert len(results) == 2
        assert results[0].chunk.content in ["Content 0", "Content 2"]  # Closest to query
        assert results[0].score > 0

    def test_clear(self):
        """Clear removes all entries."""
        store = FaissVectorStore(dimension=4)
        store.add(self._make_chunks(2), self._make_embeddings(2, 4))
        assert store.size() == 2

        store.clear()
        assert store.size() == 0

    def test_save_and_load(self, tmp_path):
        """Save and load the vector store."""
        store = FaissVectorStore(dimension=4)
        chunks = self._make_chunks(3)
        embeddings = self._make_embeddings(3, 4)
        store.add(chunks, embeddings)

        save_path = tmp_path / "vectorstore"
        store.save(str(save_path))

        # Load into a new store
        new_store = FaissVectorStore(dimension=4)
        new_store.load(str(save_path))

        assert new_store.size() == 3
        loaded_chunks = new_store.get_chunks()
        assert loaded_chunks[0].content == "Content 0"

    def test_load_nonexistent_path(self, tmp_path):
        """Loading from a nonexistent path does nothing."""
        store = FaissVectorStore(dimension=4)
        store.load(str(tmp_path / "nonexistent"))
        assert store.size() == 0

    def test_get_chunks(self):
        """get_chunks returns all stored chunks."""
        store = FaissVectorStore(dimension=4)
        chunks = self._make_chunks(2)
        store.add(chunks, self._make_embeddings(2, 4))

        retrieved = store.get_chunks()
        assert len(retrieved) == 2
        assert retrieved[0].content == "Content 0"


class TestEmbeddingProvider:
    """Tests for EmbeddingProvider."""

    def test_init_defaults(self):
        """Provider initializes with settings defaults."""
        provider = EmbeddingProvider()
        assert provider.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert provider.dimension == 384
        assert provider._loaded is False

    def test_custom_init(self):
        """Provider accepts custom parameters."""
        provider = EmbeddingProvider(model_name="custom-model", dimension=512)
        assert provider.model_name == "custom-model"
        assert provider.dimension == 512

    def test_is_loaded_false(self):
        """is_loaded returns False before loading."""
        provider = EmbeddingProvider()
        assert provider.is_loaded() is False

    def test_get_info(self):
        """get_info returns correct metadata."""
        provider = EmbeddingProvider()
        info = provider.get_info()
        assert info["model"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert info["dimension"] == 384
        assert info["loaded"] is False


class TestRAGPipeline:
    """Tests for RAGPipeline."""

    def test_init_defaults(self):
        """Pipeline initializes with default components."""
        pipeline = RAGPipeline()
        assert pipeline.loader is not None
        assert pipeline.chunker is not None
        assert pipeline.embedder is not None
        assert pipeline.vector_store is not None
        assert pipeline._initialized is False

    def test_is_ready_false_when_not_initialized(self):
        """is_ready returns False when not initialized."""
        pipeline = RAGPipeline()
        assert pipeline.is_ready() is False

    def test_get_stats(self):
        """get_stats returns pipeline statistics."""
        pipeline = RAGPipeline()
        stats = pipeline.get_stats()
        assert "initialized" in stats
        assert "chunk_count" in stats
        assert "embedding_model" in stats
        assert "chunk_size" in stats
        assert "top_k" in stats

    def test_retrieve_not_initialized_returns_empty(self):
        """Retrieve on uninitialized pipeline returns empty response."""
        pipeline = RAGPipeline()
        result = pipeline.retrieve("test query")
        assert isinstance(result, RAGResponse)
        assert result.context == ""
        assert result.sources == []

    def test_ingest_text_with_mocked_embedder(self):
        """Ingest text with a mocked embedding provider."""
        pipeline = RAGPipeline()
        pipeline.embedder = MagicMock(spec=EmbeddingProvider)
        pipeline.embedder.embed.return_value = [[0.1, 0.2, 0.3, 0.4]]
        pipeline.embedder.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        pipeline.vector_store = FaissVectorStore(dimension=4)

        # Patch save to avoid disk writes
        with patch.object(pipeline.vector_store, "save"):
            count = pipeline.ingest_text("This is a test document.", filename="test.txt")

        assert count > 0
        assert pipeline.is_ready()

    def test_retrieve_with_mocked_embedder(self):
        """Retrieve with mocked embedder returns results."""
        pipeline = RAGPipeline()
        pipeline.embedder = MagicMock(spec=EmbeddingProvider)
        pipeline.embedder.embed.return_value = [[0.1, 0.2, 0.3, 0.4]]
        pipeline.embedder.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        pipeline.vector_store = FaissVectorStore(dimension=4)

        with patch.object(pipeline.vector_store, "save"):
            pipeline.ingest_text("Test content for retrieval.", filename="test.txt")

        result = pipeline.retrieve("test query")

        assert isinstance(result, RAGResponse)
        assert len(result.chunks) > 0
        assert len(result.sources) > 0
        assert "test.txt" in result.sources[0]

    def test_get_context_for_prompt(self):
        """get_context_for_prompt returns formatted context."""
        pipeline = RAGPipeline()
        pipeline.embedder = MagicMock(spec=EmbeddingProvider)
        pipeline.embedder.embed.return_value = [[0.1, 0.2, 0.3, 0.4]]
        pipeline.embedder.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        pipeline.vector_store = FaissVectorStore(dimension=4)

        with patch.object(pipeline.vector_store, "save"):
            pipeline.ingest_text("Important knowledge content.", filename="knowledge.txt")

        context = pipeline.get_context_for_prompt("knowledge query")

        assert "knowledge.txt" in context
        assert "Important knowledge content" in context
        assert "Source" in context

    def test_get_context_empty_when_no_data(self):
        """get_context_for_prompt returns empty string when no data."""
        pipeline = RAGPipeline()
        pipeline.vector_store = FaissVectorStore(dimension=4)
        pipeline._initialized = True

        context = pipeline.get_context_for_prompt("query")
        assert context == ""
