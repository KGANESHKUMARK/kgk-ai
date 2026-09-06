"""KGK AI RAG Pipeline — End-to-end retrieval-augmented generation.

Orchestrates the full RAG flow:
    1. Load documents from knowledge directory
    2. Chunk documents into embeddable segments
    3. Embed chunks using the embedding model
    4. Store embeddings in the FAISS vector store
    5. On query: embed query, retrieve relevant chunks, return with sources

The pipeline is designed to be rebuildable — all data can be reconstructed
from source documents in the knowledge directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from app.config import get_settings
from app.logging_config import get_logger
from app.rag.chunker import TextChunker
from app.rag.embedder import EmbeddingProvider
from app.rag.faiss_store import FaissVectorStore
from app.rag.loader import DocumentLoader, LoadedDocument
from app.rag.vector_store import DocumentChunk, RetrievalResult

logger = get_logger("rag.pipeline")


@dataclass
class RAGResponse:
    """Result of a RAG retrieval query.

    Attributes:
        context: Concatenated context text from retrieved chunks.
        sources: List of source citations (filename:chunk_id).
        chunks: List of retrieved DocumentChunk instances.
        scores: Similarity scores for each retrieved chunk.
    """

    context: str = ""
    sources: list[str] = field(default_factory=list)
    chunks: list[DocumentChunk] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)


class RAGPipeline:
    """End-to-end RAG pipeline for KGK AI.

    Attributes:
        loader: Document loader instance.
        chunker: Text chunker instance.
        embedder: Embedding provider instance.
        vector_store: FAISS vector store instance.
        top_k: Default number of chunks to retrieve.
    """

    def __init__(
        self,
        loader: Optional[DocumentLoader] = None,
        chunker: Optional[TextChunker] = None,
        embedder: Optional[EmbeddingProvider] = None,
        vector_store: Optional[FaissVectorStore] = None,
        top_k: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self.loader: DocumentLoader = loader or DocumentLoader()
        self.chunker: TextChunker = chunker or TextChunker()
        self.embedder: EmbeddingProvider = embedder or EmbeddingProvider()
        self.vector_store: FaissVectorStore = vector_store or FaissVectorStore()
        self.top_k: int = top_k or settings.top_k_retrieval
        self._initialized: bool = False

    def initialize(self, force_rebuild: bool = False) -> int:
        """Initialize the RAG pipeline.

        Attempts to load an existing vector store from disk. If not found
        or if force_rebuild is True, rebuilds from source documents.

        Args:
            force_rebuild: If True, ignore cached store and rebuild.

        Returns:
            Number of chunks in the vector store.
        """
        if self._initialized and not force_rebuild:
            return self.vector_store.size()

        settings = get_settings()
        vectorstore_path = Path(settings.vectorstore_dir)

        # Try loading existing store
        if not force_rebuild and vectorstore_path.exists():
            try:
                self.vector_store.load(str(vectorstore_path))
                count = self.vector_store.size()
                if count > 0:
                    logger.info(
                        f"RAG initialized from cache: {count} chunks",
                        extra={"component": "rag.pipeline", "chunks": count},
                    )
                    self._initialized = True
                    return count
            except Exception as e:
                logger.warning(f"Failed to load cached store: {e}, rebuilding...")

        # Rebuild from documents
        return self.ingest_directory(settings.knowledge_dir)

    def ingest_directory(self, dir_path: str) -> int:
        """Ingest all documents from a directory.

        Args:
            dir_path: Path to the documents directory.

        Returns:
            Number of chunks added to the vector store.
        """
        logger.info(
            f"Ingesting documents from: {dir_path}",
            extra={"component": "rag.pipeline", "dir": dir_path},
        )

        # Load documents
        documents = self.loader.load_directory(dir_path)
        if not documents:
            logger.warning("No documents found to ingest")
            return 0

        return self.ingest_documents(documents)

    def ingest_documents(self, documents: list[LoadedDocument]) -> int:
        """Ingest a list of documents into the vector store.

        Args:
            documents: List of LoadedDocument instances.

        Returns:
            Number of chunks added to the vector store.
        """
        if not documents:
            return 0

        # Chunk documents
        chunks = self.chunker.chunk_documents(documents)
        if not chunks:
            logger.warning("No chunks generated from documents")
            return 0

        # Embed chunks
        texts = [c.content for c in chunks]
        embeddings = self.embedder.embed(texts)

        # Store in vector store
        self.vector_store.clear()
        self.vector_store.add(chunks, embeddings)

        # Persist to disk
        settings = get_settings()
        self.vector_store.save(settings.vectorstore_dir)

        self._initialized = True

        logger.info(
            f"Ingestion complete: {len(chunks)} chunks from {len(documents)} documents",
            extra={"component": "rag.pipeline", "chunks": len(chunks), "documents": len(documents)},
        )

        return len(chunks)

    def ingest_text(
        self,
        text: str,
        source: str = "inline",
        filename: str = "inline.txt",
    ) -> int:
        """Ingest raw text into the vector store.

        Args:
            text: Raw text to ingest.
            source: Source path.
            filename: Filename for metadata.

        Returns:
            Number of chunks added.
        """
        chunks = self.chunker.chunk_text(text, source=source, filename=filename)
        if not chunks:
            return 0

        embeddings = self.embedder.embed([c.content for c in chunks])
        self.vector_store.add(chunks, embeddings)

        settings = get_settings()
        self.vector_store.save(settings.vectorstore_dir)

        self._initialized = True

        return len(chunks)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        """Retrieve relevant chunks for a query.

        Args:
            query: Query text.
            top_k: Number of chunks to retrieve (defaults to self.top_k).

        Returns:
            RAGResponse with context, sources, chunks, and scores.
        """
        if not self._initialized:
            self.initialize()

        if self.vector_store.size() == 0:
            return RAGResponse()

        k = top_k or self.top_k

        # Embed query
        query_embedding = self.embedder.embed_query(query)

        # Search
        results: list[RetrievalResult] = self.vector_store.search(query_embedding, top_k=k)

        if not results:
            return RAGResponse()

        # Build response
        context_parts = []
        sources = []
        chunks = []
        scores = []

        for result in results:
            chunk = result.chunk
            context_parts.append(chunk.content)
            source = f"{chunk.filename}:{chunk.chunk_id}" if chunk.filename else chunk.chunk_id
            sources.append(source)
            chunks.append(chunk)
            scores.append(result.score)

        context = "\n\n---\n\n".join(context_parts)

        logger.info(
            f"RAG retrieval: query='{query[:50]}...', results={len(results)}",
            extra={
                "component": "rag.pipeline",
                "query_length": len(query),
                "results": len(results),
                "top_score": scores[0] if scores else 0,
            },
        )

        return RAGResponse(
            context=context,
            sources=sources,
            chunks=chunks,
            scores=scores,
        )

    def get_context_for_prompt(self, query: str, top_k: Optional[int] = None) -> str:
        """Retrieve context formatted for injection into a model prompt.

        Args:
            query: Query text.
            top_k: Number of chunks to retrieve.

        Returns:
            Formatted context string with source citations, or empty string
            if no results.
        """
        response = self.retrieve(query, top_k=top_k)

        if not response.chunks:
            return ""

        parts = []
        for i, chunk in enumerate(response.chunks, 1):
            source = chunk.filename or chunk.source or "unknown"
            parts.append(f"[Source {i}: {source}]\n{chunk.content}")

        return "\n\n".join(parts)

    def is_ready(self) -> bool:
        """Check if the RAG pipeline is initialized and has data.

        Returns:
            True if the pipeline is ready for retrieval.
        """
        return self._initialized and self.vector_store.size() > 0

    def get_stats(self) -> dict[str, Any]:
        """Return pipeline statistics.

        Returns:
            Dict with chunk count, embedding model info, and status.
        """
        return {
            "initialized": self._initialized,
            "chunk_count": self.vector_store.size(),
            "embedding_model": self.embedder.model_name,
            "embedding_dimension": self.embedder.dimension,
            "chunk_size": self.chunker.chunk_size,
            "chunk_overlap": self.chunker.chunk_overlap,
            "top_k": self.top_k,
        }


# Singleton instance
_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Return the singleton RAG pipeline instance.

    Returns:
        RAGPipeline instance.
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
        logger.info("RAG pipeline instance created")
    return _pipeline
