"""KGK AI RAG Vector Store — FAISS-based vector storage.

Manages document embeddings, similarity search, and persistence.
Designed to be lightweight and rebuildable from source documents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DocumentChunk:
    """A chunk of a document with metadata.

    Attributes:
        content: The text content of the chunk.
        source: Original source path or URL.
        filename: Name of the source file.
        chunk_id: Unique identifier for this chunk.
        document_id: Identifier for the parent document.
        timestamp: When the chunk was created/ingested.
        metadata: Additional metadata.
    """

    content: str
    source: str = ""
    filename: str = ""
    chunk_id: str = ""
    document_id: str = ""
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result of a retrieval query.

    Attributes:
        chunk: The retrieved document chunk.
        score: Similarity score (higher is better).
    """

    chunk: DocumentChunk
    score: float


class BaseVectorStore(ABC):
    """Abstract base class for vector stores.

    Implementations must provide:
        - add(): Insert embeddings with metadata.
        - search(): Similarity search.
        - save(): Persist to disk.
        - load(): Load from disk.
        - clear(): Remove all entries.
        - size(): Return number of stored vectors.
    """

    @abstractmethod
    def add(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Add document chunks with their embeddings.

        Args:
            chunks: List of DocumentChunk with metadata.
            embeddings: Corresponding embedding vectors.
        """
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Search for similar documents.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult sorted by similarity.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the vector store to disk.

        Args:
            path: Directory or file path to save to.
        """
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the vector store from disk.

        Args:
            path: Directory or file path to load from.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries from the vector store."""
        ...

    @abstractmethod
    def size(self) -> int:
        """Return the number of stored vectors."""
        ...
