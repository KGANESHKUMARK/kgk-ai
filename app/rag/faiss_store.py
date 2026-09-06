"""KGK AI FAISS Vector Store — Concrete implementation of BaseVectorStore.

Uses FAISS for fast similarity search. Stores document chunks and their
embeddings with full metadata. Supports persistence to disk.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from app.config import get_settings
from app.logging_config import get_logger
from app.rag.vector_store import BaseVectorStore, DocumentChunk, RetrievalResult

logger = get_logger("rag.faiss_store")


class FaissVectorStore(BaseVectorStore):
    """FAISS-based vector store for KGK AI RAG.

    Stores embeddings in a FAISS index and document metadata in a
    parallel list. Supports L2 (flat) similarity search.

    Attributes:
        dimension: Embedding vector dimension.
    """

    def __init__(self, dimension: Optional[int] = None) -> None:
        settings = get_settings()
        self.dimension: int = dimension or settings.embedding_dimension
        self._index: Any = None
        self._chunks: list[DocumentChunk] = []
        self._vectors: list[list[float]] = []

    def _ensure_index(self) -> None:
        """Create the FAISS index if it doesn't exist."""
        if self._index is not None:
            return

        try:
            import faiss
            import numpy as np

            self._index = faiss.IndexFlatL2(self.dimension)
        except ImportError:
            logger.error(
                "faiss-cpu not installed. Install with: pip install faiss-cpu"
            )
            raise

    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add document chunks with their embeddings.

        Args:
            chunks: List of DocumentChunk with metadata.
            embeddings: Corresponding embedding vectors.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        if not chunks:
            return

        self._ensure_index()

        import numpy as np

        vectors = np.array(embeddings, dtype=np.float32)
        self._index.add(vectors)
        self._chunks.extend(chunks)
        self._vectors.extend(embeddings)

        logger.info(
            f"Added {len(chunks)} chunks to vector store (total: {len(self._chunks)})",
            extra={"component": "rag.faiss_store", "added": len(chunks), "total": len(self._chunks)},
        )

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
            List of RetrievalResult sorted by similarity (best first).
        """
        if not self._chunks or self._index is None:
            return []

        import numpy as np

        query = np.array([query_embedding], dtype=np.float32)
        k = min(top_k, len(self._chunks))

        scores, indices = self._index.search(query, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            score = 1.0 / (1.0 + float(scores[0][i]))  # Convert L2 distance to similarity
            results.append(RetrievalResult(
                chunk=self._chunks[idx],
                score=score,
            ))

        return results

    def save(self, path: str) -> None:
        """Persist the vector store to disk.

        Saves FAISS index and chunk metadata as separate files.

        Args:
            path: Directory path to save to.
        """
        if self._index is None:
            logger.warning("No index to save")
            return

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            import faiss

            index_path = save_path / "kgk_index.faiss"
            faiss.write_index(self._index, str(index_path))

            metadata_path = save_path / "kgk_chunks.json"
            chunks_data = [
                {
                    "content": c.content,
                    "source": c.source,
                    "filename": c.filename,
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "timestamp": c.timestamp,
                    "metadata": c.metadata,
                }
                for c in self._chunks
            ]
            metadata_path.write_text(
                json.dumps(chunks_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            logger.info(
                f"Vector store saved to {save_path} ({len(self._chunks)} chunks)",
                extra={"component": "rag.faiss_store", "path": str(save_path), "chunks": len(self._chunks)},
            )
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise

    def load(self, path: str) -> None:
        """Load the vector store from disk.

        Args:
            path: Directory path to load from.
        """
        load_path = Path(path)

        index_path = load_path / "kgk_index.faiss"
        metadata_path = load_path / "kgk_chunks.json"

        if not index_path.exists() or not metadata_path.exists():
            logger.warning(f"Vector store files not found in {load_path}")
            return

        try:
            import faiss

            self._index = faiss.read_index(str(index_path))

            chunks_data = json.loads(metadata_path.read_text(encoding="utf-8"))
            self._chunks = [
                DocumentChunk(
                    content=c["content"],
                    source=c.get("source", ""),
                    filename=c.get("filename", ""),
                    chunk_id=c.get("chunk_id", ""),
                    document_id=c.get("document_id", ""),
                    timestamp=c.get("timestamp", ""),
                    metadata=c.get("metadata", {}),
                )
                for c in chunks_data
            ]

            logger.info(
                f"Vector store loaded from {load_path} ({len(self._chunks)} chunks)",
                extra={"component": "rag.faiss_store", "path": str(load_path), "chunks": len(self._chunks)},
            )
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise

    def clear(self) -> None:
        """Remove all entries from the vector store."""
        self._ensure_index()
        self._index.reset()
        self._chunks.clear()
        self._vectors.clear()
        logger.info("Vector store cleared")

    def size(self) -> int:
        """Return the number of stored vectors."""
        return len(self._chunks)

    def get_chunks(self) -> list[DocumentChunk]:
        """Return all stored chunks (for inspection/debugging).

        Returns:
            List of all DocumentChunk instances.
        """
        return list(self._chunks)
