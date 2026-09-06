"""KGK AI Text Chunker — Splits documents into embeddable chunks.

Uses a simple word-count-based chunking strategy with configurable
chunk size and overlap. Designed to be lightweight and dependency-free.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from app.config import get_settings
from app.logging_config import get_logger
from app.rag.loader import LoadedDocument
from app.rag.vector_store import DocumentChunk

logger = get_logger("rag.chunker")


class TextChunker:
    """Splits documents into chunks for embedding.

    Chunking strategy:
        1. Split text into paragraphs (double newlines)
        2. Group paragraphs until chunk_size words is reached
        3. Overlap chunks by chunk_overlap words
        4. Assign unique chunk IDs and metadata

    Attributes:
        chunk_size: Target chunk size in words.
        chunk_overlap: Overlap between chunks in words.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self.chunk_size: int = chunk_size or settings.chunk_size
        self.chunk_overlap: int = chunk_overlap or settings.chunk_overlap

    def chunk_document(self, document: LoadedDocument) -> list[DocumentChunk]:
        """Split a loaded document into chunks.

        Args:
            document: LoadedDocument to chunk.

        Returns:
            List of DocumentChunk instances.
        """
        if not document.content.strip():
            return []

        text = document.content
        words = text.split()

        if len(words) <= self.chunk_size:
            # Document is small enough to be a single chunk
            return [self._make_chunk(text, document, 0)]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunk = self._make_chunk(chunk_text, document, chunk_index)
            chunks.append(chunk)

            chunk_index += 1
            start += self.chunk_size - self.chunk_overlap

            if end >= len(words):
                break

        logger.info(
            f"Chunked {document.filename}: {len(chunks)} chunks",
            extra={
                "component": "rag.chunker",
                "filename": document.filename,
                "chunks": len(chunks),
                "chunk_size": self.chunk_size,
            },
        )

        return chunks

    def chunk_documents(
        self, documents: list[LoadedDocument]
    ) -> list[DocumentChunk]:
        """Chunk multiple documents.

        Args:
            documents: List of LoadedDocument instances.

        Returns:
            List of all DocumentChunk instances from all documents.
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        logger.info(
            f"Total chunks from {len(documents)} documents: {len(all_chunks)}",
            extra={"component": "rag.chunker", "total_chunks": len(all_chunks), "documents": len(documents)},
        )
        return all_chunks

    def chunk_text(
        self,
        text: str,
        source: str = "",
        filename: str = "",
    ) -> list[DocumentChunk]:
        """Chunk raw text without a LoadedDocument.

        Args:
            text: Raw text to chunk.
            source: Optional source path.
            filename: Optional filename.

        Returns:
            List of DocumentChunk instances.
        """
        doc = LoadedDocument(
            content=text,
            source=source,
            filename=filename,
            document_id=filename or "inline",
        )
        return self.chunk_document(doc)

    def _make_chunk(
        self,
        text: str,
        document: LoadedDocument,
        chunk_index: int,
    ) -> DocumentChunk:
        """Create a DocumentChunk with metadata.

        Args:
            text: Chunk text content.
            document: Parent document.
            chunk_index: Index of this chunk within the document.

        Returns:
            DocumentChunk instance.
        """
        chunk_id = f"{document.document_id}_{chunk_index:04d}"
        return DocumentChunk(
            content=text,
            source=document.source,
            filename=document.filename,
            chunk_id=chunk_id,
            document_id=document.document_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                "chunk_index": chunk_index,
                "word_count": len(text.split()),
                "file_type": document.file_type,
            },
        )
