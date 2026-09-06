"""KGK AI Document Search Tool — Search the KGK knowledge base via RAG.

Uses the RAG pipeline to retrieve relevant document chunks for a query.
Returns formatted results with source citations.
"""

from __future__ import annotations

from typing import Any, Optional

from app.logging_config import get_logger
from app.tools.registry import BaseTool, ToolInfo, ToolResult

logger = get_logger("tools.doc_search")


class DocumentSearchTool(BaseTool):
    """Search the KGK knowledge base using the RAG pipeline.

    Retrieves relevant document chunks and returns them formatted with
    source citations. Requires the RAG pipeline to be initialized.
    """

    def __init__(self, pipeline=None) -> None:
        self._pipeline = pipeline

    def _get_pipeline(self):
        """Get or lazily initialize the RAG pipeline."""
        if self._pipeline is not None:
            return self._pipeline
        from app.rag.pipeline import get_rag_pipeline
        self._pipeline = get_rag_pipeline()
        return self._pipeline

    def info(self) -> ToolInfo:
        return ToolInfo(
            name="document_search",
            description="Search the KGK knowledge base for relevant information. Returns document excerpts with source citations.",
            parameters={
                "query": "Search query string",
                "top_k": "Number of results to return (default: 5)",
            },
            safe_for_public=True,
        )

    def execute(self, query: str = "", top_k: int = 5, **kwargs: Any) -> ToolResult:
        """Search the knowledge base for relevant documents.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            ToolResult with formatted search results.
        """
        if not query or not query.strip():
            return ToolResult(success=False, error="No search query provided.")

        try:
            pipeline = self._get_pipeline()

            if not pipeline.is_ready():
                return ToolResult(
                    success=False,
                    error="Knowledge base is not initialized. Run ingestion first: python scripts/ingest.py",
                )

            response = pipeline.retrieve(query, top_k=top_k)

            if not response.chunks:
                return ToolResult(
                    success=True,
                    output="No relevant documents found for your query.",
                    metadata={"query": query, "results": 0},
                )

            # Format results
            parts = []
            for i, (chunk, score) in enumerate(zip(response.chunks, response.scores), 1):
                source = chunk.filename or chunk.source or "unknown"
                parts.append(
                    f"[Result {i} | Source: {source} | Score: {score:.3f}]\n{chunk.content}"
                )

            output = "\n\n---\n\n".join(parts)

            logger.info(
                f"Document search: query='{query[:50]}...', results={len(response.chunks)}",
                extra={
                    "component": "tools.doc_search",
                    "query_length": len(query),
                    "results": len(response.chunks),
                },
            )

            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "query": query,
                    "results": len(response.chunks),
                    "sources": response.sources,
                    "scores": response.scores,
                },
            )

        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return ToolResult(success=False, error=f"Search failed: {e}")
