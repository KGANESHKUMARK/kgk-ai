"""KGK AI RAG package — document ingestion, chunking, embeddings, and retrieval."""

from app.rag.chunker import TextChunker
from app.rag.embedder import EmbeddingProvider
from app.rag.faiss_store import FaissVectorStore
from app.rag.loader import DocumentLoader, LoadedDocument
from app.rag.pipeline import RAGPipeline, RAGResponse, get_rag_pipeline
from app.rag.vector_store import BaseVectorStore, DocumentChunk, RetrievalResult

__all__ = [
    "TextChunker",
    "EmbeddingProvider",
    "FaissVectorStore",
    "DocumentLoader",
    "LoadedDocument",
    "RAGPipeline",
    "RAGResponse",
    "get_rag_pipeline",
    "BaseVectorStore",
    "DocumentChunk",
    "RetrievalResult",
]
