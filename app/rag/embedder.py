"""KGK AI Embedding Provider — Sentence-transformers wrapper.

Provides text embeddings using sentence-transformers models.
The default model is all-MiniLM-L6-v2 (384 dimensions, ~90 MB).
"""

from __future__ import annotations

from typing import Any, Optional

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger("rag.embedder")


class EmbeddingProvider:
    """Embedding provider using sentence-transformers.

    Attributes:
        model_name: Hugging Face model ID for the embedding model.
        dimension: Embedding vector dimension.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self.model_name: str = model_name or settings.embedding_model
        self.dimension: int = dimension or settings.embedding_dimension
        self._model: Any = None
        self._loaded: bool = False

    def load(self) -> None:
        """Load the embedding model into memory."""
        if self._loaded:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(
                f"Loading embedding model: {self.model_name}",
                extra={"component": "rag.embedder", "model": self.model_name},
            )
            self._model = SentenceTransformer(self.model_name)
            self._loaded = True
            logger.info("Embedding model loaded")
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def unload(self) -> None:
        """Release the embedding model from memory."""
        self._model = None
        self._loaded = False
        logger.info("Embedding model unloaded")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors (each a list of floats).

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self._loaded:
            self.load()

        if not texts:
            return []

        try:
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single query text.

        Args:
            text: Query string.

        Returns:
            Embedding vector as a list of floats.
        """
        results = self.embed([text])
        return results[0] if results else []

    def is_loaded(self) -> bool:
        """Check if the embedding model is loaded.

        Returns:
            True if the model is loaded and ready.
        """
        return self._loaded

    def get_info(self) -> dict[str, Any]:
        """Return embedding model metadata.

        Returns:
            Dict with model name, dimension, and loaded status.
        """
        return {
            "model": self.model_name,
            "dimension": self.dimension,
            "loaded": self._loaded,
        }
