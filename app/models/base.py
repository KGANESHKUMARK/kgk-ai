"""KGK AI Model Layer — Abstract base provider interface.

All model providers must implement this interface. The rest of KGK AI
never imports transformers or any specific model library directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generator, Optional


@dataclass
class GenerationParams:
    """Parameters controlling text generation.

    Attributes:
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0.0 = greedy).
        top_p: Nucleus sampling probability.
        top_k: Top-k sampling cutoff.
        repetition_penalty: Penalty for repeated tokens.
        do_sample: Whether to use sampling (vs greedy decoding).
    """

    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True


@dataclass
class GenerationResult:
    """Result of a generation call.

    Attributes:
        text: The generated text.
        finish_reason: Why generation stopped (e.g. 'stop', 'length').
        metadata: Additional provider-specific metadata.
    """

    text: str
    finish_reason: str = "stop"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInfo:
    """Information about a loaded model.

    Attributes:
        name: Model name or Hugging Face ID.
        provider: Provider type (e.g. 'local_transformers', 'api').
        device: Compute device (e.g. 'cuda', 'cpu').
        quantization: Quantization mode (e.g. 'none', '4bit', '8bit').
        loaded: Whether the model is currently loaded in memory.
        context_length: Maximum context window in tokens.
    """

    name: str = ""
    provider: str = ""
    device: str = ""
    quantization: str = "none"
    loaded: bool = False
    context_length: int = 32768


class BaseModelProvider(ABC):
    """Abstract base class for all model providers in KGK AI.

    Implementations must provide:
        - generate(): Non-streaming text generation.
        - stream(): Streaming text generation (generator).
        - embed(): Text embeddings (optional, raise NotImplementedError if unsupported).
        - health_check(): Verify the model is operational.
        - get_info(): Return model metadata.
        - load(): Load the model into memory.
        - unload(): Release the model from memory.
    """

    @abstractmethod
    def load(self) -> None:
        """Load the model and tokenizer into memory."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release the model from memory."""
        ...

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        params: Optional[GenerationParams] = None,
    ) -> GenerationResult:
        """Generate a response from a list of chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content'.
                      Example: [{"role": "user", "content": "Hello"}]
            params: Generation parameters. Uses defaults if None.

        Returns:
            GenerationResult containing the generated text.
        """
        ...

    @abstractmethod
    def stream(
        self,
        messages: list[dict[str, str]],
        params: Optional[GenerationParams] = None,
    ) -> Generator[str, None, None]:
        """Stream a response token-by-token.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            params: Generation parameters. Uses defaults if None.

        Yields:
            Text chunks as they are generated.
        """
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Not all providers support embeddings. Override if supported.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors.

        Raises:
            NotImplementedError: If the provider does not support embeddings.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support embeddings."
        )

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the model is loaded and operational.

        Returns:
            True if the model is ready for inference, False otherwise.
        """
        ...

    @abstractmethod
    def get_info(self) -> ModelInfo:
        """Return metadata about the loaded model.

        Returns:
            ModelInfo instance with current model details.
        """
        ...
