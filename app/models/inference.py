"""KGK AI Local Transformers Provider.

Concrete implementation of BaseModelProvider using Hugging Face Transformers
for local model inference. Supports quantization (4-bit, 8-bit) and streaming.

This is the primary model provider for KGK AI v1. It can be replaced by
an API provider or other implementations without changing any other code.
"""

from __future__ import annotations

from typing import Any, Generator, Optional

from app.config import get_settings
from app.logging_config import get_logger
from app.models.base import (
    BaseModelProvider,
    GenerationParams,
    GenerationResult,
    ModelInfo,
)
from app.models.loader import get_model_context_length, load_model_and_tokenizer

logger = get_logger("models.inference")


class LocalTransformersProvider(BaseModelProvider):
    """Model provider using Hugging Face Transformers for local inference.

    Attributes:
        model_name: Hugging Face model ID.
        quantization: Quantization mode.
        device: Compute device.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        quantization: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.model_name: str = model_name or settings.model_name
        self.quantization: str = quantization or settings.quantization
        self.device: str = device or settings.device

        self._model: Any = None
        self._tokenizer: Any = None
        self._context_length: int = 32768
        self._loaded: bool = False

    def load(self) -> None:
        """Load the model and tokenizer into memory."""
        if self._loaded:
            logger.info("Model already loaded, skipping")
            return

        try:
            self._model, self._tokenizer = load_model_and_tokenizer(
                model_name=self.model_name,
                quantization=self.quantization,
                device=self.device,
            )
            self._context_length = get_model_context_length(self.model_name)
            self._loaded = True
            logger.info(
                f"Model loaded: {self.model_name} (context={self._context_length})",
                extra={"component": "models.inference", "model": self.model_name},
            )
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def unload(self) -> None:
        """Release the model from memory."""
        import gc

        self._model = None
        self._tokenizer = None
        self._loaded = False
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("Model unloaded from memory")

    def generate(
        self,
        messages: list[dict[str, str]],
        params: Optional[GenerationParams] = None,
    ) -> GenerationResult:
        """Generate a response from chat messages (non-streaming).

        Args:
            messages: List of {'role': ..., 'content': ...} dicts.
            params: Generation parameters. Uses defaults if None.

        Returns:
            GenerationResult with the complete generated text.
        """
        if not self._loaded:
            self.load()

        params = params or self._get_default_params()

        try:
            input_text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            import torch

            inputs = self._tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=self._context_length - params.max_new_tokens,
            ).to(self._model.device if hasattr(self._model, "device") else self.device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=params.max_new_tokens,
                    temperature=params.temperature if params.do_sample else 1.0,
                    top_p=params.top_p if params.do_sample else 1.0,
                    top_k=params.top_k if params.do_sample else 0,
                    repetition_penalty=params.repetition_penalty,
                    do_sample=params.do_sample,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[0][input_len:]
            text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

            finish_reason = "length" if len(generated_ids) >= params.max_new_tokens else "stop"

            logger.info(
                f"Generation complete: {len(generated_ids)} tokens, finish={finish_reason}",
                extra={"component": "models.inference", "tokens": len(generated_ids), "finish_reason": finish_reason},
            )

            return GenerationResult(text=text, finish_reason=finish_reason)

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return GenerationResult(
                text="KGK AI is temporarily unavailable. Please try again.",
                finish_reason="error",
                metadata={"error": str(e)},
            )

    def stream(
        self,
        messages: list[dict[str, str]],
        params: Optional[GenerationParams] = None,
    ) -> Generator[str, None, None]:
        """Stream a response token-by-token.

        Args:
            messages: List of {'role': ..., 'content': ...} dicts.
            params: Generation parameters. Uses defaults if None.

        Yields:
            Text chunks as they are generated.
        """
        if not self._loaded:
            self.load()

        params = params or self._get_default_params()

        try:
            from transformers import TextIteratorStreamer
            import torch

            input_text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self._tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=self._context_length - params.max_new_tokens,
            ).to(self._model.device if hasattr(self._model, "device") else self.device)

            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            generation_kwargs = {
                **inputs,
                "max_new_tokens": params.max_new_tokens,
                "temperature": params.temperature if params.do_sample else 1.0,
                "top_p": params.top_p if params.do_sample else 1.0,
                "top_k": params.top_k if params.do_sample else 0,
                "repetition_penalty": params.repetition_penalty,
                "do_sample": params.do_sample,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
                "streamer": streamer,
            }

            from threading import Thread

            thread = Thread(target=self._safe_generate, kwargs=generation_kwargs)
            thread.start()

            for text_chunk in streamer:
                if text_chunk:
                    yield text_chunk

            thread.join(timeout=30)

        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield "KGK AI is temporarily unavailable. Please try again."

    def _safe_generate(self, **kwargs: Any) -> None:
        """Wrapper for model.generate that catches errors in the thread."""
        try:
            import torch

            with torch.no_grad():
                self._model.generate(**kwargs)
        except Exception as e:
            logger.error(f"Error in generation thread: {e}")

    def health_check(self) -> bool:
        """Check if the model is loaded and operational.

        Returns:
            True if the model is ready for inference.
        """
        return self._loaded and self._model is not None and self._tokenizer is not None

    def get_info(self) -> ModelInfo:
        """Return metadata about the loaded model.

        Returns:
            ModelInfo instance with current model details.
        """
        return ModelInfo(
            name=self.model_name,
            provider="local_transformers",
            device=self.device,
            quantization=self.quantization,
            loaded=self._loaded,
            context_length=self._context_length,
        )

    def _get_default_params(self) -> GenerationParams:
        """Return GenerationParams from settings."""
        settings = get_settings()
        return GenerationParams(
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            top_k=settings.top_k,
            repetition_penalty=settings.repetition_penalty,
            do_sample=settings.temperature > 0,
        )
