"""KGK AI Model Loader — Handles model loading, quantization, and device placement.

This module isolates all transformers-specific logic here. The rest of KGK AI
interacts only through BaseModelProvider interfaces.
"""

from __future__ import annotations

from typing import Any, Optional

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger("models.loader")


def load_model_and_tokenizer(
    model_name: Optional[str] = None,
    quantization: Optional[str] = None,
    device: Optional[str] = None,
) -> tuple[Any, Any]:
    """Load a Hugging Face model and tokenizer with quantization support.

    Args:
        model_name: Hugging Face model ID. Defaults to settings.
        quantization: Quantization mode ('none', '4bit', '8bit'). Defaults to settings.
        device: Target device ('cuda', 'cpu'). Defaults to settings.

    Returns:
        Tuple of (model, tokenizer).

    Raises:
        ImportError: If transformers/torch are not installed.
        Exception: If model loading fails.
    """
    settings = get_settings()
    model_name = model_name or settings.model_name
    quantization = quantization or settings.quantization
    device = device or settings.device

    logger.info(
        f"Loading model: {model_name} (quantization={quantization}, device={device})",
        extra={"component": "models.loader", "model": model_name, "quantization": quantization, "device": device},
    )

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        logger.error(f"Required libraries not installed: {e}")
        raise ImportError(
            "transformers and torch are required for model loading. "
            "Install with: pip install transformers torch"
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=settings.hf_token,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = _get_quantization_config(quantization)

    if quantization_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            token=settings.hf_token,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            token=settings.hf_token,
        )

    if device == "cpu" and quantization == "none":
        model = model.to("cpu")

    model.eval()

    logger.info(
        f"Model loaded successfully: {model_name}",
        extra={"component": "models.loader", "model": model_name},
    )

    return model, tokenizer


def _get_quantization_config(quantization: str) -> Any:
    """Build quantization configuration for the specified mode.

    Args:
        quantization: One of 'none', '4bit', '8bit'.

    Returns:
        Quantization config object or None.
    """
    if quantization == "none":
        return None

    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        logger.warning(
            "BitsAndBytesConfig not available. Install bitsandbytes for quantization. "
            "Falling back to no quantization."
        )
        return None

    if quantization == "4bit":
        logger.info("Using 4-bit NF4 quantization")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
        )

    if quantization == "8bit":
        logger.info("Using 8-bit quantization")
        return BitsAndBytesConfig(load_in_8bit=True)

    logger.warning(f"Unknown quantization mode: {quantization}. Using none.")
    return None


def get_model_context_length(model_name: str) -> int:
    """Get the context length for a model from its config.

    Args:
        model_name: Hugging Face model ID.

    Returns:
        Context length in tokens (default 32768 if not found).
    """
    try:
        from transformers import AutoConfig

        settings = get_settings()
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=settings.hf_token,
        )
        return getattr(config, "max_position_embeddings", 32768)
    except Exception as e:
        logger.warning(f"Could not determine context length for {model_name}: {e}")
        return 32768
