"""Unit tests for KGK AI model layer.

Tests cover:
- LocalTransformersProvider initialization and configuration
- ModelInfo and GenerationParams data structures
- ModelRegistry registration and selection
- Health check behavior
- Graceful error handling
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.requires_torch

from unittest.mock import MagicMock, patch, PropertyMock

from app.models.base import (
    BaseModelProvider,
    GenerationParams,
    GenerationResult,
    ModelInfo,
)
from app.models.registry import ModelRegistry
from app.models.inference import LocalTransformersProvider


class TestLocalTransformersProviderInit:
    """Tests for LocalTransformersProvider initialization."""

    def test_default_init(self):
        """Verify provider initializes with settings defaults."""
        provider = LocalTransformersProvider()
        assert provider.model_name == "Qwen/Qwen3-4B-Instruct-2507"
        assert provider.quantization == "4bit"
        assert provider.device in ("cuda", "cpu")
        assert provider._loaded is False

    def test_custom_init(self):
        """Verify provider accepts custom parameters."""
        provider = LocalTransformersProvider(
            model_name="Qwen/Qwen3-0.6B",
            quantization="none",
            device="cpu",
        )
        assert provider.model_name == "Qwen/Qwen3-0.6B"
        assert provider.quantization == "none"
        assert provider.device == "cpu"

    def test_health_check_not_loaded(self):
        """Health check returns False when model is not loaded."""
        provider = LocalTransformersProvider(device="cpu")
        assert provider.health_check() is False

    def test_get_info_not_loaded(self):
        """get_info returns correct metadata when not loaded."""
        provider = LocalTransformersProvider(
            model_name="test-model",
            quantization="4bit",
            device="cpu",
        )
        info = provider.get_info()
        assert info.name == "test-model"
        assert info.provider == "local_transformers"
        assert info.device == "cpu"
        assert info.quantization == "4bit"
        assert info.loaded is False

    def test_get_info_loaded(self):
        """get_info returns loaded=True when model is loaded."""
        provider = LocalTransformersProvider(device="cpu")
        provider._loaded = True
        provider._context_length = 32768
        info = provider.get_info()
        assert info.loaded is True
        assert info.context_length == 32768


class TestLocalTransformersProviderUnload:
    """Tests for unloading the model."""

    def test_unload_clears_state(self):
        """Unload resets model state."""
        provider = LocalTransformersProvider(device="cpu")
        provider._loaded = True
        provider._model = MagicMock()
        provider._tokenizer = MagicMock()

        provider.unload()

        assert provider._loaded is False
        assert provider._model is None
        assert provider._tokenizer is None
        assert provider.health_check() is False


class TestLocalTransformersProviderGenerate:
    """Tests for generation with mocked model."""

    def test_generate_not_loaded_auto_loads(self):
        """Generate should auto-load if not loaded."""
        provider = LocalTransformersProvider(device="cpu")

        with patch.object(provider, "load") as mock_load:
            with patch.object(provider, "_tokenizer") as mock_tok:
                mock_tok.apply_chat_template.return_value = "test"
                mock_tok.return_value = {"input_ids": MagicMock()}
                mock_tok.pad_token_id = 0
                mock_tok.eos_token_id = 1

                with patch.object(provider, "_model") as mock_model:
                    mock_output = MagicMock()
                    mock_output.__getitem__ = MagicMock(return_value=MagicMock())
                    mock_output[0].__getitem__ = MagicMock(return_value=[1, 2, 3])
                    mock_model.generate.return_value = mock_output
                    mock_model.device = "cpu"

                    with patch("torch.no_grad"):
                        result = provider.generate([{"role": "user", "content": "hi"}])

        mock_load.assert_called_once()

    def test_generate_returns_error_result_on_failure(self):
        """Generate returns graceful error on exception."""
        provider = LocalTransformersProvider(device="cpu")
        provider._loaded = True
        provider._tokenizer = MagicMock()
        provider._tokenizer.apply_chat_template.side_effect = Exception("Tokenizer error")

        result = provider.generate([{"role": "user", "content": "hi"}])

        assert isinstance(result, GenerationResult)
        assert "temporarily unavailable" in result.text
        assert result.finish_reason == "error"

    def test_generate_uses_custom_params(self):
        """Generate respects custom GenerationParams."""
        provider = LocalTransformersProvider(device="cpu")
        provider._loaded = True

        params = GenerationParams(max_new_tokens=100, temperature=0.5)

        with patch.object(provider, "_tokenizer") as mock_tok:
            mock_tok.apply_chat_template.return_value = "test"
            mock_input = MagicMock()
            mock_tok.return_value = mock_input
            mock_tok.pad_token_id = 0
            mock_tok.eos_token_id = 1

            with patch.object(provider, "_model") as mock_model:
                mock_output = MagicMock()
                mock_output[0] = MagicMock()
                mock_model.generate.return_value = mock_output
                mock_model.device = "cpu"

                with patch("torch.no_grad"):
                    provider.generate([{"role": "user", "content": "hi"}], params)

                call_kwargs = mock_model.generate.call_args
                assert call_kwargs.kwargs["max_new_tokens"] == 100


class TestLocalTransformersProviderStream:
    """Tests for streaming generation."""

    def test_stream_not_loaded_auto_loads(self):
        """Stream should auto-load if not loaded.

        The transformers import inside stream() will fail (caught by except),
        but load() must have been called first.
        """
        provider = LocalTransformersProvider(device="cpu")

        with patch.object(provider, "load") as mock_load:
            chunks = list(provider.stream([{"role": "user", "content": "hi"}]))

        mock_load.assert_called_once()
        assert len(chunks) > 0  # Error message yielded

    def test_stream_yields_error_on_failure(self):
        """Stream yields error message on exception."""
        provider = LocalTransformersProvider(device="cpu")
        provider._loaded = True
        provider._tokenizer = MagicMock()
        provider._tokenizer.apply_chat_template.side_effect = Exception("Stream error")

        chunks = list(provider.stream([{"role": "user", "content": "hi"}]))

        assert len(chunks) > 0
        assert "temporarily unavailable" in chunks[0]


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_register_and_list(self):
        reg = ModelRegistry()
        reg.register("local", LocalTransformersProvider)
        assert "local" in reg.list_providers()

    def test_get_provider_class(self):
        reg = ModelRegistry()
        reg.register("local", LocalTransformersProvider)
        cls = reg.get_provider_class("local")
        assert cls is LocalTransformersProvider

    def test_get_provider_class_not_found(self):
        reg = ModelRegistry()
        assert reg.get_provider_class("nonexistent") is None

    def test_set_active_and_active_property(self):
        reg = ModelRegistry()
        provider = LocalTransformersProvider(device="cpu")
        reg.set_active(provider, "local")
        assert reg.active is provider
        assert reg.active_name == "local"

    def test_is_ready_false_when_no_provider(self):
        reg = ModelRegistry()
        assert reg.is_ready() is False

    def test_is_ready_false_when_not_healthy(self):
        reg = ModelRegistry()
        provider = LocalTransformersProvider(device="cpu")
        reg.set_active(provider)
        assert reg.is_ready() is False  # Not loaded

    def test_is_ready_true_when_healthy(self):
        reg = ModelRegistry()
        provider = LocalTransformersProvider(device="cpu")
        provider._loaded = True
        provider._model = MagicMock()
        provider._tokenizer = MagicMock()
        reg.set_active(provider)
        assert reg.is_ready() is True

    def test_get_info_returns_none_when_no_provider(self):
        reg = ModelRegistry()
        assert reg.get_info() is None

    def test_get_info_returns_info_when_active(self):
        reg = ModelRegistry()
        provider = LocalTransformersProvider(device="cpu")
        reg.set_active(provider)
        info = reg.get_info()
        assert info is not None
        assert info.provider == "local_transformers"
