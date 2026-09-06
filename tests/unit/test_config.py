"""Unit tests for KGK AI configuration."""

import os
import pytest
from app.config import KGKSettings, get_settings


class TestKGKSettings:
    """Tests for the KGKSettings configuration class."""

    def test_default_settings(self):
        """Verify default values are set correctly."""
        settings = KGKSettings()
        assert settings.model_name == "Qwen/Qwen3-4B-Instruct-2507"
        assert settings.model_name_fallback == "Qwen/Qwen3-0.6B"
        assert settings.max_new_tokens == 1024
        assert settings.temperature == 0.7
        assert settings.top_p == 0.9
        assert settings.quantization == "4bit"
        assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert settings.embedding_dimension == 384
        assert settings.enable_rag is True
        assert settings.enable_memory is True
        assert settings.enable_tools is True
        assert settings.enable_agents is False
        assert settings.ui_title == "KGK AI"
        assert settings.ui_subtitle == "AI Intelligence by KGK"

    def test_env_override(self, monkeypatch):
        """Verify environment variables override defaults."""
        monkeypatch.setenv("MODEL_NAME", "Qwen/Qwen3-0.6B")
        monkeypatch.setenv("TEMPERATURE", "0.5")
        monkeypatch.setenv("ENABLE_RAG", "false")

        settings = KGKSettings()
        assert settings.model_name == "Qwen/Qwen3-0.6B"
        assert settings.temperature == 0.5
        assert settings.enable_rag is False

    def test_get_settings_cached(self):
        """Verify get_settings returns cached instance."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_device_property(self):
        """Verify device property returns a valid string."""
        settings = KGKSettings()
        device = settings.device
        assert device in ("cuda", "cpu")

    def test_is_hf_space_property(self):
        """Verify is_hf_space detection."""
        settings = KGKSettings()
        assert isinstance(settings.is_hf_space, bool)

    def test_max_input_length_validation(self):
        """Verify max_input_length has a minimum."""
        settings = KGKSettings()
        assert settings.max_input_length >= 100
