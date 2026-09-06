"""Unit tests for KGK AI UI module.

Tests cover:
- Model info text generation
- About text generation
- Chat streaming function
- Chat non-streaming function
- Clear conversation function
- UI creation (requires gradio installed)
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from app.ui.gradio_app import (
    _get_model_info_text,
    _get_about_text,
    chat_stream,
    chat_respond,
    clear_conversation,
)


class TestModelInfoText:
    """Tests for model info markdown generation."""

    def test_contains_model_name(self):
        text = _get_model_info_text()
        assert "Qwen/Qwen3-4B-Instruct-2507" in text

    def test_contains_quantization(self):
        text = _get_model_info_text()
        assert "4bit" in text

    def test_contains_features(self):
        text = _get_model_info_text()
        assert "RAG" in text
        assert "Memory" in text
        assert "Tools" in text

    def test_contains_device(self):
        text = _get_model_info_text()
        assert "Device" in text


class TestAboutText:
    """Tests for about text generation."""

    def test_contains_title(self):
        text = _get_about_text()
        assert "KGK AI" in text

    def test_contains_company(self):
        text = _get_about_text()
        assert "Siddharitha" in text

    def test_contains_open_source_note(self):
        text = _get_about_text()
        assert "open-source" in text.lower()
        assert "foundation model" in text.lower()

    def test_contains_usage_instructions(self):
        text = _get_about_text()
        assert "How to use" in text


class TestChatStream:
    """Tests for streaming chat function."""

    def test_stream_yields_text(self):
        """chat_stream yields accumulated text chunks."""
        with patch("app.ui.gradio_app.get_session_manager") as mock_get:
            session_mgr = MagicMock()
            session_mgr.stream_message.return_value = iter(["Hello ", "world!"])
            mock_get.return_value = session_mgr

            chunks = list(chat_stream("Hi", [], "test-conv"))

        assert len(chunks) == 2
        assert chunks[0] == "Hello "
        assert chunks[1] == "Hello world!"

    def test_stream_empty_message(self):
        """chat_stream handles empty messages gracefully."""
        with patch("app.ui.gradio_app.get_session_manager") as mock_get:
            session_mgr = MagicMock()
            session_mgr.stream_message.return_value = iter(["Please enter a message."])
            mock_get.return_value = session_mgr

            chunks = list(chat_stream("   ", [], "test-conv"))

        assert len(chunks) > 0

    def test_stream_error_handling(self):
        """chat_stream handles errors gracefully."""
        with patch("app.ui.gradio_app.get_session_manager") as mock_get:
            session_mgr = MagicMock()
            session_mgr.stream_message.side_effect = Exception("Stream error")
            mock_get.return_value = session_mgr

            chunks = list(chat_stream("Hi", [], "test-conv"))

        assert len(chunks) > 0
        assert "Error" in chunks[-1]


class TestChatRespond:
    """Tests for non-streaming chat function."""

    def test_respond_returns_text(self):
        with patch("app.ui.gradio_app.get_session_manager") as mock_get:
            session_mgr = MagicMock()
            session_mgr.send_message.return_value = MagicMock(text="Hello from KGK!")
            mock_get.return_value = session_mgr

            result = chat_respond("Hi", [], "test-conv")

        assert result == "Hello from KGK!"


class TestClearConversation:
    """Tests for clear conversation function."""

    def test_clear_success(self):
        with patch("app.ui.gradio_app.get_session_manager") as mock_get:
            session_mgr = MagicMock()
            session_mgr.clear_session.return_value = True
            mock_get.return_value = session_mgr

            result = clear_conversation("test-conv")

        assert "cleared" in result.lower()

    def test_clear_not_found(self):
        with patch("app.ui.gradio_app.get_session_manager") as mock_get:
            session_mgr = MagicMock()
            session_mgr.clear_session.return_value = False
            mock_get.return_value = session_mgr

            result = clear_conversation("nonexistent")

        assert "No conversation" in result


class TestCreateUI:
    """Tests for Gradio UI creation."""

    def test_create_ui_returns_blocks(self):
        """create_ui returns a Gradio Blocks application."""
        gradio = pytest.importorskip("gradio")
        from app.ui.gradio_app import create_ui

        demo = create_ui()
        assert demo is not None
        # Gradio Blocks has a .blocks attribute
        assert hasattr(demo, "launch")
