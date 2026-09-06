"""Unit tests for KGK AI chat controller and session manager.

Tests cover:
- Chat controller: input validation, no-model handling, conversation management,
  message history, memory integration, error handling
- Session manager: message sending, history retrieval, session clearing, listing
- Streaming: error handling, chunk yielding
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call

from app.chat.controller import KGKChatController, ChatResponseData
from app.chat.session import ChatSessionManager, get_session_manager
from app.chat.conversation import Conversation, Message
from app.models.base import BaseModelProvider, GenerationParams, GenerationResult, ModelInfo


class TestChatResponseData:
    """Tests for ChatResponseData dataclass."""

    def test_defaults(self):
        resp = ChatResponseData(text="Hello")
        assert resp.text == "Hello"
        assert resp.conversation_id == "default"
        assert resp.sources == []
        assert resp.tools_used == []
        assert resp.model == ""
        assert resp.latency_ms == 0.0
        assert resp.error == ""

    def test_with_metadata(self):
        resp = ChatResponseData(
            text="Hello",
            conversation_id="conv-123",
            sources=["doc1.txt"],
            tools_used=["calculator"],
            model="Qwen/Qwen3-4B",
            latency_ms=150.5,
            request_id="abc123",
        )
        assert resp.conversation_id == "conv-123"
        assert resp.sources == ["doc1.txt"]
        assert resp.request_id == "abc123"


class TestKGKChatControllerNoModel:
    """Tests for controller behavior when no model is loaded."""

    def test_chat_no_model_returns_message(self):
        """Controller returns graceful message when no model is available."""
        controller = KGKChatController(model=None)
        result = controller.chat("Hello")
        assert "not fully initialized" in result.text
        assert result.error == "model_not_loaded"
        assert result.latency_ms >= 0

    def test_stream_chat_no_model_yields_message(self):
        """Stream controller yields graceful message when no model is available."""
        controller = KGKChatController(model=None)
        chunks = list(controller.stream_chat("Hello"))
        assert len(chunks) > 0
        assert "not fully initialized" in chunks[0]


class TestKGKChatControllerValidation:
    """Tests for input validation."""

    def test_empty_message(self):
        controller = KGKChatController(model=None)
        result = controller.chat("   ")
        assert result.text == "Please enter a message."
        assert result.error == "empty_input"

    def test_empty_message_stream(self):
        controller = KGKChatController(model=None)
        chunks = list(controller.stream_chat("   "))
        assert chunks == ["Please enter a message."]

    def test_too_long_message(self):
        controller = KGKChatController(model=None)
        long_msg = "x" * 10001
        result = controller.chat(long_msg)
        assert "too long" in result.text
        assert result.error == "input_too_long"


class TestKGKChatControllerWithModel:
    """Tests for controller with a mocked model provider."""

    def _make_mock_model(self, response_text="Hello from KGK AI!"):
        """Create a mock model provider."""
        model = MagicMock(spec=BaseModelProvider)
        model.health_check.return_value = True
        model.generate.return_value = GenerationResult(
            text=response_text,
            finish_reason="stop",
        )
        model.stream.return_value = iter(["Hello ", "from ", "KGK AI!"])
        model.get_info.return_value = ModelInfo(
            name="test-model",
            provider="mock",
            device="cpu",
            quantization="none",
            loaded=True,
            context_length=32768,
        )
        return model

    def test_chat_with_model(self):
        """Controller generates response using the model."""
        model = self._make_mock_model()
        controller = KGKChatController(model=model)
        result = controller.chat("What is AI?")

        assert result.text == "Hello from KGK AI!"
        assert result.model == "test-model"
        assert result.error == ""
        assert result.latency_ms >= 0
        model.generate.assert_called_once()

    def test_chat_adds_system_prompt(self):
        """Controller adds system prompt to new conversations."""
        model = self._make_mock_model()
        controller = KGKChatController(model=model)
        controller.chat("Hello", conversation_id="test-conv")

        conv = controller.get_conversation("test-conv")
        assert conv is not None
        assert conv.messages[0].role == "system"
        assert "KGK AI" in conv.messages[0].content

    def test_chat_adds_user_and_assistant_messages(self):
        """Controller adds both user and assistant messages to conversation."""
        model = self._make_mock_model()
        controller = KGKChatController(model=model)
        controller.chat("Hello", conversation_id="test-conv")

        conv = controller.get_conversation("test-conv")
        roles = [m.role for m in conv.messages]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles

    def test_chat_preserves_conversation_history(self):
        """Multiple messages in same conversation maintain history."""
        model = self._make_mock_model()
        controller = KGKChatController(model=model)

        controller.chat("Hello", conversation_id="test-conv")
        controller.chat("How are you?", conversation_id="test-conv")

        conv = controller.get_conversation("test-conv")
        # system + user1 + assistant1 + user2 + assistant2 = 5
        assert conv.length == 5

    def test_chat_separate_conversations(self):
        """Different conversation IDs maintain separate histories."""
        model = self._make_mock_model()
        controller = KGKChatController(model=model)

        controller.chat("Hello", conversation_id="conv-1")
        controller.chat("Hi there", conversation_id="conv-2")

        conv1 = controller.get_conversation("conv-1")
        conv2 = controller.get_conversation("conv-2")
        assert conv1.length == 3  # system + user + assistant
        assert conv2.length == 3

    def test_chat_model_error_graceful(self):
        """Controller handles model errors gracefully."""
        model = MagicMock(spec=BaseModelProvider)
        model.health_check.return_value = True
        model.generate.side_effect = Exception("Model crashed")
        model.get_info.return_value = ModelInfo(name="test", loaded=True)

        controller = KGKChatController(model=model)
        result = controller.chat("Hello")

        assert "encountered an error" in result.text
        assert result.error == "generation_error"

    def test_stream_chat_with_model(self):
        """Stream controller yields chunks from model."""
        model = self._make_mock_model()
        controller = KGKChatController(model=model)
        chunks = list(controller.stream_chat("Hello", conversation_id="test-stream"))

        assert chunks == ["Hello ", "from ", "KGK AI!"]

        # Verify conversation was updated
        conv = controller.get_conversation("test-stream")
        assert conv is not None
        assistant_msgs = [m for m in conv.messages if m.role == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].content == "Hello from KGK AI!"

    def test_stream_chat_model_error(self):
        """Stream controller handles model errors gracefully."""
        model = MagicMock(spec=BaseModelProvider)
        model.health_check.return_value = True
        model.stream.side_effect = Exception("Stream failed")

        controller = KGKChatController(model=model)
        chunks = list(controller.stream_chat("Hello"))

        assert len(chunks) > 0
        assert "error" in chunks[-1].lower()

    def test_chat_with_memory(self):
        """Controller saves to memory when enabled."""
        model = self._make_mock_model()
        mock_memory = MagicMock()
        controller = KGKChatController(model=model, memory=mock_memory)

        controller.chat("Remember this", conversation_id="conv-mem")

        mock_memory.save_memory.assert_called_once()
        call_kwargs = mock_memory.save_memory.call_args
        assert call_kwargs.kwargs["conversation_id"] == "conv-mem"

    def test_chat_memory_error_doesnt_break(self):
        """Memory save errors don't break the chat response."""
        model = self._make_mock_model()
        mock_memory = MagicMock()
        mock_memory.save_memory.side_effect = Exception("Memory error")

        controller = KGKChatController(model=model, memory=mock_memory)
        result = controller.chat("Hello")

        assert result.text == "Hello from KGK AI!"
        assert result.error == ""


class TestKGKChatControllerManagement:
    """Tests for conversation management methods."""

    def test_get_conversation_not_found(self):
        controller = KGKChatController(model=None)
        assert controller.get_conversation("nonexistent") is None

    def test_clear_conversation(self):
        model = MagicMock(spec=BaseModelProvider)
        model.health_check.return_value = True
        model.generate.return_value = GenerationResult(text="Hi")
        model.get_info.return_value = ModelInfo(name="test", loaded=True)

        controller = KGKChatController(model=model)
        controller.chat("Hello", conversation_id="test-clear")

        assert controller.clear_conversation("test-clear") is True
        conv = controller.get_conversation("test-clear")
        assert conv.length == 0

    def test_clear_nonexistent_conversation(self):
        controller = KGKChatController(model=None)
        assert controller.clear_conversation("nonexistent") is False

    def test_list_conversations(self):
        model = MagicMock(spec=BaseModelProvider)
        model.health_check.return_value = True
        model.generate.return_value = GenerationResult(text="Hi")
        model.get_info.return_value = ModelInfo(name="test", loaded=True)

        controller = KGKChatController(model=model)
        controller.chat("Hello", conversation_id="conv-1")
        controller.chat("Hi", conversation_id="conv-2")

        convs = controller.list_conversations()
        assert "conv-1" in convs
        assert "conv-2" in convs

    def test_set_model(self):
        controller = KGKChatController(model=None)
        new_model = MagicMock(spec=BaseModelProvider)
        new_model.get_info.return_value = ModelInfo(name="new-model", loaded=True)

        controller.set_model(new_model)
        assert controller.model is new_model


class TestChatSessionManager:
    """Tests for ChatSessionManager."""

    def test_send_message(self):
        model = MagicMock(spec=BaseModelProvider)
        model.health_check.return_value = True
        model.generate.return_value = GenerationResult(text="Hello!")
        model.get_info.return_value = ModelInfo(name="test", loaded=True)

        controller = KGKChatController(model=model)
        session_mgr = ChatSessionManager(controller=controller)

        result = session_mgr.send_message("Hi", conversation_id="test-session")
        assert result.text == "Hello!"
        assert result.conversation_id == "test-session"

    def test_stream_message(self):
        model = MagicMock(spec=BaseModelProvider)
        model.health_check.return_value = True
        model.stream.return_value = iter(["chunk1", "chunk2"])
        model.get_info.return_value = ModelInfo(name="test", loaded=True)

        controller = KGKChatController(model=model)
        session_mgr = ChatSessionManager(controller=controller)

        chunks = list(session_mgr.stream_message("Hi", conversation_id="test-stream"))
        assert chunks == ["chunk1", "chunk2"]

    def test_get_history_empty(self):
        controller = KGKChatController(model=None)
        session_mgr = ChatSessionManager(controller=controller)
        history = session_mgr.get_history("nonexistent")
        assert history == []

    def test_get_history_after_chat(self):
        model = MagicMock(spec=BaseModelProvider)
        model.health_check.return_value = True
        model.generate.return_value = GenerationResult(text="Hello!")
        model.get_info.return_value = ModelInfo(name="test", loaded=True)

        controller = KGKChatController(model=model)
        session_mgr = ChatSessionManager(controller=controller)
        session_mgr.send_message("Hi", conversation_id="test-hist")

        history = session_mgr.get_history("test-hist")
        # Should exclude system message
        roles = [m["role"] for m in history]
        assert "user" in roles
        assert "assistant" in roles
        assert "system" not in roles

    def test_get_history_text(self):
        model = MagicMock(spec=BaseModelProvider)
        model.health_check.return_value = True
        model.generate.return_value = GenerationResult(text="Hello!")
        model.get_info.return_value = ModelInfo(name="test", loaded=True)

        controller = KGKChatController(model=model)
        session_mgr = ChatSessionManager(controller=controller)
        session_mgr.send_message("Hi", conversation_id="test-hist-text")

        history_text = session_mgr.get_history_text("test-hist-text")
        assert "Hi" in history_text
        assert "Hello!" in history_text

    def test_clear_session(self):
        model = MagicMock(spec=BaseModelProvider)
        model.health_check.return_value = True
        model.generate.return_value = GenerationResult(text="Hi")
        model.get_info.return_value = ModelInfo(name="test", loaded=True)

        controller = KGKChatController(model=model)
        session_mgr = ChatSessionManager(controller=controller)
        session_mgr.send_message("Hello", conversation_id="test-clear")

        assert session_mgr.clear_session("test-clear") is True
        assert session_mgr.clear_session("nonexistent") is False

    def test_list_sessions(self):
        model = MagicMock(spec=BaseModelProvider)
        model.health_check.return_value = True
        model.generate.return_value = GenerationResult(text="Hi")
        model.get_info.return_value = ModelInfo(name="test", loaded=True)

        controller = KGKChatController(model=model)
        session_mgr = ChatSessionManager(controller=controller)
        session_mgr.send_message("Hello", conversation_id="s1")
        session_mgr.send_message("Hi", conversation_id="s2")

        sessions = session_mgr.list_sessions()
        assert "s1" in sessions
        assert "s2" in sessions

    def test_session_exists(self):
        controller = KGKChatController(model=None)
        session_mgr = ChatSessionManager(controller=controller)

        assert session_mgr.session_exists("nonexistent") is False

    def test_session_length(self):
        controller = KGKChatController(model=None)
        session_mgr = ChatSessionManager(controller=controller)

        assert session_mgr.session_length("nonexistent") == 0

    def test_session_length_after_chat(self):
        model = MagicMock(spec=BaseModelProvider)
        model.health_check.return_value = True
        model.generate.return_value = GenerationResult(text="Hi")
        model.get_info.return_value = ModelInfo(name="test", loaded=True)

        controller = KGKChatController(model=model)
        session_mgr = ChatSessionManager(controller=controller)
        session_mgr.send_message("Hello", conversation_id="test-len")

        # system + user + assistant = 3
        assert session_mgr.session_length("test-len") == 3


class TestGetSessionManager:
    """Tests for the singleton session manager."""

    def test_singleton(self):
        mgr1 = get_session_manager()
        mgr2 = get_session_manager()
        assert mgr1 is mgr2
