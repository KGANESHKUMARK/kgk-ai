"""Unit tests for KGK AI memory components.

Tests cover:
- InMemoryStore: save, retrieve, delete, clear, concurrency
- FileMemoryStore: persistence, retrieve, delete, clear
- ContextWindowManager: truncation, token estimation, summarization integration
- ConversationSummarizer: extractive summary, model-based summary, edge cases
- MemoryManager integration with stores
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock

import pytest

from app.memory.manager import MemoryManager, MemoryItem
from app.memory.stores import InMemoryStore, FileMemoryStore
from app.memory.context_window import (
    ContextWindowManager,
    TruncationResult,
    estimate_tokens,
    estimate_message_tokens,
)
from app.memory.summarizer import ConversationSummarizer
from app.chat.conversation import Conversation, Message


class TestEstimateTokens:
    """Tests for token estimation utilities."""

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_string(self):
        assert estimate_tokens("hello") == 1

    def test_long_string(self):
        text = "a" * 100
        assert estimate_tokens(text) == 25

    def test_custom_chars_per_token(self):
        assert estimate_tokens("aaaa", chars_per_token=2) == 2

    def test_message_tokens_includes_overhead(self):
        msg = Message(role="user", content="hello")
        tokens = estimate_message_tokens(msg)
        assert tokens == 5  # 1 for content + 4 overhead


class TestInMemoryStore:
    """Tests for InMemoryStore."""

    def test_save_and_retrieve(self):
        store = InMemoryStore()
        item = MemoryItem(content="test memory")
        store.save("conv1", item)

        results = store.retrieve("conv1")
        assert len(results) == 1
        assert results[0].content == "test memory"

    def test_retrieve_nonexistent_key(self):
        store = InMemoryStore()
        results = store.retrieve("nonexistent")
        assert results == []

    def test_retrieve_with_limit(self):
        store = InMemoryStore()
        for i in range(15):
            store.save("conv1", MemoryItem(content=f"item {i}"))

        results = store.retrieve("conv1", limit=5)
        assert len(results) == 5
        assert results[0].content == "item 10"
        assert results[4].content == "item 14"

    def test_delete(self):
        store = InMemoryStore()
        store.save("conv1", MemoryItem(content="test"))
        assert store.delete("conv1") is True
        assert store.retrieve("conv1") == []

    def test_delete_nonexistent(self):
        store = InMemoryStore()
        assert store.delete("nonexistent") is False

    def test_clear(self):
        store = InMemoryStore()
        store.save("conv1", MemoryItem(content="a"))
        store.save("conv2", MemoryItem(content="b"))
        store.clear()
        assert store.retrieve("conv1") == []
        assert store.retrieve("conv2") == []

    def test_size(self):
        store = InMemoryStore()
        store.save("conv1", MemoryItem(content="a"))
        store.save("conv1", MemoryItem(content="b"))
        store.save("conv2", MemoryItem(content="c"))
        assert store.size == 3

    def test_multiple_keys_isolated(self):
        store = InMemoryStore()
        store.save("conv1", MemoryItem(content="a"))
        store.save("conv2", MemoryItem(content="b"))
        assert len(store.retrieve("conv1")) == 1
        assert len(store.retrieve("conv2")) == 1


class TestFileMemoryStore:
    """Tests for FileMemoryStore."""

    def test_save_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileMemoryStore(base_dir=tmpdir)
            store.save("conv1", MemoryItem(content="file memory"))

            results = store.retrieve("conv1")
            assert len(results) == 1
            assert results[0].content == "file memory"

    def test_persistence_across_instances(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store1 = FileMemoryStore(base_dir=tmpdir)
            store1.save("conv1", MemoryItem(content="persistent"))

            store2 = FileMemoryStore(base_dir=tmpdir)
            results = store2.retrieve("conv1")
            assert len(results) == 1
            assert results[0].content == "persistent"

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileMemoryStore(base_dir=tmpdir)
            store.save("conv1", MemoryItem(content="test"))
            assert store.delete("conv1") is True
            assert store.retrieve("conv1") == []

    def test_delete_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileMemoryStore(base_dir=tmpdir)
            assert store.delete("nonexistent") is False

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileMemoryStore(base_dir=tmpdir)
            store.save("conv1", MemoryItem(content="a"))
            store.save("conv2", MemoryItem(content="b"))
            store.clear()
            assert store.retrieve("conv1") == []
            assert store.retrieve("conv2") == []

    def test_retrieve_with_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileMemoryStore(base_dir=tmpdir)
            for i in range(10):
                store.save("conv1", MemoryItem(content=f"item {i}"))

            results = store.retrieve("conv1", limit=3)
            assert len(results) == 3
            assert results[0].content == "item 7"

    def test_safe_key_sanitization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileMemoryStore(base_dir=tmpdir)
            store.save("conv/with:slashes", MemoryItem(content="safe"))
            results = store.retrieve("conv/with:slashes")
            assert len(results) == 1
            assert results[0].content == "safe"


class TestContextWindowManager:
    """Tests for ContextWindowManager."""

    def test_no_truncation_needed(self):
        conv = Conversation("test")
        conv.add_message("system", "You are KGK AI.")
        conv.add_message("user", "Hello")
        conv.add_message("assistant", "Hi there!")

        manager = ContextWindowManager(max_tokens=4096, reserved_for_response=1024)
        result = manager.truncate(conv)

        assert result.truncated_count == 0
        assert len(result.messages) == 3
        assert result.summary is None

    def test_truncation_drops_old_messages(self):
        conv = Conversation("test")
        conv.add_message("system", "You are KGK AI.")
        for i in range(20):
            conv.add_message("user", f"Message {i} " * 50)
            conv.add_message("assistant", f"Response {i} " * 50)

        manager = ContextWindowManager(max_tokens=500, reserved_for_response=100)
        result = manager.truncate(conv)

        assert result.truncated_count > 0
        assert result.original_tokens > result.final_tokens
        assert result.final_tokens <= manager.available_tokens

    def test_system_prompt_always_kept(self):
        conv = Conversation("test")
        conv.add_message("system", "You are KGK AI. " * 10)
        for i in range(20):
            conv.add_message("user", f"Message {i} " * 50)

        manager = ContextWindowManager(max_tokens=500, reserved_for_response=100)
        result = manager.truncate(conv)

        assert result.messages[0].role == "system"
        assert "KGK AI" in result.messages[0].content

    def test_summary_prepended_when_summarizer_present(self):
        conv = Conversation("test")
        conv.add_message("system", "You are KGK AI.")
        for i in range(20):
            conv.add_message("user", f"Message {i} " * 50)
            conv.add_message("assistant", f"Response {i} " * 50)

        mock_summarizer = MagicMock()
        mock_summarizer.summarize.return_value = "Summary of old conversation"

        manager = ContextWindowManager(
            max_tokens=500,
            reserved_for_response=100,
            summarizer=mock_summarizer,
        )
        result = manager.truncate(conv)

        assert result.summary == "Summary of old conversation"
        summary_msgs = [m for m in result.messages if "Previous conversation summary" in m.content]
        assert len(summary_msgs) == 1

    def test_no_summary_without_summarizer(self):
        conv = Conversation("test")
        conv.add_message("system", "You are KGK AI.")
        for i in range(20):
            conv.add_message("user", f"Message {i} " * 50)

        manager = ContextWindowManager(max_tokens=500, reserved_for_response=100)
        result = manager.truncate(conv)

        assert result.summary is None

    def test_get_messages_for_model(self):
        conv = Conversation("test")
        conv.add_message("system", "You are KGK AI.")
        conv.add_message("user", "Hello")
        conv.add_message("assistant", "Hi!")

        manager = ContextWindowManager(max_tokens=4096, reserved_for_response=1024)
        msgs = manager.get_messages_for_model(conv, include_system=True)

        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"

    def test_get_messages_for_model_no_system(self):
        conv = Conversation("test")
        conv.add_message("system", "You are KGK AI.")
        conv.add_message("user", "Hello")

        manager = ContextWindowManager(max_tokens=4096, reserved_for_response=1024)
        msgs = manager.get_messages_for_model(conv, include_system=False)

        assert all(m["role"] != "system" for m in msgs)

    def test_available_tokens(self):
        manager = ContextWindowManager(max_tokens=4096, reserved_for_response=1024)
        assert manager.available_tokens == 3072

    def test_empty_conversation(self):
        conv = Conversation("test")
        manager = ContextWindowManager(max_tokens=4096, reserved_for_response=1024)
        result = manager.truncate(conv)

        assert result.truncated_count == 0
        assert result.messages == []
        assert result.original_tokens == 0


class TestConversationSummarizer:
    """Tests for ConversationSummarizer."""

    def test_empty_messages(self):
        summarizer = ConversationSummarizer()
        assert summarizer.summarize([]) == ""

    def test_extractive_short_conversation(self):
        summarizer = ConversationSummarizer()
        messages = [
            Message(role="user", content="What is KGK?"),
            Message(role="assistant", content="KGK is a company."),
        ]
        summary = summarizer.summarize(messages)
        assert "KGK" in summary

    def test_extractive_long_conversation(self):
        summarizer = ConversationSummarizer(max_summary_chars=300)
        messages = []
        for i in range(10):
            messages.append(Message(role="user", content=f"Question {i} about topic {i}"))
            messages.append(Message(role="assistant", content=f"Answer {i} about topic {i}"))

        summary = summarizer.summarize(messages)
        assert "omitted" in summary
        assert len(summary) <= 303  # 300 + "..."

    def test_extractive_truncates_long_content(self):
        summarizer = ConversationSummarizer()
        messages = [
            Message(role="user", content="A" * 500),
        ]
        summary = summarizer.summarize(messages)
        assert "..." in summary

    def test_model_summarization(self):
        mock_model = MagicMock()
        mock_model.generate.return_value = MagicMock(text="This is a model summary.")

        summarizer = ConversationSummarizer(model=mock_model)
        messages = [
            Message(role="user", content="Tell me about KGK"),
            Message(role="assistant", content="KGK is a great company."),
        ]
        summary = summarizer.summarize(messages)

        assert summary == "This is a model summary."
        mock_model.generate.assert_called_once()

    def test_model_summarization_falls_back_on_error(self):
        mock_model = MagicMock()
        mock_model.generate.side_effect = Exception("Model error")

        summarizer = ConversationSummarizer(model=mock_model)
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]
        summary = summarizer.summarize(messages)

        assert "USER" in summary
        assert "ASSISTANT" in summary

    def test_model_summary_truncated_to_max(self):
        mock_model = MagicMock()
        mock_model.generate.return_value = MagicMock(text="A" * 1000)

        summarizer = ConversationSummarizer(model=mock_model, max_summary_chars=100)
        messages = [Message(role="user", content="test")]
        summary = summarizer.summarize(messages)

        assert len(summary) <= 103  # 100 + "..."
        assert summary.endswith("...")


class TestMemoryManagerIntegration:
    """Tests for MemoryManager with concrete stores."""

    def test_manager_with_in_memory_store(self):
        short_term = InMemoryStore()
        manager = MemoryManager(short_term=short_term)

        manager.save_memory("Hello", conversation_id="conv1")
        items = manager.retrieve_memory("conv1")
        assert len(items) == 1
        assert items[0].content == "Hello"

    def test_manager_with_file_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            short_term = FileMemoryStore(base_dir=tmpdir)
            manager = MemoryManager(short_term=short_term)

            manager.save_memory("File memory", conversation_id="conv1")
            items = manager.retrieve_memory("conv1")
            assert len(items) == 1
            assert items[0].content == "File memory"

    def test_manager_long_term(self):
        short_term = InMemoryStore()
        long_term = InMemoryStore()
        manager = MemoryManager(short_term=short_term, long_term=long_term)

        manager.save_memory("Important", conversation_id="conv1", long_term=True)
        assert len(manager.retrieve_memory("conv1")) == 1
        assert len(manager.retrieve_memory("conv1", include_long_term=True)) == 2

    def test_manager_delete(self):
        short_term = InMemoryStore()
        manager = MemoryManager(short_term=short_term)

        manager.save_memory("test", conversation_id="conv1")
        assert manager.delete_memory("conv1") is True
        assert manager.retrieve_memory("conv1") == []

    def test_manager_clear(self):
        short_term = InMemoryStore()
        long_term = InMemoryStore()
        manager = MemoryManager(short_term=short_term, long_term=long_term)

        manager.save_memory("a", conversation_id="conv1")
        manager.save_memory("b", conversation_id="conv2", long_term=True)
        manager.clear_memory()
        assert manager.retrieve_memory("conv1") == []
        assert manager.retrieve_memory("conv2", include_long_term=True) == []
