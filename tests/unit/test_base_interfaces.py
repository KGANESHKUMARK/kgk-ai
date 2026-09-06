"""Unit tests for KGK AI base interfaces."""

import pytest
from app.models.base import BaseModelProvider, GenerationParams, GenerationResult, ModelInfo
from app.tools.registry import BaseTool, ToolInfo, ToolResult, ToolRegistry
from app.agents.base import AgentTask, AgentResult, BaseAgent
from app.memory.manager import MemoryItem, BaseMemoryStore, MemoryManager
from app.rag.vector_store import DocumentChunk, RetrievalResult, BaseVectorStore
from app.chat.conversation import Conversation, Message
from app.chat.prompts import get_system_prompt


class TestGenerationParams:
    """Tests for GenerationParams dataclass."""

    def test_defaults(self):
        params = GenerationParams()
        assert params.max_new_tokens == 1024
        assert params.temperature == 0.7
        assert params.do_sample is True

    def test_custom_values(self):
        params = GenerationParams(max_new_tokens=512, temperature=0.3)
        assert params.max_new_tokens == 512
        assert params.temperature == 0.3


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_creation(self):
        result = GenerationResult(text="Hello", finish_reason="stop")
        assert result.text == "Hello"
        assert result.finish_reason == "stop"
        assert result.metadata == {}


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_defaults(self):
        info = ModelInfo()
        assert info.name == ""
        assert info.loaded is False
        assert info.quantization == "none"


class TestBaseModelProvider:
    """Tests for BaseModelProvider abstract class."""

    def test_is_abstract(self):
        """Verify BaseModelProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModelProvider()

    def test_embed_not_implemented(self):
        """Verify default embed raises NotImplementedError."""
        class DummyProvider(BaseModelProvider):
            def load(self): pass
            def unload(self): pass
            def generate(self, messages, params=None): return GenerationResult(text="")
            def stream(self, messages, params=None): yield ""
            def health_check(self): return True
            def get_info(self): return ModelInfo()

        provider = DummyProvider()
        with pytest.raises(NotImplementedError):
            provider.embed(["test"])


class TestConversation:
    """Tests for Conversation class."""

    def test_add_and_get_messages(self):
        conv = Conversation("test-123")
        conv.add_message("system", "You are KGK AI")
        conv.add_message("user", "Hello")
        conv.add_message("assistant", "Hi there!")

        assert conv.length == 3
        assert conv.conversation_id == "test-123"

        msgs = conv.get_messages()
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["content"] == "Hello"

    def test_get_messages_exclude_system(self):
        conv = Conversation()
        conv.add_message("system", "system prompt")
        conv.add_message("user", "hello")

        msgs = conv.get_messages(include_system=False)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_clear(self):
        conv = Conversation()
        conv.add_message("user", "test")
        conv.clear()
        assert conv.length == 0

    def test_history_text(self):
        conv = Conversation()
        conv.add_message("user", "Hello")
        conv.add_message("assistant", "Hi!")
        history = conv.get_history_text()
        assert "USER" in history
        assert "ASSISTANT" in history
        assert "Hello" in history


class TestSystemPrompt:
    """Tests for KGK system prompt."""

    def test_prompt_exists(self):
        prompt = get_system_prompt()
        assert len(prompt) > 100
        assert "KGK AI" in prompt

    def test_prompt_contains_personality(self):
        prompt = get_system_prompt()
        assert "intelligent" in prompt.lower()
        assert "honest" in prompt.lower()
        assert "uncertain" in prompt.lower()

    def test_prompt_contains_source_distinction(self):
        prompt = get_system_prompt()
        assert "General model knowledge" in prompt
        assert "Retrieved KGK knowledge" in prompt
        assert "Tool-generated information" in prompt
        assert "User-provided information" in prompt

    def test_prompt_contains_no_hallucination_rule(self):
        prompt = get_system_prompt()
        assert "hallucinate" in prompt.lower()
        assert "foundation model" in prompt.lower()


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_register_and_list(self):
        registry = ToolRegistry()

        class DummyTool(BaseTool):
            def info(self):
                return ToolInfo(name="dummy", description="A dummy tool")
            def execute(self, **kwargs):
                return ToolResult(success=True, output="done")

        registry.register_tool(DummyTool())
        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "dummy"

    def test_remove_tool(self):
        registry = ToolRegistry()

        class DummyTool(BaseTool):
            def info(self):
                return ToolInfo(name="removable", description="test")
            def execute(self, **kwargs):
                return ToolResult(success=True)

        registry.register_tool(DummyTool())
        assert registry.remove_tool("removable") is True
        assert registry.remove_tool("nonexistent") is False

    def test_execute_nonexistent_tool(self):
        registry = ToolRegistry()
        result = registry.execute_tool("nonexistent")
        assert result.success is False
        assert "not found" in result.error

    def test_public_context_blocks_unsafe(self):
        registry = ToolRegistry()

        class UnsafeTool(BaseTool):
            def info(self):
                return ToolInfo(name="unsafe", description="unsafe", safe_for_public=False)
            def execute(self, **kwargs):
                return ToolResult(success=True, output="secret")

        registry.register_tool(UnsafeTool())
        result = registry.execute_tool("unsafe", public_context=True)
        assert result.success is False
        assert "not available" in result.error


class TestMemoryManager:
    """Tests for MemoryManager class."""

    def test_save_and_retrieve(self):
        class InMemoryStore(BaseMemoryStore):
            def __init__(self):
                self._data: dict[str, list] = {}
            def save(self, key, item):
                self._data.setdefault(key, []).append(item)
            def retrieve(self, key, limit=10):
                return self._data.get(key, [])[:limit]
            def delete(self, key):
                if key in self._data:
                    del self._data[key]
                    return True
                return False
            def clear(self):
                self._data.clear()

        manager = MemoryManager(short_term=InMemoryStore())
        manager.save_memory("test memory", conversation_id="conv1")
        items = manager.retrieve_memory("conv1")
        assert len(items) == 1
        assert items[0].content == "test memory"

    def test_delete_memory(self):
        class InMemoryStore(BaseMemoryStore):
            def __init__(self):
                self._data: dict[str, list] = {}
            def save(self, key, item):
                self._data.setdefault(key, []).append(item)
            def retrieve(self, key, limit=10):
                return self._data.get(key, [])[:limit]
            def delete(self, key):
                if key in self._data:
                    del self._data[key]
                    return True
                return False
            def clear(self):
                self._data.clear()

        manager = MemoryManager(short_term=InMemoryStore())
        manager.save_memory("test", conversation_id="conv1")
        assert manager.delete_memory("conv1") is True
        items = manager.retrieve_memory("conv1")
        assert len(items) == 0


class TestDocumentChunk:
    """Tests for DocumentChunk dataclass."""

    def test_creation(self):
        chunk = DocumentChunk(
            content="test content",
            source="/path/to/file.txt",
            filename="file.txt",
            chunk_id="chunk_001",
            document_id="doc_001",
        )
        assert chunk.content == "test content"
        assert chunk.filename == "file.txt"
        assert chunk.chunk_id == "chunk_001"


class TestAgentTask:
    """Tests for AgentTask and AgentResult."""

    def test_task_creation(self):
        task = AgentTask(query="What is AI?", context="Previous conversation")
        assert task.query == "What is AI?"
        assert task.context == "Previous conversation"

    def test_result_creation(self):
        result = AgentResult(success=True, response="AI is...", tools_used=["search"])
        assert result.success is True
        assert "search" in result.tools_used
