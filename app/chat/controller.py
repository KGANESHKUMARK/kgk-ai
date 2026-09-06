"""KGK AI Chat Controller — Core orchestration layer.

The controller is the brain of KGK AI. It receives user messages, manages
conversation context, optionally retrieves knowledge (RAG), executes tools,
calls the model, and returns structured responses with source citations.

This is the single entry point for all chat interactions — both the API
and the Gradio UI call through the controller.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

from app.chat.prompts import get_system_prompt
from app.chat.conversation import Conversation
from app.config import get_settings
from app.logging_config import get_logger, generate_request_id
from app.memory.manager import MemoryManager
from app.models.base import BaseModelProvider, GenerationParams, GenerationResult
from app.models.registry import registry as model_registry
from app.tools.registry import ToolRegistry, tool_registry

logger = get_logger("chat.controller")


@dataclass
class ChatResponseData:
    """Structured response from the chat controller.

    Attributes:
        text: The generated response text.
        conversation_id: Conversation identifier.
        sources: List of source citations (if RAG was used).
        tools_used: List of tool names that were executed.
        model: Model name that generated the response.
        latency_ms: Total processing time in milliseconds.
        request_id: Unique request identifier for tracing.
        error: Error message if the request failed (empty if successful).
    """

    text: str
    conversation_id: str = "default"
    sources: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    model: str = ""
    latency_ms: float = 0.0
    request_id: str = ""
    error: str = ""


class KGKChatController:
    """Orchestrates the full KGK AI chat pipeline.

    Pipeline:
        1. Validate and sanitize input
        2. Load or create conversation
        3. Add user message to conversation
        4. Optionally retrieve relevant knowledge (RAG)
        5. Optionally execute tools
        6. Build model input with system prompt + context
        7. Generate response (streaming or non-streaming)
        8. Add assistant response to conversation
        9. Optionally save to memory
        10. Return structured response

    Attributes:
        model: Active model provider (from registry or injected).
        memory: Memory manager instance (optional).
        tools: Tool registry instance (optional).
        conversations: In-memory conversation store.
    """

    def __init__(
        self,
        model: Optional[BaseModelProvider] = None,
        memory: Optional[MemoryManager] = None,
        tools: Optional[ToolRegistry] = None,
    ) -> None:
        settings = get_settings()
        self.model: Optional[BaseModelProvider] = model or model_registry.active
        self.memory: Optional[MemoryManager] = memory if settings.enable_memory else None
        self.tools: Optional[ToolRegistry] = tools if settings.enable_tools else None
        self._conversations: dict[str, Conversation] = {}
        self._system_prompt: str = get_system_prompt()

    def chat(
        self,
        message: str,
        conversation_id: str = "default",
        use_memory: bool = True,
        use_rag: bool = True,
        use_tools: bool = True,
    ) -> ChatResponseData:
        """Process a chat message and return a complete response.

        Args:
            message: User's message text.
            conversation_id: Conversation identifier.
            use_memory: Whether to include memory context.
            use_rag: Whether to use RAG retrieval (if available).
            use_tools: Whether to use tools (if available).

        Returns:
            ChatResponseData with the generated response and metadata.
        """
        request_id = generate_request_id()
        start_time = time.time()

        logger.info(
            f"Chat request: conv={conversation_id}, len={len(message)}",
            extra={"component": "chat.controller", "request_id": request_id, "conversation_id": conversation_id},
        )

        response = ChatResponseData(
            text="",
            conversation_id=conversation_id,
            request_id=request_id,
        )

        # 1. Validate input
        settings = get_settings()
        if len(message) > settings.max_input_length:
            response.text = f"Your message is too long. Please keep it under {settings.max_input_length} characters."
            response.error = "input_too_long"
            response.latency_ms = (time.time() - start_time) * 1000
            return response

        if not message.strip():
            response.text = "Please enter a message."
            response.error = "empty_input"
            response.latency_ms = (time.time() - start_time) * 1000
            return response

        # 2. Load or create conversation
        conv = self._get_conversation(conversation_id)

        # Ensure system prompt is present
        if conv.length == 0 or conv.messages[0].role != "system":
            conv.messages.insert(0, __import__("app.chat.conversation", fromlist=["Message"]).Message(
                role="system", content=self._system_prompt
            ))

        # 3. Add user message
        conv.add_message("user", message)

        # 4. Check model availability
        if self.model is None:
            response.text = (
                "KGK AI is not fully initialized yet. The model has not been loaded. "
                "This is expected during Phase 1-3 setup. Model loading will be "
                "available after dependencies are installed."
            )
            response.error = "model_not_loaded"
            response.latency_ms = (time.time() - start_time) * 1000
            logger.warning(
                "No model provider available",
                extra={"component": "chat.controller", "request_id": request_id},
            )
            return response

        # 5. Build model input
        model_messages = conv.get_messages(include_system=True)

        # 6. Generate response
        try:
            result = self.model.generate(model_messages)
            response.text = result.text
            response.model = self.model.get_info().name if self.model.health_check() else "unknown"
        except Exception as e:
            logger.error(
                f"Model generation failed: {e}",
                extra={"component": "chat.controller", "request_id": request_id, "error": str(e)},
            )
            response.text = "KGK AI encountered an error while generating a response. Please try again."
            response.error = "generation_error"
            response.latency_ms = (time.time() - start_time) * 1000
            return response

        # 7. Add assistant response to conversation
        conv.add_message("assistant", result.text, metadata={
            "request_id": request_id,
            "finish_reason": result.finish_reason,
        })

        # 8. Save to memory if enabled
        if use_memory and self.memory is not None:
            try:
                self.memory.save_memory(
                    content=f"User: {message}\nAssistant: {result.text}",
                    conversation_id=conversation_id,
                    metadata={"request_id": request_id},
                )
            except Exception as e:
                logger.warning(f"Memory save failed: {e}")

        # 9. Finalize response
        response.latency_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Chat response: {len(response.text)} chars, {response.latency_ms:.0f}ms",
            extra={
                "component": "chat.controller",
                "request_id": request_id,
                "latency_ms": response.latency_ms,
                "response_length": len(response.text),
            },
        )

        return response

    def stream_chat(
        self,
        message: str,
        conversation_id: str = "default",
        use_memory: bool = True,
        use_rag: bool = True,
        use_tools: bool = True,
    ) -> Generator[str, None, None]:
        """Stream a chat response token-by-token.

        Args:
            message: User's message text.
            conversation_id: Conversation identifier.
            use_memory: Whether to include memory context.
            use_rag: Whether to use RAG retrieval.
            use_tools: Whether to use tools.

        Yields:
            Text chunks as they are generated.
        """
        request_id = generate_request_id()
        start_time = time.time()

        logger.info(
            f"Stream chat request: conv={conversation_id}, len={len(message)}",
            extra={"component": "chat.controller", "request_id": request_id, "conversation_id": conversation_id},
        )

        # Validate input
        settings = get_settings()
        if len(message) > settings.max_input_length:
            yield f"Your message is too long. Please keep it under {settings.max_input_length} characters."
            return

        if not message.strip():
            yield "Please enter a message."
            return

        # Check model
        if self.model is None:
            yield (
                "KGK AI is not fully initialized yet. The model has not been loaded. "
                "This is expected during early setup phases."
            )
            return

        # Load conversation
        conv = self._get_conversation(conversation_id)

        # Ensure system prompt
        if conv.length == 0 or conv.messages[0].role != "system":
            from app.chat.conversation import Message
            conv.messages.insert(0, Message(role="system", content=self._system_prompt))

        # Add user message
        conv.add_message("user", message)

        # Build model input
        model_messages = conv.get_messages(include_system=True)

        # Stream generation
        full_text = ""
        try:
            for chunk in self.model.stream(model_messages):
                full_text += chunk
                yield chunk
        except Exception as e:
            logger.error(
                f"Streaming failed: {e}",
                extra={"component": "chat.controller", "request_id": request_id, "error": str(e)},
            )
            yield "\n\n[KGK AI encountered an error while streaming. Please try again.]"
            return

        # Add assistant response to conversation
        conv.add_message("assistant", full_text, metadata={
            "request_id": request_id,
            "streamed": True,
        })

        # Save to memory
        if use_memory and self.memory is not None:
            try:
                self.memory.save_memory(
                    content=f"User: {message}\nAssistant: {full_text}",
                    conversation_id=conversation_id,
                    metadata={"request_id": request_id},
                )
            except Exception as e:
                logger.warning(f"Memory save failed: {e}")

        latency_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Stream chat complete: {len(full_text)} chars, {latency_ms:.0f}ms",
            extra={
                "component": "chat.controller",
                "request_id": request_id,
                "latency_ms": latency_ms,
                "response_length": len(full_text),
            },
        )

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Retrieve a conversation by ID.

        Args:
            conversation_id: Conversation identifier.

        Returns:
            Conversation instance or None if not found.
        """
        return self._conversations.get(conversation_id)

    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear a conversation's history.

        Args:
            conversation_id: Conversation identifier.

        Returns:
            True if conversation was found and cleared.
        """
        conv = self._conversations.get(conversation_id)
        if conv is None:
            return False
        conv.clear()
        return True

    def list_conversations(self) -> list[str]:
        """List all active conversation IDs.

        Returns:
            List of conversation IDs.
        """
        return list(self._conversations.keys())

    def set_model(self, model: BaseModelProvider) -> None:
        """Set or replace the active model provider.

        Args:
            model: BaseModelProvider instance.
        """
        self.model = model
        logger.info(f"Model set: {model.get_info().name}")

    def _get_conversation(self, conversation_id: str) -> Conversation:
        """Get or create a conversation.

        Args:
            conversation_id: Conversation identifier.

        Returns:
            Conversation instance (created if not exists).
        """
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = Conversation(conversation_id)
            logger.info(
                f"New conversation created: {conversation_id}",
                extra={"component": "chat.controller", "conversation_id": conversation_id},
            )
        return self._conversations[conversation_id]
