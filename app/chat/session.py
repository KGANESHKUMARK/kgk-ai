"""KGK AI Chat Session Manager — Manages conversation lifecycle.

Provides a higher-level interface for managing multiple conversations,
including creation, retrieval, clearing, and listing. Also handles
memory integration for conversation persistence.
"""

from __future__ import annotations

from typing import Optional

from app.chat.controller import KGKChatController, ChatResponseData
from app.chat.conversation import Conversation
from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger("chat.session")


class ChatSessionManager:
    """Manages chat sessions and provides a simplified interface.

    Wraps KGKChatController with convenience methods for session management.
    Designed to be used by both the API layer and the Gradio UI.

    Attributes:
        controller: The underlying chat controller.
    """

    def __init__(self, controller: Optional[KGKChatController] = None) -> None:
        self.controller: KGKChatController = controller or KGKChatController()

    def send_message(
        self,
        message: str,
        conversation_id: str = "default",
        **kwargs: object,
    ) -> ChatResponseData:
        """Send a message and get a complete response.

        Args:
            message: User's message text.
            conversation_id: Conversation identifier.
            **kwargs: Additional arguments passed to controller.chat().

        Returns:
            ChatResponseData with the response.
        """
        return self.controller.chat(message, conversation_id, **kwargs)

    def stream_message(
        self,
        message: str,
        conversation_id: str = "default",
        **kwargs: object,
    ):
        """Send a message and stream the response.

        Args:
            message: User's message text.
            conversation_id: Conversation identifier.
            **kwargs: Additional arguments passed to controller.stream_chat().

        Yields:
            Text chunks as they are generated.
        """
        yield from self.controller.stream_chat(message, conversation_id, **kwargs)

    def get_history(self, conversation_id: str = "default") -> list[dict[str, str]]:
        """Get conversation history as message dicts.

        Args:
            conversation_id: Conversation identifier.

        Returns:
            List of {'role': ..., 'content': ...} dicts.
        """
        conv = self.controller.get_conversation(conversation_id)
        if conv is None:
            return []
        return conv.get_messages(include_system=False)

    def get_history_text(self, conversation_id: str = "default") -> str:
        """Get conversation history as formatted text.

        Args:
            conversation_id: Conversation identifier.

        Returns:
            Formatted history string.
        """
        conv = self.controller.get_conversation(conversation_id)
        if conv is None:
            return ""
        return conv.get_history_text()

    def clear_session(self, conversation_id: str) -> bool:
        """Clear a conversation session.

        Args:
            conversation_id: Conversation identifier.

        Returns:
            True if session was found and cleared.
        """
        return self.controller.clear_conversation(conversation_id)

    def list_sessions(self) -> list[str]:
        """List all active session IDs.

        Returns:
            List of conversation IDs.
        """
        return self.controller.list_conversations()

    def session_exists(self, conversation_id: str) -> bool:
        """Check if a session exists.

        Args:
            conversation_id: Conversation identifier.

        Returns:
            True if the session exists.
        """
        return conversation_id in self.controller.list_conversations()

    def session_length(self, conversation_id: str) -> int:
        """Get the number of messages in a session.

        Args:
            conversation_id: Conversation identifier.

        Returns:
            Number of messages (0 if session doesn't exist).
        """
        conv = self.controller.get_conversation(conversation_id)
        if conv is None:
            return 0
        return conv.length


# Singleton instance for app-wide use
_session_manager: Optional[ChatSessionManager] = None


def get_session_manager() -> ChatSessionManager:
    """Return the singleton ChatSessionManager instance.

    Returns:
        ChatSessionManager instance.
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = ChatSessionManager()
        logger.info("Chat session manager initialized")
    return _session_manager
