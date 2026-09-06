"""KGK AI Conversation Management.

Tracks conversation state, message history, and provides
structured access to conversation data for the controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class Message:
    """A single message in a conversation.

    Attributes:
        role: Message role ('system', 'user', 'assistant').
        content: Message text content.
        timestamp: When the message was created.
        metadata: Additional metadata (e.g. sources, tools_used).
    """

    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, str]:
        """Convert to a dict suitable for model input.

        Returns:
            Dict with 'role' and 'content' keys.
        """
        return {"role": self.role, "content": self.content}


class Conversation:
    """A conversation session with message history.

    Attributes:
        conversation_id: Unique identifier for this conversation.
        messages: List of Message objects in order.
    """

    def __init__(self, conversation_id: str = "default") -> None:
        self.conversation_id: str = conversation_id
        self.messages: list[Message] = []

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Message:
        """Add a message to the conversation.

        Args:
            role: Message role ('system', 'user', 'assistant').
            content: Message text.
            metadata: Optional metadata.

        Returns:
            The created Message instance.
        """
        msg = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)
        return msg

    def get_messages(self, include_system: bool = True) -> list[dict[str, str]]:
        """Return messages as dicts for model input.

        Args:
            include_system: Whether to include system messages.

        Returns:
            List of {'role': ..., 'content': ...} dicts.
        """
        msgs = []
        for m in self.messages:
            if not include_system and m.role == "system":
                continue
            msgs.append(m.to_dict())
        return msgs

    def get_history_text(self, max_messages: int = 20) -> str:
        """Return conversation history as formatted text.

        Args:
            max_messages: Maximum number of recent messages to include.

        Returns:
            Formatted string of conversation history.
        """
        recent = self.messages[-max_messages:]
        lines = []
        for m in recent:
            prefix = m.role.upper()
            lines.append(f"[{prefix}]: {m.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Remove all messages from the conversation."""
        self.messages.clear()

    @property
    def length(self) -> int:
        """Return the number of messages."""
        return len(self.messages)
