"""KGK AI Memory Manager — Unified memory interface.

Provides a consistent API for short-term (conversation) and long-term
(persistent) memory. The storage mechanism is abstracted so it can
be swapped (in-memory, file, database) without changing callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class MemoryItem:
    """A single memory entry.

    Attributes:
        content: The memory content (text).
        timestamp: When the memory was created.
        metadata: Additional metadata (e.g. conversation_id, source).
    """

    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseMemoryStore(ABC):
    """Abstract base class for memory storage backends."""

    @abstractmethod
    def save(self, key: str, item: MemoryItem) -> None:
        """Save a memory item under a key."""
        ...

    @abstractmethod
    def retrieve(self, key: str, limit: int = 10) -> list[MemoryItem]:
        """Retrieve memory items for a key.

        Args:
            key: Lookup key (e.g. conversation_id).
            limit: Maximum number of items to return.

        Returns:
            List of MemoryItem instances.
        """
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete all memory items for a key.

        Returns:
            True if items were deleted, False if key not found.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all memory items."""
        ...


class MemoryManager:
    """Unified memory manager for KGK AI.

    Manages both short-term (conversation) and long-term (persistent)
    memory through a single interface.

    Attributes:
        short_term: Short-term memory store (current conversation).
        long_term: Long-term memory store (persistent across sessions).
    """

    def __init__(
        self,
        short_term: BaseMemoryStore,
        long_term: Optional[BaseMemoryStore] = None,
    ) -> None:
        self.short_term = short_term
        self.long_term = long_term

    def save_memory(
        self,
        content: str,
        conversation_id: str = "default",
        long_term: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Save a memory item.

        Args:
            content: The memory content.
            conversation_id: Conversation identifier.
            long_term: If True, also save to long-term memory.
            metadata: Additional metadata.
        """
        item = MemoryItem(content=content, metadata=metadata or {})
        self.short_term.save(conversation_id, item)

        if long_term and self.long_term is not None:
            self.long_term.save(conversation_id, item)

    def retrieve_memory(
        self,
        conversation_id: str = "default",
        limit: int = 10,
        include_long_term: bool = False,
    ) -> list[MemoryItem]:
        """Retrieve memory items for a conversation.

        Args:
            conversation_id: Conversation identifier.
            limit: Maximum items to return.
            include_long_term: Also search long-term memory.

        Returns:
            List of MemoryItem instances.
        """
        items = self.short_term.retrieve(conversation_id, limit=limit)

        if include_long_term and self.long_term is not None:
            long_items = self.long_term.retrieve(conversation_id, limit=limit)
            items.extend(long_items)

        return items[:limit]

    def delete_memory(self, conversation_id: str) -> bool:
        """Delete memory for a conversation.

        Args:
            conversation_id: Conversation identifier.

        Returns:
            True if memory was deleted.
        """
        deleted = self.short_term.delete(conversation_id)
        if self.long_term is not None:
            self.long_term.delete(conversation_id)
        return deleted

    def clear_memory(self) -> None:
        """Clear all memory (short-term and long-term)."""
        self.short_term.clear()
        if self.long_term is not None:
            self.long_term.clear()
