"""KGK AI Context Window Manager — Token-aware conversation truncation.

Manages conversation history to fit within a model's context window.
When the conversation exceeds the token budget, older messages are
truncated or summarized to make room for new ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from app.logging_config import get_logger

if TYPE_CHECKING:
    from app.chat.conversation import Conversation, Message

logger = get_logger("memory.context_window")


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate the number of tokens in a text string.

    Uses a simple heuristic: ~4 characters per token for English text.
    This is a rough approximation; actual tokenization depends on the model.

    Args:
        text: Input text.
        chars_per_token: Approximate characters per token.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token))


def estimate_message_tokens(message: Message) -> int:
    """Estimate tokens for a single message including role overhead.

    Args:
        message: Message to estimate.

    Returns:
        Estimated token count (content + ~4 tokens overhead per message).
    """
    return estimate_tokens(message.content) + 4


@dataclass
class TruncationResult:
    """Result of context window truncation.

    Attributes:
        messages: The truncated list of messages to send to the model.
        truncated_count: Number of messages removed from the front.
        original_tokens: Estimated tokens before truncation.
        final_tokens: Estimated tokens after truncation.
        summary: Optional summary of truncated messages.
    """

    messages: list[Message]
    truncated_count: int
    original_tokens: int
    final_tokens: int
    summary: Optional[str] = None


class ContextWindowManager:
    """Manages conversation context to fit within token limits.

    Strategies (in order of preference):
    1. Keep all messages if within budget.
    2. Drop oldest messages (keep system prompt + recent messages).
    3. Summarize dropped messages and prepend as a system summary.

    Attributes:
        max_tokens: Maximum tokens for the context window.
        reserved_for_response: Tokens reserved for the model's response.
        summarizer: Optional summarizer for old messages.
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        reserved_for_response: int = 1024,
        summarizer: Optional[Any] = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.reserved_for_response = reserved_for_response
        self.summarizer = summarizer

    @property
    def available_tokens(self) -> int:
        """Tokens available for conversation context."""
        return self.max_tokens - self.reserved_for_response

    def truncate(self, conversation: Conversation) -> TruncationResult:
        """Truncate conversation to fit within the token budget.

        Args:
            conversation: The conversation to truncate.

        Returns:
            TruncationResult with the truncated message list and metadata.
        """
        all_messages = list(conversation.messages)
        total_tokens = sum(estimate_message_tokens(m) for m in all_messages)

        if total_tokens <= self.available_tokens:
            return TruncationResult(
                messages=all_messages,
                truncated_count=0,
                original_tokens=total_tokens,
                final_tokens=total_tokens,
            )

        # Need to truncate — always keep system prompt(s) at the front
        system_messages = [m for m in all_messages if m.role == "system"]
        non_system = [m for m in all_messages if m.role != "system"]

        system_tokens = sum(estimate_message_tokens(m) for m in system_messages)
        budget = self.available_tokens - system_tokens

        # Keep as many recent non-system messages as fit
        kept: list[Message] = []
        kept_tokens = 0

        for msg in reversed(non_system):
            msg_tokens = estimate_message_tokens(msg)
            if kept_tokens + msg_tokens > budget:
                break
            kept.insert(0, msg)
            kept_tokens += msg_tokens

        truncated_count = len(non_system) - len(kept)
        truncated_messages = non_system[:truncated_count]

        # Optionally summarize truncated messages
        summary = None
        if self.summarizer is not None and truncated_messages:
            try:
                summary = self.summarizer.summarize(truncated_messages)
            except Exception as e:
                logger.warning(f"Summarization failed: {e}")

        # Build final message list
        final_messages = list(system_messages)

        if summary:
            from app.chat.conversation import Message as _Message
            summary_msg = _Message(
                role="system",
                content=f"[Previous conversation summary]\n{summary}",
            )
            final_messages.append(summary_msg)

        final_messages.extend(kept)

        final_tokens = sum(estimate_message_tokens(m) for m in final_messages)

        logger.info(
            f"Context truncated: {truncated_count} messages removed, "
            f"{total_tokens}→{final_tokens} tokens",
            extra={
                "component": "memory.context_window",
                "truncated_count": truncated_count,
                "original_tokens": total_tokens,
                "final_tokens": final_tokens,
            },
        )

        return TruncationResult(
            messages=final_messages,
            truncated_count=truncated_count,
            original_tokens=total_tokens,
            final_tokens=final_tokens,
            summary=summary,
        )

    def get_messages_for_model(
        self,
        conversation: Conversation,
        include_system: bool = True,
    ) -> list[dict[str, str]]:
        """Get truncated messages as dicts ready for model input.

        Args:
            conversation: Conversation to process.
            include_system: Whether to include system messages.

        Returns:
            List of {'role': ..., 'content': ...} dicts.
        """
        result = self.truncate(conversation)
        messages = result.messages
        if not include_system:
            messages = [m for m in messages if m.role != "system"]
        return [m.to_dict() for m in messages]
