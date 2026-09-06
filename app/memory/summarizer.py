"""KGK AI Conversation Summarizer — Condense old messages.

Summarizes older conversation messages into a concise summary so that
context can be preserved without exceeding the token budget.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from app.logging_config import get_logger

if TYPE_CHECKING:
    from app.chat.conversation import Message

logger = get_logger("memory.summarizer")

_DEFAULT_MAX_SUMMARY_CHARS = 500


class ConversationSummarizer:
    """Summarizes conversation messages for context compression.

    Can use a model provider for intelligent summarization, or fall back
    to a simple extractive summary (first/last messages + counts).

    Attributes:
        model: Optional model provider for abstractive summarization.
        max_summary_chars: Maximum length of the summary text.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        max_summary_chars: int = _DEFAULT_MAX_SUMMARY_CHARS,
    ) -> None:
        self.model = model
        self.max_summary_chars = max_summary_chars

    def summarize(self, messages: list[Message]) -> str:
        """Summarize a list of messages.

        Args:
            messages: Messages to summarize.

        Returns:
            Summary text string.
        """
        if not messages:
            return ""

        if self.model is not None:
            try:
                return self._summarize_with_model(messages)
            except Exception as e:
                logger.warning(f"Model summarization failed, falling back to extractive: {e}")

        return self._extractive_summary(messages)

    def _summarize_with_model(self, messages: list[Message]) -> str:
        """Use the model provider to generate an abstractive summary.

        Args:
            messages: Messages to summarize.

        Returns:
            Model-generated summary.
        """
        conversation_text = "\n".join(
            f"{m.role.upper()}: {m.content}" for m in messages
        )

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a conversation summarizer. Summarize the following "
                    "conversation concisely, preserving key facts, decisions, and "
                    "context. Keep it under 200 words."
                ),
            },
            {
                "role": "user",
                "content": f"Summarize this conversation:\n\n{conversation_text}",
            },
        ]

        result = self.model.generate(prompt)
        summary = result.text.strip()

        if len(summary) > self.max_summary_chars:
            summary = summary[: self.max_summary_chars] + "..."

        logger.info(
            f"Model summary generated: {len(summary)} chars from {len(messages)} messages",
            extra={"component": "memory.summarizer", "message_count": len(messages)},
        )

        return summary

    def _extractive_summary(self, messages: list[Message]) -> str:
        """Create a simple extractive summary without a model.

        Takes the first user message, first assistant response, and
        the last exchange, with a count of messages in between.

        Args:
            messages: Messages to summarize.

        Returns:
            Extractive summary string.
        """
        if len(messages) <= 4:
            parts = []
            for m in messages:
                prefix = m.role.upper()
                content = m.content[:200]
                if len(m.content) > 200:
                    content += "..."
                parts.append(f"{prefix}: {content}")
            return "\n".join(parts)

        first_user = next((m for m in messages if m.role == "user"), None)
        first_assistant = next((m for m in messages if m.role == "assistant"), None)
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        last_assistant = next((m for m in reversed(messages) if m.role == "assistant"), None)

        parts = []
        if first_user:
            content = first_user.content[:200]
            if len(first_user.content) > 200:
                content += "..."
            parts.append(f"USER (first): {content}")

        if first_assistant:
            content = first_assistant.content[:200]
            if len(first_assistant.content) > 200:
                content += "..."
            parts.append(f"ASSISTANT (first): {content}")

        middle_count = len(messages) - (4 if last_user and last_assistant else 2)
        if middle_count > 0:
            parts.append(f"[... {middle_count} messages omitted ...]")

        if last_user and last_user != first_user:
            content = last_user.content[:200]
            if len(last_user.content) > 200:
                content += "..."
            parts.append(f"USER (recent): {content}")

        if last_assistant and last_assistant != first_assistant:
            content = last_assistant.content[:200]
            if len(last_assistant.content) > 200:
                content += "..."
            parts.append(f"ASSISTANT (recent): {content}")

        summary = "\n".join(parts)

        if len(summary) > self.max_summary_chars:
            summary = summary[: self.max_summary_chars] + "..."

        return summary
