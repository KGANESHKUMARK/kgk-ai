"""KGK AI Chat package — controller, prompts, and conversation management."""

from app.chat.controller import KGKChatController, ChatResponseData
from app.chat.conversation import Conversation, Message
from app.chat.prompts import get_system_prompt
from app.chat.session import ChatSessionManager, get_session_manager

__all__ = [
    "KGKChatController",
    "ChatResponseData",
    "Conversation",
    "Message",
    "get_system_prompt",
    "ChatSessionManager",
    "get_session_manager",
]
