"""KGK AI API package — FastAPI routes, schemas, and authentication."""

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    ToolListResponse,
    WebSocketMessage,
)
from app.api.auth import verify_api_key

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ErrorResponse",
    "HealthResponse",
    "ToolListResponse",
    "WebSocketMessage",
    "verify_api_key",
]
