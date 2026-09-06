"""KGK AI API Schemas — Pydantic models for request/response validation.

These schemas define the API contract for the FastAPI layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for the chat endpoint.

    Attributes:
        message: The user's message text.
        conversation_id: Optional conversation identifier.
        stream: Whether to stream the response.
    """

    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: str = Field(default="default")
    stream: bool = Field(default=False)


class ChatResponse(BaseModel):
    """Response body for the chat endpoint.

    Attributes:
        response: The generated response text.
        conversation_id: Conversation identifier.
        sources: List of sources cited (if RAG was used).
        tools_used: List of tools that were used.
        model: Model name that generated the response.
        latency_ms: Response latency in milliseconds.
    """

    response: str
    conversation_id: str = "default"
    sources: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    model: str = ""
    latency_ms: float = 0.0


class HealthResponse(BaseModel):
    """Response body for the health check endpoint.

    Attributes:
        status: Overall system status ('healthy', 'degraded', 'unhealthy').
        model_loaded: Whether the model is loaded.
        rag_enabled: Whether RAG is enabled.
        memory_enabled: Whether memory is enabled.
        tools_enabled: Whether tools are enabled.
        model_info: Optional model metadata.
    """

    status: str
    model_loaded: bool = False
    rag_enabled: bool = False
    memory_enabled: bool = False
    tools_enabled: bool = False
    model_info: Optional[dict[str, Any]] = None


class ToolListResponse(BaseModel):
    """Response body for the tool list endpoint.

    Attributes:
        tools: List of available tool info dicts.
    """

    tools: list[dict[str, Any]] = Field(default_factory=list)
