"""KGK AI API Routes — FastAPI route definitions.

Defines the chat, health, and tool endpoints. Routes are mounted
on a FastAPI application in app/main.py.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import ChatRequest, ChatResponse, HealthResponse, ToolListResponse
from app.api.health import router as health_router

router = APIRouter()
router.include_router(health_router)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a chat message and return a response.

    This endpoint is a stub for Phase 1. The full chat controller
    will be implemented in Phase 4 (KGK Chat).

    Args:
        request: Chat request with message and options.

    Returns:
        ChatResponse with the generated response.
    """
    return ChatResponse(
        response="KGK AI chat is not yet fully implemented. This is a Phase 1 stub.",
        conversation_id=request.conversation_id,
        model="not_loaded",
    )


@router.get("/tools", response_model=ToolListResponse)
async def list_tools() -> ToolListResponse:
    """List available tools.

    Returns:
        ToolListResponse with available tool information.
    """
    try:
        from app.tools.registry import tool_registry

        tools = [t.__dict__ for t in tool_registry.list_tools(public_only=True)]
        return ToolListResponse(tools=tools)
    except Exception:
        return ToolListResponse(tools=[])
