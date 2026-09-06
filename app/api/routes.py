"""KGK AI API Routes — FastAPI route definitions.

Defines the chat, health, tool, and conversation endpoints.
Routes are mounted on a FastAPI application in app/main.py.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.schemas import ChatRequest, ChatResponse, HealthResponse, ToolListResponse
from app.api.health import router as health_router
from app.chat.session import get_session_manager
from app.logging_config import get_logger

logger = get_logger("api.routes")

router = APIRouter()
router.include_router(health_router)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return a response.

    If request.stream is True, returns a StreamingResponse with
    Server-Sent Events. Otherwise returns a ChatResponse JSON.

    Args:
        request: Chat request with message and options.

    Returns:
        ChatResponse or StreamingResponse.
    """
    session_mgr = get_session_manager()

    if request.stream:
        return StreamingResponse(
            _stream_response(session_mgr, request),
            media_type="text/event-stream",
        )

    result = session_mgr.send_message(
        message=request.message,
        conversation_id=request.conversation_id,
    )

    return ChatResponse(
        response=result.text,
        conversation_id=result.conversation_id,
        sources=result.sources,
        tools_used=result.tools_used,
        model=result.model,
        latency_ms=result.latency_ms,
    )


def _stream_response(session_mgr, request: ChatRequest):
    """Generator that yields SSE-formatted chunks for streaming.

    Args:
        session_mgr: ChatSessionManager instance.
        request: ChatRequest with message and options.

    Yields:
        Server-Sent Events formatted text chunks.
    """
    try:
        for chunk in session_mgr.stream_message(
            message=request.message,
            conversation_id=request.conversation_id,
        ):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: [ERROR: {str(e)}]\n\n"


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


@router.get("/conversations")
async def list_conversations() -> dict:
    """List all active conversations.

    Returns:
        Dict with list of conversation IDs.
    """
    session_mgr = get_session_manager()
    return {"conversations": session_mgr.list_sessions()}


@router.delete("/conversations/{conversation_id}")
async def clear_conversation(conversation_id: str) -> dict:
    """Clear a conversation's history.

    Args:
        conversation_id: Conversation identifier.

    Returns:
        Dict with success status.
    """
    session_mgr = get_session_manager()
    cleared = session_mgr.clear_session(conversation_id)
    return {"success": cleared, "conversation_id": conversation_id}


@router.get("/conversations/{conversation_id}/history")
async def get_history(conversation_id: str) -> dict:
    """Get conversation history.

    Args:
        conversation_id: Conversation identifier.

    Returns:
        Dict with conversation history.
    """
    session_mgr = get_session_manager()
    history = session_mgr.get_history(conversation_id)
    return {"conversation_id": conversation_id, "messages": history}
