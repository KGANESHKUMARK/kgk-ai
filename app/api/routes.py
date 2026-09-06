"""KGK AI API Routes — FastAPI route definitions.

Defines the chat, health, tool, conversation, and WebSocket endpoints.
Routes are mounted on a FastAPI application in app/main.py.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from app.api.auth import verify_api_key
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    ToolListResponse,
    WebSocketMessage,
)
from app.api.health import router as health_router
from app.chat.session import get_session_manager
from app.logging_config import get_logger

logger = get_logger("api.routes")

router = APIRouter()
router.include_router(health_router)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    """Process a chat message and return a response.

    If request.stream is True, returns a StreamingResponse with
    Server-Sent Events. Otherwise returns a ChatResponse JSON.

    Args:
        request: Chat request with message and options.
        api_key: Verified API key.

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
        use_memory=request.use_memory,
        use_rag=request.use_rag,
        use_tools=request.use_tools,
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
            use_memory=request.use_memory,
            use_rag=request.use_rag,
            use_tools=request.use_tools,
        ):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: [ERROR: {str(e)}]\n\n"


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat streaming.

    Accepts JSON messages with the WebSocketMessage schema.
    Streams response chunks back as JSON messages.

    Message format (incoming):
        {"type": "chat", "message": "Hello", "conversation_id": "default", ...}
        {"type": "ping"}

    Message format (outgoing):
        {"type": "chunk", "content": "Hello"}
        {"type": "done", "conversation_id": "default"}
        {"type": "error", "message": "..."}
        {"type": "pong"}
    """
    await websocket.accept()

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                data = json.loads(raw)
                msg = WebSocketMessage(**data)
            except (json.JSONDecodeError, ValueError) as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Invalid message format: {e}",
                })
                continue

            if msg.type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if msg.type != "chat":
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg.type}",
                })
                continue

            if not msg.message.strip():
                await websocket.send_json({
                    "type": "error",
                    "message": "Empty message.",
                })
                continue

            session_mgr = get_session_manager()

            try:
                for chunk in session_mgr.stream_message(
                    message=msg.message,
                    conversation_id=msg.conversation_id,
                    use_memory=msg.use_memory,
                    use_rag=msg.use_rag,
                    use_tools=msg.use_tools,
                ):
                    await websocket.send_json({
                        "type": "chunk",
                        "content": chunk,
                        "conversation_id": msg.conversation_id,
                    })

                await websocket.send_json({
                    "type": "done",
                    "conversation_id": msg.conversation_id,
                })
            except Exception as e:
                logger.error(f"WebSocket streaming error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                    "conversation_id": msg.conversation_id,
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


@router.get("/tools", response_model=ToolListResponse)
async def list_tools(api_key: str = Depends(verify_api_key)) -> ToolListResponse:
    """List available tools.

    Args:
        api_key: Verified API key.

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
async def list_conversations(api_key: str = Depends(verify_api_key)) -> dict:
    """List all active conversations.

    Args:
        api_key: Verified API key.

    Returns:
        Dict with list of conversation IDs.
    """
    session_mgr = get_session_manager()
    return {"conversations": session_mgr.list_sessions()}


@router.delete("/conversations/{conversation_id}")
async def clear_conversation(conversation_id: str, api_key: str = Depends(verify_api_key)) -> dict:
    """Clear a conversation's history.

    Args:
        conversation_id: Conversation identifier.
        api_key: Verified API key.

    Returns:
        Dict with success status.
    """
    session_mgr = get_session_manager()
    cleared = session_mgr.clear_session(conversation_id)
    return {"success": cleared, "conversation_id": conversation_id}


@router.get("/conversations/{conversation_id}/history")
async def get_history(conversation_id: str, api_key: str = Depends(verify_api_key)) -> dict:
    """Get conversation history.

    Args:
        conversation_id: Conversation identifier.
        api_key: Verified API key.

    Returns:
        Dict with conversation history.
    """
    session_mgr = get_session_manager()
    history = session_mgr.get_history(conversation_id)
    return {"conversation_id": conversation_id, "messages": history}
