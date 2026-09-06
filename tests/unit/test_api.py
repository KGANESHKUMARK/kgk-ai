"""Unit tests for KGK AI API components.

Tests cover:
- Schemas: ChatRequest, ChatResponse, ErrorResponse, WebSocketMessage validation
- Auth: API key verification (enabled/disabled, valid/invalid/missing)
- Routes: health, chat, tools, conversations endpoints
- WebSocket: connection, chat, ping, error handling
- CORS: middleware configuration
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    ToolListResponse,
    WebSocketMessage,
)
from app.main import create_fastapi_app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    app = create_fastapi_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_client():
    """Create a test client with API auth enabled."""
    with patch("app.api.auth.get_settings") as mock_settings:
        settings = MagicMock()
        settings.enable_api_auth = True
        settings.api_key = "test-secret-key"
        settings.enable_cors = True
        settings.cors_origins = "*"
        mock_settings.return_value = settings
        app = create_fastapi_app()
        with TestClient(app) as client:
            yield client


class TestSchemas:
    """Tests for Pydantic schema validation."""

    def test_chat_request_defaults(self):
        req = ChatRequest(message="Hello")
        assert req.message == "Hello"
        assert req.conversation_id == "default"
        assert req.stream is False
        assert req.use_memory is True
        assert req.use_rag is True
        assert req.use_tools is True

    def test_chat_request_validation_empty_message(self):
        with pytest.raises(ValueError):
            ChatRequest(message="")

    def test_chat_request_validation_too_long(self):
        with pytest.raises(ValueError):
            ChatRequest(message="a" * 10001)

    def test_chat_request_custom_values(self):
        req = ChatRequest(
            message="test",
            conversation_id="conv123",
            stream=True,
            use_memory=False,
            use_rag=False,
            use_tools=False,
        )
        assert req.conversation_id == "conv123"
        assert req.stream is True
        assert req.use_memory is False

    def test_chat_response_defaults(self):
        resp = ChatResponse(response="Hello back")
        assert resp.response == "Hello back"
        assert resp.conversation_id == "default"
        assert resp.sources == []
        assert resp.tools_used == []
        assert resp.model == ""
        assert resp.latency_ms == 0.0

    def test_error_response(self):
        err = ErrorResponse(error="test_error", message="Something went wrong")
        assert err.error == "test_error"
        assert err.message == "Something went wrong"
        assert err.request_id == ""

    def test_websocket_message_defaults(self):
        msg = WebSocketMessage()
        assert msg.type == "chat"
        assert msg.message == ""
        assert msg.conversation_id == "default"

    def test_websocket_message_ping(self):
        msg = WebSocketMessage(type="ping")
        assert msg.type == "ping"

    def test_health_response(self):
        resp = HealthResponse(status="healthy", model_loaded=True)
        assert resp.status == "healthy"
        assert resp.model_loaded is True
        assert resp.rag_enabled is False

    def test_tool_list_response(self):
        resp = ToolListResponse(tools=[{"name": "calculator"}])
        assert len(resp.tools) == 1
        assert resp.tools[0]["name"] == "calculator"


class TestAuth:
    """Tests for API key authentication."""

    def test_auth_disabled_allows_access(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_auth_enabled_missing_key(self, auth_client):
        response = auth_client.get("/api/v1/tools")
        assert response.status_code == 401
        assert "API key required" in response.json()["detail"]

    def test_auth_enabled_invalid_key(self, auth_client):
        response = auth_client.get(
            "/api/v1/tools",
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]

    def test_auth_enabled_valid_key(self, auth_client):
        response = auth_client.get(
            "/api/v1/tools",
            headers={"X-API-Key": "test-secret-key"},
        )
        assert response.status_code == 200

    def test_auth_enabled_chat_with_valid_key(self, auth_client):
        response = auth_client.post(
            "/api/v1/chat",
            json={"message": "Hello"},
            headers={"X-API-Key": "test-secret-key"},
        )
        assert response.status_code == 200


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_returns_200(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_has_status(self, client):
        response = client.get("/api/v1/health")
        data = response.json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded")

    def test_health_has_model_loaded(self, client):
        response = client.get("/api/v1/health")
        data = response.json()
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)

    def test_health_has_component_flags(self, client):
        response = client.get("/api/v1/health")
        data = response.json()
        assert "rag_enabled" in data
        assert "memory_enabled" in data
        assert "tools_enabled" in data


class TestChatEndpoint:
    """Tests for the chat endpoint."""

    def test_chat_returns_response(self, client):
        response = client.post(
            "/api/v1/chat",
            json={"message": "Hello"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "conversation_id" in data
        assert "latency_ms" in data

    def test_chat_with_conversation_id(self, client):
        response = client.post(
            "/api/v1/chat",
            json={"message": "Hello", "conversation_id": "test-conv"},
        )
        assert response.status_code == 200
        assert response.json()["conversation_id"] == "test-conv"

    def test_chat_empty_message_rejected(self, client):
        response = client.post(
            "/api/v1/chat",
            json={"message": ""},
        )
        assert response.status_code == 422

    def test_chat_with_flags(self, client):
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Hello",
                "use_memory": False,
                "use_rag": False,
                "use_tools": False,
            },
        )
        assert response.status_code == 200

    def test_chat_stream_returns_sse(self, client):
        response = client.post(
            "/api/v1/chat",
            json={"message": "Hello", "stream": True},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")


class TestToolsEndpoint:
    """Tests for the tools endpoint."""

    def test_tools_returns_list(self, client):
        response = client.get("/api/v1/tools")
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert isinstance(data["tools"], list)


class TestConversationsEndpoint:
    """Tests for conversation management endpoints."""

    def test_list_conversations(self, client):
        response = client.get("/api/v1/conversations")
        assert response.status_code == 200
        data = response.json()
        assert "conversations" in data

    def test_clear_conversation(self, client):
        response = client.delete("/api/v1/conversations/test-conv")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert data["conversation_id"] == "test-conv"

    def test_get_history_nonexistent(self, client):
        response = client.get("/api/v1/conversations/nonexistent/history")
        assert response.status_code == 200
        data = response.json()
        assert data["messages"] == []

    def test_get_history_after_chat(self, client):
        client.post("/api/v1/chat", json={"message": "Hello", "conversation_id": "hist-test"})
        response = client.get("/api/v1/conversations/hist-test/history")
        assert response.status_code == 200
        data = response.json()
        assert len(data["messages"]) > 0


class TestWebSocket:
    """Tests for the WebSocket chat endpoint."""

    def test_websocket_ping(self, client):
        with client.websocket_connect("/api/v1/ws/chat") as ws:
            ws.send_text(json.dumps({"type": "ping"}))
            data = ws.receive_json()
            assert data["type"] == "pong"

    def test_websocket_invalid_json(self, client):
        with client.websocket_connect("/api/v1/ws/chat") as ws:
            ws.send_text("not valid json")
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "Invalid message format" in data["message"]

    def test_websocket_unknown_type(self, client):
        with client.websocket_connect("/api/v1/ws/chat") as ws:
            ws.send_text(json.dumps({"type": "unknown"}))
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "Unknown message type" in data["message"]

    def test_websocket_empty_message(self, client):
        with client.websocket_connect("/api/v1/ws/chat") as ws:
            ws.send_text(json.dumps({"type": "chat", "message": "  "}))
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "Empty message" in data["message"]


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_returns_info(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "KGK AI"
        assert "version" in data
        assert "docs" in data
        assert "health" in data


class TestCORS:
    """Tests for CORS middleware."""

    def test_cors_headers_present(self, client):
        response = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    def test_cors_allows_all_origins(self, client):
        response = client.get(
            "/api/v1/health",
            headers={"Origin": "http://example.com"},
        )
        assert response.status_code == 200
