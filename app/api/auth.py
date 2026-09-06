"""KGK AI API Authentication — API key-based auth dependency.

Provides a FastAPI dependency that validates API keys from request headers.
When authentication is enabled, all API endpoints require a valid API key.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger("api.auth")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: Optional[str] = Security(_api_key_header),
) -> str:
    """Verify the API key from the X-API-Key header.

    If authentication is disabled in config, all requests are allowed.
    If enabled, the header must match the configured API key.

    Args:
        api_key: API key from request header.

    Returns:
        The verified API key.

    Raises:
        HTTPException: 401 if API key is missing or invalid.
    """
    settings = get_settings()

    if not settings.enable_api_auth:
        return "anonymous"

    if not settings.api_key:
        logger.warning("API auth enabled but no API key configured")
        return "anonymous"

    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide it in the X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if api_key != settings.api_key:
        logger.warning("Invalid API key attempt")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key


async def optional_api_key(
    request: Request,
) -> str:
    """Optional API key verification — skips auth for health endpoints.

    Args:
        request: Incoming request.

    Returns:
        API key or "anonymous".
    """
    settings = get_settings()

    if not settings.enable_api_auth or not settings.api_key:
        return "anonymous"

    api_key = request.headers.get("X-API-Key")
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide it in the X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if api_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key
