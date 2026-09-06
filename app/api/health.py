"""KGK AI API Health Check — System health endpoints.

Provides endpoints to verify system status and component availability.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import HealthResponse
from app.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check the health of all KGK AI components.

    Returns:
        HealthResponse with status of each subsystem.
    """
    settings = get_settings()

    model_loaded = False
    model_info = None

    try:
        from app.models.registry import registry as model_registry

        if model_registry.active is not None:
            model_loaded = model_registry.active.health_check()
            info = model_registry.get_info()
            if info:
                model_info = {
                    "name": info.name,
                    "provider": info.provider,
                    "device": info.device,
                    "quantization": info.quantization,
                    "context_length": info.context_length,
                }
    except Exception:
        pass

    status = "healthy" if model_loaded else "degraded"
    if not model_loaded:
        status = "degraded"

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        rag_enabled=settings.enable_rag,
        memory_enabled=settings.enable_memory,
        tools_enabled=settings.enable_tools,
        model_info=model_info,
    )
