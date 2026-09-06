"""KGK AI Main Application — Entry point for local development.

Launches either the FastAPI server or the Gradio UI based on configuration.
For Hugging Face deployment, app.py is used instead (it imports from here).
"""

from __future__ import annotations

import sys

from app.config import get_settings
from app.logging_config import setup_logging, get_logger


def create_fastapi_app():
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance.
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from app.api.routes import router

    settings = get_settings()

    app = FastAPI(
        title="KGK AI API",
        description="AI Intelligence by KGK",
        version="0.1.0",
    )

    if settings.enable_cors:
        origins = [o.strip() for o in settings.cors_origins.split(",")]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(router, prefix="/api/v1")

    @app.get("/")
    async def root():
        return {
            "name": "KGK AI",
            "version": "0.1.0",
            "description": "AI Intelligence by KGK",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


def run_api() -> None:
    """Run the FastAPI server."""
    import uvicorn

    settings = get_settings()
    logger = get_logger("main")

    logger.info(
        f"Starting KGK AI API on {settings.api_host}:{settings.api_port}",
        extra={"component": "main"},
    )

    app = create_fastapi_app()
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


def run_ui() -> None:
    """Run the Gradio UI."""
    from app.ui.gradio_app import launch_ui

    settings = get_settings()
    logger = get_logger("main")

    logger.info("Starting KGK AI Gradio UI", extra={"component": "main"})

    launch_ui(host=settings.api_host, port=settings.api_port)


def main() -> None:
    """Main entry point.

    Usage:
        python -m app.main          # Launches UI (default)
        python -m app.main --api    # Launches API server
    """
    setup_logging()
    logger = get_logger("main")

    settings = get_settings()
    logger.info(
        f"KGK AI starting — model={settings.model_name}, device={settings.device}",
        extra={"component": "main", "model": settings.model_name, "device": settings.device},
    )

    mode = "ui"
    if "--api" in sys.argv:
        mode = "api"

    if mode == "api":
        run_api()
    else:
        run_ui()


if __name__ == "__main__":
    main()
