"""KGK AI — Hugging Face Space entry point.

This file is used by Hugging Face Spaces to launch the application.
It imports from the app/ package and starts either the Gradio UI
(default) or the FastAPI server based on the KGK_MODE environment variable.

For local development, use: python -m app.main
For API mode, use: python -m app.main --api or set KGK_MODE=api
"""

from __future__ import annotations

import os

from app.logging_config import setup_logging, get_logger


def main():
    setup_logging()
    logger = get_logger("app")

    mode = os.environ.get("KGK_MODE", "ui").lower()

    if mode == "api":
        logger.info("Starting KGK AI in API mode (KGK_MODE=api)")
        from app.main import run_api
        run_api()
    else:
        logger.info("Starting KGK AI in UI mode (default)")
        from app.main import run_ui
        run_ui()


if __name__ == "__main__":
    main()
