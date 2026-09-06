"""KGK AI — Hugging Face Space entry point.

This file is used by Hugging Face Spaces to launch the application.
It imports from the app/ package and starts the Gradio UI.

For local development, use: python -m app.main
For API mode, use: python -m app.main --api
"""

from __future__ import annotations

from app.main import run_ui

if __name__ == "__main__":
    run_ui()
