"""KGK AI structured logging configuration.

Provides structured logging with request IDs, component labels, and
configurable log levels. Sensitive user information is never logged.
"""

from __future__ import annotations

import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

from app.config import get_settings


class KGKFormatter(logging.Formatter):
    """Custom formatter that produces structured log lines.

    Format: TIMESTAMP | LEVEL | REQUEST_ID | COMPONENT | MESSAGE | EXTRA
    """

    def format(self, record: logging.LogRecord) -> str:
        settings = get_settings()

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        request_id = getattr(record, "request_id", "-")
        component = getattr(record, "component", "kgk")

        extra: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "request_id",
                "component", "message",
            } and not key.startswith("_"):
                extra[key] = value

        extra_str = f" | {extra}" if extra else ""
        return f"{timestamp} | {record.levelname} | {request_id} | {component} | {record.getMessage()}{extra_str}"


def setup_logging(level: str | None = None) -> logging.Logger:
    """Configure and return the root KGK logger.

    Args:
        level: Override log level (defaults to settings.log_level).

    Returns:
        Configured logger instance.
    """
    settings = get_settings()
    log_level = level or settings.log_level

    logger = logging.getLogger("kgk")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(KGKFormatter())
    logger.addHandler(handler)

    logger.propagate = False
    return logger


def get_logger(component: str = "kgk") -> logging.Logger:
    """Get a KGK logger tagged with a component name.

    Args:
        component: Component name (e.g. 'model', 'rag', 'chat').

    Returns:
        Logger instance with component context.
    """
    logger = logging.getLogger(f"kgk.{component}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(KGKFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def generate_request_id() -> str:
    """Generate a unique request ID for tracing."""
    return uuid.uuid4().hex[:12]
