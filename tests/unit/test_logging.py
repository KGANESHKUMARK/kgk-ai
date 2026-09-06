"""Unit tests for KGK AI logging configuration."""

import logging
import pytest
from app.logging_config import setup_logging, get_logger, generate_request_id, KGKFormatter


class TestLogging:
    """Tests for logging configuration."""

    def test_setup_logging(self):
        logger = setup_logging("DEBUG")
        assert logger.level == logging.DEBUG

    def test_get_logger_with_component(self):
        logger = get_logger("test_component")
        assert logger.name == "kgk.test_component"

    def test_generate_request_id(self):
        rid = generate_request_id()
        assert len(rid) == 12
        assert isinstance(rid, str)

    def test_request_id_uniqueness(self):
        ids = {generate_request_id() for _ in range(100)}
        assert len(ids) == 100  # All unique

    def test_formatter_produces_structured_output(self):
        formatter = KGKFormatter()
        record = logging.LogRecord(
            name="kgk.test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="Test message",
            args=None,
            exc_info=None,
        )
        record.request_id = "abc123"
        record.component = "test"
        output = formatter.format(record)
        assert "abc123" in output
        assert "test" in output
        assert "Test message" in output
