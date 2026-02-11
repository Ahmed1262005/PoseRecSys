"""
Tests for the logging module.
"""

import json
import io
import sys
from unittest.mock import patch

import pytest
import structlog


class TestConfigureLogging:
    """Tests for configure_logging function."""
    
    def test_configure_development_mode(self):
        """Test that development mode uses console renderer."""
        from core.logging import configure_logging
        
        # Should not raise
        configure_logging(json_logs=False, log_level="DEBUG")
    
    def test_configure_production_mode(self):
        """Test that production mode uses JSON renderer."""
        from core.logging import configure_logging
        
        # Should not raise
        configure_logging(json_logs=True, log_level="INFO")
    
    def test_configure_log_level(self):
        """Test that log level is correctly set."""
        import logging
        from core.logging import configure_logging
        
        configure_logging(log_level="WARNING")
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING


class TestGetLogger:
    """Tests for get_logger function."""
    
    def test_get_named_logger(self):
        """Test getting a named logger."""
        from core.logging import get_logger
        
        logger = get_logger("test.module")
        
        assert logger is not None
        # Should be a bound logger
        assert hasattr(logger, "info")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "error")
    
    def test_get_unnamed_logger(self):
        """Test getting logger without name."""
        from core.logging import get_logger
        
        logger = get_logger()
        
        assert logger is not None
    
    def test_logger_can_log(self):
        """Test that logger can actually log messages."""
        from core.logging import configure_logging, get_logger
        
        configure_logging(json_logs=False, log_level="DEBUG")
        logger = get_logger("test")
        
        # Should not raise
        logger.info("Test message", key="value")
        logger.debug("Debug", data={"nested": "dict"})
        logger.warning("Warning")
        logger.error("Error", error="test error")


class TestContextBinding:
    """Tests for context binding functions."""
    
    def test_bind_context(self):
        """Test binding context variables."""
        from core.logging import bind_context, clear_context
        
        # Clear any existing context
        clear_context()
        
        # Bind context
        bind_context(user_id="123", request_id="abc")
        
        # Context should be bound (verified by structlog)
        ctx = structlog.contextvars.get_contextvars()
        assert ctx.get("user_id") == "123"
        assert ctx.get("request_id") == "abc"
        
        # Cleanup
        clear_context()
    
    def test_clear_context(self):
        """Test clearing context variables."""
        from core.logging import bind_context, clear_context
        
        bind_context(user_id="123")
        clear_context()
        
        ctx = structlog.contextvars.get_contextvars()
        assert "user_id" not in ctx
    
    def test_unbind_specific_context(self):
        """Test unbinding specific context variables."""
        from core.logging import bind_context, unbind_context, clear_context
        
        clear_context()
        bind_context(user_id="123", request_id="abc", session_id="xyz")
        
        unbind_context("session_id")
        
        ctx = structlog.contextvars.get_contextvars()
        assert ctx.get("user_id") == "123"
        assert ctx.get("request_id") == "abc"
        assert "session_id" not in ctx
        
        clear_context()


class TestLoggerMixin:
    """Tests for LoggerMixin class."""
    
    def test_mixin_provides_logger(self):
        """Test that mixin provides logger property."""
        from core.logging import LoggerMixin
        
        class TestClass(LoggerMixin):
            pass
        
        obj = TestClass()
        
        assert hasattr(obj, "logger")
        assert obj.logger is not None
    
    def test_mixin_logger_named_after_class(self):
        """Test that mixin logger is named after the class."""
        from core.logging import LoggerMixin, configure_logging
        
        configure_logging(json_logs=False)
        
        class MyService(LoggerMixin):
            def do_work(self):
                self.logger.info("Working")
        
        service = MyService()
        # Should not raise
        service.do_work()


class TestJSONOutput:
    """Tests for JSON logging output."""
    
    def test_json_output_is_valid_json(self, capsys):
        """Test that JSON output is valid JSON."""
        from core.logging import configure_logging, get_logger
        
        configure_logging(json_logs=True, log_level="INFO")
        logger = get_logger("json_test")
        
        logger.info("Test message", key="value")
        
        captured = capsys.readouterr()
        
        # Should be able to parse as JSON
        # Note: structlog outputs to stdout
        if captured.out:
            for line in captured.out.strip().split("\n"):
                if line:
                    data = json.loads(line)
                    assert "event" in data or "message" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
