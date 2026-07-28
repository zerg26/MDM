"""Logging setup for the MDM pipeline.

Console output stays human-readable in development; set ``MDM_LOG_JSON=1`` to emit
structured JSON lines suitable for observability platforms.
"""
from __future__ import annotations

import json
import logging
import os

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Minimal structured JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once. JSON when ``MDM_LOG_JSON=1``, else plain text."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler()
    if os.getenv("MDM_LOG_JSON") == "1":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    root = logging.getLogger()
    root.setLevel(level)
    # Replace existing handlers so we don't double-log (e.g. after basicConfig).
    root.handlers = [handler]
    _CONFIGURED = True
