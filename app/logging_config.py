"""Central loguru configuration.

We keep logging configuration in one place so:

* Every module just does ``from loguru import logger`` and gets the same sink,
  format, and level without having to initialise anything itself.
* The level is driven entirely by ``Settings.log_level`` (``LOG_LEVEL`` env
  var), so operators can flip between ``DEBUG`` and ``INFO`` without code
  changes.
* Tests that want to capture logs can call :func:`configure_logging` in a
  fixture without worrying about handler duplication — we always remove the
  default handler before adding ours.

``configure_logging`` is idempotent: calling it multiple times resets the
sinks to the current settings rather than stacking handlers on top of each
other.
"""
from __future__ import annotations

import sys

from loguru import logger

from app.config import get_settings

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
    "| <level>{level: <8}</level> "
    "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
    "- <level>{message}</level>"
)


def configure_logging(level: str | None = None) -> None:
    """Reset loguru sinks and install a single stderr sink at the chosen level.

    Args:
        level: Explicit level name (e.g. ``"DEBUG"``). If ``None`` we read from
            ``Settings.log_level`` so all callers share the same default.
    """
    resolved = level or get_settings().log_level
    logger.remove()
    logger.add(
        sys.stderr,
        level=resolved,
        format=_CONSOLE_FORMAT,
        backtrace=False,
        diagnose=False,
        enqueue=False,
    )
    logger.debug("loguru configured at level={}", resolved)
