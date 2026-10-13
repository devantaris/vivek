"""Logging setup for PJ-03 experiments."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "pj03", level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a console handler attached exactly once.

    Args:
        name: Logger name (child loggers under ``pj03`` are fine).
        level: Logging level for the handler.

    Returns:
        A configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def add_file_handler(logger: logging.Logger, log_file: str | Path) -> None:
    """Also write ``logger`` output to ``log_file`` (parents created)."""
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    logger.addHandler(handler)
