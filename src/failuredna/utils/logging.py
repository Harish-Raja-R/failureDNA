"""Structured logging utility for FailureDNA pipelines."""

import logging
import os
import sys
from typing import Optional


def get_logger(
    name: str = "failuredna",
    level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a structured logger.

    Args:
        name: Name of the logger (e.g. module __name__).
        level: Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').
            Defaults to FAILUREDNA_LOG_LEVEL environment variable or 'INFO'.
        log_file: Optional path to write log output to a file.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    log_level_str = level or os.getenv("FAILUREDNA_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
