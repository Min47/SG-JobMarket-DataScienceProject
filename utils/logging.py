"""Logging utilities.

All scripts should call `configure_logging()` as early as possible.
Each process should call this once at startup.
"""

from __future__ import annotations

import glob
import logging
import os
from datetime import datetime
from pathlib import Path

MAX_LOG_FILES_DEFAULT = 10

def _cleanup_old_logs(log_dir: Path, max_logs: int) -> None:
    """Keep only the most recent N log files."""
    log_files = sorted(
        glob.glob(str(log_dir / "*.log")),
        key=os.path.getmtime,
        reverse=True,
    )
    for old_log in log_files[max_logs:]:
        try:
            os.remove(old_log)
        except OSError:
            pass


def configure_logging(
    *,
    service_name: str,
    level: str | None = None,
    log_dir: str | None = None,
    max_log_files: int = MAX_LOG_FILES_DEFAULT,
) -> logging.Logger:
    """Configure root logging with console and file output.
    
    This should be called once per process at startup.
    
    Args:
        service_name: Name of the service/component.
        level: Log level (defaults to LOG_LEVEL env var or INFO).
        log_dir: Directory for log files (defaults to 'logs').
        max_log_files: Maximum number of log files to retain.
    
    Returns:
        Configured logger instance.
    """
    resolved_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, resolved_level, logging.INFO)
    
    # Create logs directory
    log_path = Path(log_dir or "logs")
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{service_name}_{timestamp}.log"
    
    # Cleanup old logs before creating new one
    _cleanup_old_logs(log_path, max_log_files)
    
    # Configure root logger
    root_logger = logging.getLogger()
    
    # Clear existing handlers to allow reconfiguration
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)
    
    # Format for both handlers
    formatter = logging.Formatter(
        "%(asctime)s (%(levelname)s) | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    return logging.getLogger(service_name)

