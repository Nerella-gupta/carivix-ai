"""Small shared utilities used by the CARIVIX AI services."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def setup_logger(
    name: str = "CARIVIX_AI",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create or return a configured project logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(console_handler)

    if log_file:
        parent = os.path.dirname(log_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger


def ensure_directory(path: str) -> str:
    """Create a directory when needed and return its path."""
    os.makedirs(path, exist_ok=True)
    return path


def get_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def load_config(path: str, base_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load a YAML or JSON configuration file."""
    config_path = path
    if base_dir and not os.path.isabs(config_path):
        config_path = os.path.join(base_dir, config_path)

    with open(config_path, "r", encoding="utf-8") as config_file:
        if config_path.lower().endswith(".json"):
            loaded = json.load(config_file)
        else:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("PyYAML is required to load YAML configuration files.") from exc
            loaded = yaml.safe_load(config_file)

    return loaded if isinstance(loaded, dict) else {}
