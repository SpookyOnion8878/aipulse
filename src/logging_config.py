"""Central logging configuration."""

from __future__ import annotations

import logging
import os
import sys


def configure_logging() -> None:
    """Configure stdout logging once at the application boundary."""

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
