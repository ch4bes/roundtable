"""
Centralised logging configuration for Roundtable.

Call ``setup_logging()`` once at application startup (in ``main()``).
All other modules obtain their logger via ``logging.getLogger(__name__)``.
"""

import logging
import sys

_FMT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
_DATE = "%H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
) -> None:
    """Configure the root logger.

    Args:
        level:    Minimum log level (e.g. ``logging.DEBUG``).
        log_file: Optional path to write logs to in addition to stderr.
    """
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stderr),
    ]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    fmt = logging.Formatter(_FMT, _DATE)
    for h in handlers:
        h.setFormatter(fmt)

    logging.basicConfig(level=level, handlers=handlers, force=True)
    # Silence noisy third-party loggers at WARNING unless debugging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
