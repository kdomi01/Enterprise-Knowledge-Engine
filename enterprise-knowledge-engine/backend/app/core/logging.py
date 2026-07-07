import logging
import sys
import structlog
from backend.app.core.config import get_settings

settings = get_settings()


def setup_logging() -> None:
    """
    Configures structlog to output structured JSON logs in production/staging
    and clean, colorized text logs for local development.
    """
    # 1. Map the string level from config to standard logging module integers
    log_level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    base_level = log_level_map.get(settings.LOG_LEVEL.lower(), logging.INFO)

    # 2. Define our processing pipeline
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    # 3. Environment check: JSON for containers/cloud, pretty color text for local dev
    if settings.ENVIRONMENT in ["production", "staging"]:
        # Standard production processors
        processors.extend(
            [
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer(),
            ]
        )
        handler = logging.StreamHandler(sys.stdout)
    else:
        # User-friendly local development processors (colored readable text)
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
        handler = logging.StreamHandler(sys.stdout)

    # 4. Bind processors to structlog configuration
    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # 5. Hook into standard Python logging so third-party logs pass through cleanly
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(base_level)

    # Mute chatty third-party libraries (like HTTP requests or vector clients)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)