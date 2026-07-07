from contextlib import asynccontextmanager
from fastapi import FastAPI
import structlog
from backend.app.api.v1.router import api_router
from backend.app.api.v1.endpoints import ingest, search, query
from backend.app.core.config import get_settings
from backend.app.core.logging import setup_logging

# Initialize the central structured logger
setup_logging()
logger = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown lifecycle events.
    Initializes database connection pools.
    """
    logger.info(
        "Application startup sequence initiated",
        environment=settings.ENVIRONMENT,
        project=settings.PROJECT_NAME,
    )
    yield
    logger.info("Application shutdown sequence complete")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    lifespan=lifespan,
)

# Mount the modular v1 endpoint architecture router
app.include_router(api_router, prefix="/api/v1")
app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["Ingestion"])
app.include_router(search.router, prefix="/api/v1/search", tags=["Retrieval"])
app.include_router(query.router, prefix="/api/v1/query", tags=["Agent Graph"])

@app.get("/health", tags=["System"])
async def health_check():
    """ Simple status checkpoint for Docker health-checks and load-balancers. """
    logger.debug("Health check endpoint pinged")
    return {"status": "healthy", "environment": settings.ENVIRONMENT}