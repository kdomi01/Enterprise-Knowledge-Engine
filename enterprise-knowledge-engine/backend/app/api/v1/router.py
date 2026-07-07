from fastapi import APIRouter
from backend.app.api.v1.endpoints import ingest

api_router = APIRouter()

# Mount endpoints onto functional domains
api_router.include_router(ingest.router, prefix="/ingest", tags=["Ingestion Pipeline"])