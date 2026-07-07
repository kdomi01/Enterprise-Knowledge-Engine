from fastapi import APIRouter, Depends, HTTPException, Query
import structlog
from backend.app.services.vector_store import VectorStoreService

router = APIRouter()
logger = structlog.get_logger()

@router.get("/query")
def execute_hybrid_search(
    q: str = Query(..., description="The search query or question to ask the knowledge engine"),
    limit: int = Query(5, ge=1, le=20, description="Number of relevant document chunks to retrieve"),
    vector_store: VectorStoreService = Depends()
):
    """
    Performs a local hybrid (dense + sparse) semantic search across indexed enterprise assets.
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Search query string cannot be empty.")
        
    try:
        results = vector_store.hybrid_search(query_text=q, limit=limit)
        return {
            "query": q,
            "results_count": len(results),
            "matches": results
        }
    except Exception as e:
        logger.error("Search endpoint execution failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Search operation failed: {str(e)}")