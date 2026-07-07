import structlog
from typing import List, Dict, Any
from backend.app.services.vector_store import VectorStoreService

logger = structlog.get_logger()

def retrieve_hybrid_documents(query: str, limit: int = 4) -> List[Dict[str, Any]]:
    """
    Graph tool wrapped around the local Qdrant collection to fetch 
    hybrid-indexed matching text pieces.
    """
    logger.info("Graph tool executing document retrieval pipeline", query=query)
    try:
        vector_store = VectorStoreService()
        return vector_store.hybrid_search(query_text=query, limit=limit)
    except Exception as e:
        logger.error("Graph tool retrieval operation critically failed", error=str(e))
        return []