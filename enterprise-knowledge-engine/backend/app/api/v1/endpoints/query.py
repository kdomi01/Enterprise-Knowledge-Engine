from fastapi import APIRouter, HTTPException
import structlog
from backend.app.schemas.query import QueryRequest, QueryResponse
from backend.app.graph.workflow import rag_graph

router = APIRouter()
logger = structlog.get_logger()

@router.post("/engine", response_model=QueryResponse)
def execute_graph_query(payload: QueryRequest):
    """
    Injects queries directly into the compiled LangGraph execution tracks.
    Routes the workflow through local semantic tools and compiles an answers block.
    """
    logger.info("Graph query endpoint invoked", user_query=payload.query)
    
    try:
        # 1. Initialize the shared AgentState notebook dictionary
        initial_state = {
            "query": payload.query,
            "retrieved_documents": [],
            "generation": "",
            "steps": []
        }
        
        # 2. Fire the graph workflow engine cleanly down its tracks
        final_state = rag_graph.invoke(initial_state)
        
        # 3. Construct and map output directly to our strict output schema layout
        return QueryResponse(
            query=final_state.get("query"),
            generation=final_state.get("generation"),
            retrieved_context=final_state.get("retrieved_documents", []),
            visited_steps=final_state.get("steps", [])
        )
        
    except Exception as e:
        logger.error("LangGraph pipeline execution encountered a fault", error=str(e))
        raise HTTPException(
            status_code=500, 
            detail=f"Graph runtime error: {str(e)}"
        )