from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    """
    Tracks the internal runtime state of the LangGraph execution graph.
    """
    query: str                  # The original user question
    retrieved_documents: List[Dict[str, Any]]  # Text chunks pulled from hybrid search
    reranked_documents: List[Dict[str, Any]]  # Cross-encoder refined chunks
    generation: str             # The final synthesized text response from the system
    steps: List[str]            # Log tracing of visited nodes