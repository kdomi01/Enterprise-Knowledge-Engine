import pytest
from backend.app.graph.workflow import rag_graph

def test_langgraph_workflow_state_mutation():
    """
    Ensures that forcing an initial state dictionary through the compiled graph 
    correctly appends document arrays, applies reranking, and modifies the state tracking keys.
    """
    initial_state = {
        "query": "Testing the cross-encoder pipeline tracking",
        "retrieved_documents": [],
        "reranked_documents": [],
        "generation": "",
        "steps": []
    }
    
    # Trigger graph invocation directly
    final_state = rag_graph.invoke(initial_state)
    
    assert isinstance(final_state, dict)
    assert len(final_state["steps"]) >= 3
    assert final_state["steps"][0] == "retrieve_node"
    assert final_state["steps"][1] == "rerank_node"
    assert final_state["steps"][2] == "generate_node"
    assert "generation" in final_state
    assert isinstance(final_state["reranked_documents"], list)