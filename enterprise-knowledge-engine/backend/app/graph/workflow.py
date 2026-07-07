from langgraph.graph import StateGraph, END
from backend.app.graph.state import AgentState
from backend.app.graph.nodes import retrieve_node, rerank_node, generate_node

def compile_workflow():
    """
    Configures, binds, and compiles the architectural nodes into an executable graph routing layout.
    """
    # 1. Initialize our State Graph with the core architecture shape
    workflow = StateGraph(AgentState)
    
    # 2. Map nodes to unique identity descriptors
    workflow.add_node("retrieve_documents", retrieve_node)
    workflow.add_node("rerank_context", rerank_node)
    workflow.add_node("generate_response", generate_node)
    
    # 3. Establish the linear flow execution vectors
    workflow.set_entry_point("retrieve_documents")
    workflow.add_edge("retrieve_documents", "rerank_context")
    workflow.add_edge("rerank_context", "generate_response")
    workflow.add_edge("generate_response", END)
    
    # 4. Compile the configuration engine layout
    return workflow.compile()

# Single-instantiated workflow instance to consume query requests
rag_graph = compile_workflow()