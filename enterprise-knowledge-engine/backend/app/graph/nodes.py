import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import structlog
from backend.app.graph.state import AgentState
from backend.app.graph.tools import retrieve_hybrid_documents
from backend.app.services.llm_generation import LLMGenerationService

logger = structlog.get_logger()

try:
    llm_service = LLMGenerationService()
except Exception as e:
    logger.error("Failed to spin up LLM service block", error=str(e))
    llm_service = None

# Native initialization of the Cross-Encoder model using pure Hugging Face classes
try:
    logger.info("Initializing local BGE-Reranker-v2-M3 via native Transformers backend...")
    model_id = "BAAI/bge-reranker-v2-m3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()  # Lock into inference evaluation mode
    
    # Run on local GPU if available, otherwise default to local CPU matrix lanes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("Reranker model loaded successfully", device=str(device))
except Exception as e:
    logger.error("Failed to spin up local native Cross-Encoder backend", error=str(e))
    model, tokenizer, device = None, None, None


def retrieve_node(state: AgentState) -> AgentState:
    """
    Node responsible for calling the hybrid vector search engine 
    and committing results directly into the agent memory state.
    """
    logger.info("Entering Node: Retrieve Documents")
    query = state["query"]
    
    # Fire our local hybrid search tool
    documents = retrieve_hybrid_documents(query=query)
    
    # Advance the graph memory ledger safely
    return {
        **state,
        "retrieved_documents": documents,
        "steps": state.get("steps", []) + ["retrieve_node"]
    }


def rerank_node(state: AgentState) -> AgentState:
    """
    Evaluates initial document matches against the query simultaneously 
    using the v2-m3 Cross-Encoder to optimize high-precision attention scores.
    """
    logger.info("Entering Node: Cross-Encoder Rerank Blocks")
    query = state["query"]
    docs = state.get("retrieved_documents", [])
    
    if not docs or model is None or tokenizer is None:
        logger.warning("Skipping rerank step: Data chunks absent or model uninitialized")
        return {**state, "reranked_documents": docs, "steps": state.get("steps", []) + ["rerank_node_skipped"]}

    # Build sequence pairs matching the cross-encoder execution design: [[Q, P1], [Q, P2]...]
    pairs = [[query, doc.get("text", "")] for doc in docs]
    
    try:
        # Perform explicit tensor tokenization over maximum token boundaries
        with torch.no_grad():
            inputs = tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                max_length=1024, # Optimized alignment context token window
                return_tensors="pt"
            ).to(device)
            
            # Extract logit classification values directly from the model head
            logits = model(**inputs).logits.view(-1).float().cpu().tolist()
            
        # Ensure outputs handle single-item queries correctly
        scores = [logits] if isinstance(logits, float) else logits
        
        # Merge precision scores back into document metadata dictionaries
        scored_docs = []
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)
            scored_docs.append(doc)
            
        # Sort chunks based on their updated cross-encoder attention relevance 
        sorted_docs = sorted(scored_docs, key=lambda x: x["rerank_score"], reverse=True)
        
        # Prune out matches with scores lower than -3.0
        filtered_docs = [d for d in sorted_docs if d["rerank_score"] > -3.0]
        
        # Fallback mechanism: If threshold filtering starves the node completely,
        # retain the top 2 highest-scoring contextual fragments to prevent empty generation state.
        if not filtered_docs and sorted_docs:
            logger.warning("Threshold filtering returned 0 results. Executing fallback to top-2 documents.")
            filtered_docs = sorted_docs[:2]
        
        logger.info(
            "Cross-Encoder native v2-m3 sorting complete", 
            input_count=len(docs), 
            retained_count=len(filtered_docs)
        )
        
        return {
            **state,
            "reranked_documents": filtered_docs,
            "steps": state.get("steps", []) + ["rerank_node"]
        }
        
    except Exception as e:
        logger.error("Native rerank token calculation phase errored", error=str(e))
        return {**state, "reranked_documents": docs, "steps": state.get("steps", []) + ["rerank_node_failed"]}
        
def generate_node(state: AgentState) -> AgentState:
    """
    Combines queries and high-confidence cross-encoder context pieces 
    into a structured local response layout.
    """
    logger.info("Entering Node: Synthesize Response")
    query = state["query"]
    docs = state.get("reranked_documents", [])
    
    if not docs:
        logger.warning("No context fragments passed to generation node")
        generation_output = "No relevant context documentation found to formulate an answer."
    elif llm_service is None:
        generation_output = "LLM Generation layer is offline or failed initialization parameters."
    else:
        # Fire our native transformers text generation engine!
        generation_output = llm_service.synthesize_answer(query=query, context_chunks=docs)
    
    return {
        **state,
        "generation": generation_output,
        "steps": state.get("steps", []) + ["generate_node"]
    }