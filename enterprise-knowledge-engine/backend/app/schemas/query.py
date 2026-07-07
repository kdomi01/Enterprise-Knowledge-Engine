from pydantic import BaseModel, Field
from typing import List, Dict, Any

class QueryRequest(BaseModel):
    """
    Validates incoming user questions sent to the graph orchestrator.
    """
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=2000, 
        description="The natural language question to process.",
        examples=["What are the computer skills listed in the document?"]
    )

class DocumentMatch(BaseModel):
    """
    Validates the structure of a retrieved context chunk.
    """
    score: float = Field(..., description="The Reciprocal Rank Fusion relevance score.")
    text: str | None = Field(None, description="The textual chunk context content.")
    source: str | None = Field(None, description="The originating document filename source.")
    parent_id: str | None = Field(None, description="The structural parent document node tracking ID.")

class QueryResponse(BaseModel):
    """
    Structured data layout returned to the frontend client.
    """
    query: str = Field(..., description="The original evaluated query.")
    generation: str = Field(..., description="The synchronized response output text.")
    retrieved_context: List[DocumentMatch] = Field(
        default=[], 
        description="The precise context blocks used by the graph state machine."
    )
    visited_steps: List[str] = Field(
        default=[], 
        description="Execution graph tracking logs for audit transparency."
    )