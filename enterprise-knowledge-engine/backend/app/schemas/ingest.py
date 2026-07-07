from pydantic import BaseModel, Field
from typing import List, Dict, Any

class IngestionChunkMetadata(BaseModel):
    """
    Validates structural tracking metadata appended to a processed document chunk.
    """
    source: str = Field(..., description="The filename or path of the source document.")
    chunk_index: int = Field(..., ge=0, description="The sequential order of the chunk within the document.")
    parent_id: str = Field(..., description="A unique identifier tracking back to the original full document node.")

class IngestionRecord(BaseModel):
    """
    Validates a single text segment ready to be embedded and indexed into Qdrant.
    """
    id: str = Field(..., description="Unique UUID generated for this specific chunk point.")
    text: str = Field(..., min_length=1, description="The cleaned textual context extracted from the document.")
    source: str = Field(..., description="Duplicate tracking string for direct root mapping queries.")
    parent_id: str = Field(..., description="Structural tracking ID for hierarchical layout rebuilding.")
    chunk_index: int = Field(..., ge=0, description="The positional order metric.")

class IngestionResponse(BaseModel):
    """
    The standardized API payload schema returned after a successful document ingestion cycle.
    """
    status: str = Field("success", description="The overall processing execution status.")
    filename: str = Field(..., description="The name of the file processed.")
    total_chunks: int = Field(..., description="The total number of vector payloads generated.")
    collection_name: str = Field(..., description="The target Qdrant collection where elements were stored.")
    preview_text: str = Field(..., description="A snippet of the first 100 characters of the processed text for quick reference.")