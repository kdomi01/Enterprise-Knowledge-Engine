import os
import shutil
from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, HTTPException, status
import structlog
from backend.app.schemas.ingest import IngestionResponse
from backend.app.api.deps import get_document_processor, get_vector_store
from backend.app.services.document_processor import DocumentProcessorService
from backend.app.services.vector_store import VectorStoreService

router = APIRouter()
logger = structlog.get_logger()

# Temporary local storage for staging incoming binaries safely
TMP_DIR = Path("/tmp" if os.name != "nt" else os.environ.get("TEMP", "C:\\Temp")) / "knowledge_engine_uploads"
TMP_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload", response_model=IngestionResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile,
    processor: DocumentProcessorService = Depends(get_document_processor),
    vector_store: VectorStoreService = Depends(get_vector_store)
):
    """
    Accepts a PDF file upload, streams it to temporary storage, chunks it using 
    a Parent-Child strategy, and indexes the text vectors into Qdrant.
    """
    # 1. Enforce strict file extensions (German production systems value strict boundary validation)
    if not file.filename.lower().endswith('.pdf'):
        logger.warning("Rejected file submission due to incorrect extension", filename=file.filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Only standard PDF binaries are accepted."
        )

    target_path = TMP_DIR / file.filename
    logger.info("Streaming file upload to disk staging area", filename=file.filename)
    
    try:
        # 2. Stream chunked byte writes to prevent memory spikes on large uploads
        with target_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 3. Process document chunks (extract and run Parent-Child splitting)
        records = processor.process_document(
            file_path=str(target_path), 
            source_name=file.filename
        )
        
        # Grab our preview text safely before uploading
        preview_sample = records[0]["child_text"] if records else "Empty payload text"

        # 4. Ingest generated semantic points into Qdrant
        vector_store.upsert_processed_documents(records)
        
        total_chunks_created = len(records)

        # Unified response returning both the frontend preview metadata and system status
        return {
            "status": "success",
            "filename": file.filename,
            "total_chunks": total_chunks_created,
            "collection_name": "enterprise_knowledge",
            "preview_text": preview_sample
        }
        
    except Exception as e:
        logger.error("Ingestion pipeline caught critical endpoint failure", filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=f"Ingestion pipeline failed: {str(e)}")
        
    finally:
        # 5. Clean up disk workspace (Crucial for enterprise compliance and resource health)
        if target_path.exists():
            os.remove(target_path)
            logger.debug("Cleaned up local extraction staging file", filename=file.filename)