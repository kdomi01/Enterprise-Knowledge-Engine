import json
import os
import structlog
from backend.app.services.vector_store import VectorStoreService
from backend.app.services.document_processor import DocumentProcessingService

logger = structlog.get_logger()

def auto_index_test_documents(test_set_path: str):
    """
    Dynamically identifies, parses, and indexes documents listed in the test set
    using production processing logic to prevent manual data seeding.
    """
    if not os.path.exists(test_set_path):
        logger.error("Test set missing. Cannot resolve source documents.")
        return

    with open(test_set_path, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    # Gather unique source files required for this evaluation run
    source_files = list({test_case["source_file"] for test_case in test_set if "source_file" in test_case})
    
    if not source_files:
        logger.warning("No source files specified in test set. Skipping dynamic indexing.")
        return

    v_store = VectorStoreService()
    
    # Wipe old data to guarantee zero context contamination between test runs
    logger.info("Wiping evaluation vector collection...")
    try:
        v_store.qdrant_client.delete_collection(v_store.collection_name)
        v_store._create_collection_if_not_exists()
    except Exception as e:
        logger.error("Failed to reset collection layout", error=str(e))

   # Process each document through your actual chunking workflow
    for file_path in source_files:
        if not os.path.exists(file_path):
            logger.error(f"Target evaluation document missing from disk: {file_path}")
            continue

        logger.info(f"Processing evaluation asset via production processor", file_path=file_path)
        
        try:
            # ─── USE REAL PRODUCTION PIPELINE ─────────────────────────────────
            # Use DocumentProcessingService to correctly extract and chunk the PDF
            processor = DocumentProcessingService()
            records = processor.process_document(file_path, source_name=os.path.basename(file_path))
            # ──────────────────────────────────────────────────────────────────
            
            # Re-map your production records to match Qdrant upsert schema structure expectations
            chunks = []
            for i, record in enumerate(records):
                # Safely capture text data from your processor's output schema
                chunk_text = record.get("parent_text") or record.get("text") or record.get("content")
                if not chunk_text:
                    continue
                    
                chunks.append({
                    "text": chunk_text.strip(),
                    "source": os.path.basename(file_path),
                    "chunk_index": record.get("chunk_index") or i,
                    "parent_id": record.get("parent_id") or f"eval_{os.path.basename(file_path)}"
                })

            # Ingest straight through your production BGE-M3 multi-vector setup
            if chunks:
                logger.info(f"Upserting {len(chunks)} production chunks into Qdrant...")
                v_store.upsert_processed_documents(chunks)
                
        except Exception as process_error:
            logger.error(f"Production document processor failed on file: {file_path}", error=str(process_error))
            continue