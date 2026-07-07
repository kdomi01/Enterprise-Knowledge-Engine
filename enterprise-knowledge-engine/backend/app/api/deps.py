from backend.app.services.document_processor import DocumentProcessorService
from backend.app.services.vector_store import VectorStoreService

# Instantiate singletons for the backend session runtime
_document_processor = DocumentProcessorService()
_vector_store = VectorStoreService()


def get_document_processor() -> DocumentProcessorService:
    """Dependency provider for document text extraction and token chunking."""
    return _document_processor


def get_vector_store() -> VectorStoreService:
    """Dependency provider for Qdrant database interactions."""
    return _vector_store