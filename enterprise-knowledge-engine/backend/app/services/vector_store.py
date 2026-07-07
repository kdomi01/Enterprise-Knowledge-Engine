import uuid
import structlog
from qdrant_client import QdrantClient
from qdrant_client.http import models
from FlagEmbedding import BGEM3FlagModel  # <-- Official multi-vector implementation
from backend.app.core.config import get_settings

settings = get_settings()
logger = structlog.get_logger()


class VectorStoreService:

    def __init__(self):
        self.qdrant_client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
            https=False,
            prefer_grpc=False,
        )

        logger.info("Loading local BGE-M3 embedding model onto device...")
        # BGEM3FlagModel natively routes dense + sparse operations
        self.embedding_model = BGEM3FlagModel(
            "BAAI/bge-m3", use_fp16=True
        )  # use_fp16=False if your CPU lacks half-precision

        self.collection_name = "enterprise_knowledge"
        self.vector_dimension = 1024

        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        try:
            if not self.qdrant_client.collection_exists(self.collection_name):
                logger.info(
                    "Creating fresh Hybrid Qdrant collection",
                    name=self.collection_name,
                )
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_dimension,
                        distance=models.Distance.COSINE,
                    ),
                    sparse_vectors_config={
                        "sparse-text": models.SparseVectorParams(
                            index=models.SparseIndexParams(on_disk=True)
                        )
                    },
                )
        except Exception as e:
            logger.error(
                "Failed to initialize Qdrant storage engine", error=str(e)
            )
            raise e

    def upsert_processed_documents(self, records: list[dict]):
        if not records:
            return

        logger.info(
            "Starting batch local vector embedding generation",
            total_records=len(records),
        )

        try:
            texts = []
            for r in records:
                # 1. Look for explicit known content fields first
                val = (
                    r.get("text") 
                    or r.get("content") 
                    or r.get("page_content") 
                    or r.get("chunk_text")
                )
                
                # 2. Backup fallback: Grab a string, but intentionally skip keys containing 'id'
                if val is None:
                    for k, v in r.items():
                        if isinstance(v, str) and len(v) > 5 and "id" not in k.lower():
                            val = v
                            break
                texts.append(val)

            if not texts or texts[0] is None:
                raise KeyError("Could not automatically isolate raw text chunk blocks.")
        except Exception as e:
            logger.error(
                "Failed to parse text fields from incoming records", error=str(e)
            )
            raise e
        
        try:
            # Generate BOTH dense and sparse vectors natively
            outputs = self.embedding_model.encode(
                texts, return_dense=True, return_sparse=True
            )

            dense_embeddings = outputs["dense_vecs"].tolist()
            sparse_embeddings = outputs["lexical_weights"]

            points = []
            for idx, (record, dense_vec, sparse_dict) in enumerate(
                zip(records, dense_embeddings, sparse_embeddings)
            ):
                # Use the exact same fallback text snippet for the database payload storage
                text_payload = texts[idx]
                
                # Dynamic fallback for finding the source file reference key
                source_payload = record.get("source") or record.get("filename") or record.get("metadata", {}).get("source") or "Unknown Source"
                
                point_id = record.get("id") or str(uuid.uuid4())

                # Format sparse tokens for Qdrant
                indices = [int(k) for k in sparse_dict.keys()]
                values = [float(v) for v in sparse_dict.values()]

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={
                            "": dense_vec,
                            "sparse-text": models.SparseVector(
                                indices=indices, values=values
                            ),
                        },
                        payload={
                            "text": text_payload,                     # <-- Clean text snippet saved
                            "parent_id": record.get("parent_id"),
                            "source": source_payload,                 # <-- Clean filename saved
                            "chunk_index": record.get("chunk_index"),
                        },
                    )
                )

            self.qdrant_client.upsert(
                collection_name=self.collection_name, points=points
            )
            logger.info(
                "Successfully indexed hybrid local vectors into Qdrant with payload metadata",
                count=len(points),
            )

        except Exception as e:
            logger.error(
                "Local hybrid embedding generation or Qdrant upsert failed",
                error=str(e),
            )
            raise e

    def hybrid_search(self, query_text: str, limit: int = 5):
        logger.info(
            "Executing local hybrid search vector query", query=query_text
        )

        try:
            # Single forward pass to derive query properties from BGE-M3
            outputs = self.embedding_model.encode(
                [query_text], return_dense=True, return_sparse=True
            )

            dense_vector = outputs["dense_vecs"][0].tolist()
            sparse_dict = outputs["lexical_weights"][0]

            # Format sparse weights into Qdrant structure
            indices = [int(k) for k in sparse_dict.keys()]
            values = [float(v) for v in sparse_dict.values()]
            sparse_vector = models.SparseVector(indices=indices, values=values)

            # Query Qdrant with two sub-queries merged using Reciprocal Rank Fusion (RRF)
            results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    # Prefetch 1: Semantic match against the unnamed/primary dense vector
                    models.Prefetch(
                        query=dense_vector, 
                        using="", 
                        limit=limit * 2
                    ),
                    # Prefetch 2: Keyword match against the named sparse vector index
                    models.Prefetch(
                        query=sparse_vector, 
                        using="sparse-text", 
                        limit=limit * 2
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
            )

            return [
                {
                    "score": hit.score,
                    "text": hit.payload.get("text"),
                    "source": hit.payload.get("source"),
                    "parent_id": hit.payload.get("parent_id"),
                }
                for hit in results.points
            ]
        except Exception as e:
            logger.error("Hybrid vector search operation failed", error=str(e))
            raise e