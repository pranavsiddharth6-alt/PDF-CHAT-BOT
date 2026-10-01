import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from app.services.embedding_service import embedding_service, EmbeddingError
from app.services.vector_store_service import vector_store_service, VectorStoreError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The search query string.")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of top results to return (1–20).")
    document_filename: Optional[str] = Field(
        default=None,
        description="Optional active PDF filename to restrict search scope."
    )


@router.post("")
def semantic_search(request: SearchRequest):
    """
    Phase 4 Endpoint: Semantic search over stored document chunks.

    1. Validates the incoming query and top_k parameter.
    2. Generates a 384-dimensional query embedding via embedding_service.
    3. Queries Chroma Cloud for top-K matching chunks (optionally filtered by document_filename).
    4. Returns matching chunk text, page number, source filename, and cosine distance.
    """
    query = request.query.strip()
    if not query:
        logger.warning("Semantic search request received empty/whitespace query.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must not be empty or whitespace-only."
        )

    doc_filename = request.document_filename.strip() if request.document_filename else None
    where_filter = {"source": doc_filename} if doc_filename else None

    logger.info(f"Received semantic search query (top_k={request.top_k}, doc='{doc_filename}').")

    # 1. Generate query embedding
    try:
        embeddings, dimension = embedding_service.generate_embeddings([query])
        query_embedding = embeddings[0]
    except EmbeddingError as ee:
        logger.error(f"Semantic search query embedding error: {str(ee)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate query embedding: {str(ee)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during query embedding: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during query embedding: {str(e)}"
        )

    # 2. Perform similarity search in Chroma Cloud
    try:
        results = vector_store_service.search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            where=where_filter,
        )
    except VectorStoreError as ve:
        logger.error(f"Semantic search vector retrieval error: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector store search failed: {str(ve)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during similarity search: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during similarity search: {str(e)}"
        )

    logger.info(f"Semantic search returned {len(results)} matching chunk(s).")

    return {
        "query": query,
        "top_k": request.top_k,
        "embedding_model": embedding_service._model_name,
        "embedding_dimension": dimension,
        "total_results": len(results),
        "results": results,
    }
