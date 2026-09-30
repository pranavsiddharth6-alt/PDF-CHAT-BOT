import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from app.services.pdf_service import extract_pdf_text, PDFProcessingError
from app.services.chunking_service import chunk_text, ChunkingError
from app.services.embedding_service import embedding_service, EmbeddingError
from app.services.vector_store_service import vector_store_service, VectorStoreError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Phase 4 Endpoint: Upload a PDF document, extract page-by-page text,
    split into metadata-preserving chunks, generate Hugging Face embeddings,
    and store all vectors in Chroma Cloud.
    """
    if not file or not file.filename:
        logger.warning("Upload failed: No file provided in request.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided."
        )

    filename = file.filename
    if not filename.lower().endswith(".pdf"):
        logger.warning(f"Upload rejected: '{filename}' does not have a .pdf extension.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type for '{filename}'. Only PDF files (.pdf) are allowed."
        )

    try:
        contents = await file.read()
    except Exception as e:
        logger.error(f"Error reading uploaded file bytes for '{filename}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error reading file '{filename}': {str(e)}"
        )

    if not contents:
        logger.warning(f"Upload rejected: '{filename}' is empty (0 bytes).")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File '{filename}' is empty (0 bytes)."
        )

    # Magic number check for PDF format (%PDF-)
    if not contents.startswith(b"%PDF"):
        logger.warning(f"Upload rejected: '{filename}' header does not match PDF signature.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File '{filename}' is not a valid PDF file structure."
        )

    logger.info(f"Processing uploaded PDF '{filename}' ({len(contents)} bytes)...")

    # 1. PDF Text Extraction
    try:
        extraction_result = extract_pdf_text(contents)
    except PDFProcessingError as pe:
        logger.warning(f"PDF extraction error for '{filename}': {str(pe)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(pe)
        )
    except Exception as e:
        logger.error(f"Unexpected error during PDF extraction for '{filename}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while extracting PDF '{filename}': {str(e)}"
        )

    # 2. Text Chunking
    try:
        chunks = chunk_text(
            pages=extraction_result["pages"],
            source_filename=filename,
            chunk_size=500,
            chunk_overlap=50
        )
    except ChunkingError as ce:
        logger.warning(f"Chunking error for '{filename}': {str(ce)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Chunking error for '{filename}': {str(ce)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during chunking for '{filename}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while chunking text for '{filename}': {str(e)}"
        )

    if not chunks:
        logger.info(f"No text chunks generated for '{filename}' (document may be empty or non-text).")
        return {
            "filename": filename,
            "total_pages": extraction_result["total_pages"],
            "total_chunks": 0,
            "vectors_stored": 0,
            "embedding_model": embedding_service._model_name,
            "embedding_dimension": 0,
            "vector_store_status": "skipped — no chunks generated"
        }

    # 3. Batch Embedding Generation using Hugging Face sentence-transformers
    chunk_texts = [c["text"] for c in chunks]
    try:
        embeddings, dimension = embedding_service.generate_embeddings(chunk_texts)
    except EmbeddingError as ee:
        logger.error(f"Embedding error for '{filename}': {str(ee)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding error for '{filename}': {str(ee)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error generating embeddings for '{filename}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while generating embeddings for '{filename}': {str(e)}"
        )

    # 4. Store vectors in Chroma Cloud
    try:
        vectors_stored = vector_store_service.store_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )
        vector_store_status = "ok"
    except VectorStoreError as ve:
        logger.error(f"Vector store error for '{filename}': {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector store error for '{filename}': {str(ve)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error storing vectors for '{filename}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while storing vectors for '{filename}': {str(e)}"
        )

    # 5. Prepare lightweight response
    processed_chunks = []
    for i, c in enumerate(chunks):
        processed_chunks.append({
            "chunk_id": c["chunk_id"],
            "page_number": c["page_number"],
            "source": c["source"],
            "text": c["text"],
            "embedding_generated": True if (i < len(embeddings) and len(embeddings[i]) == dimension) else False
        })

    logger.info(
        f"PDF processing complete for '{filename}': {extraction_result['total_pages']} page(s), "
        f"{len(processed_chunks)} chunk(s), {vectors_stored} vector(s) stored in vector store."
    )

    return {
        "filename": filename,
        "total_pages": extraction_result["total_pages"],
        "total_chunks": len(processed_chunks),
        "vectors_stored": vectors_stored,
        "embedding_model": embedding_service._model_name,
        "embedding_dimension": dimension,
        "vector_store_status": vector_store_status,
        "chunks": processed_chunks
    }
