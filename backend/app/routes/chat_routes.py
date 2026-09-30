import logging
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from app.services.embedding_service import embedding_service, EmbeddingError
from app.services.vector_store_service import vector_store_service, VectorStoreError
from app.services.llm_service import llm_service, LLMError
from app.services.conversation_service import conversation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = Field(
        default=None,
        description="Unique conversation/session ID. If omitted or null, a new conversation is started."
    )
    question: str = Field(
        ...,
        min_length=1,
        description="User question to answer using uploaded PDF context."
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of context chunks to retrieve."
    )


class SourceInfo(BaseModel):
    source: str
    page_number: int


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: List[SourceInfo]


@router.post("", response_model=ChatResponse)
def chat_with_pdf(request: ChatRequest):
    """
    Phase 7 Endpoint: Multi-Turn RAG Question-Answering over uploaded PDF documents
    with conversation memory.

    1. Validates user question.
    2. Resolves or generates unique conversation_id.
    3. Retrieves previous dialogue history for the conversation.
    4. Generates query vector embedding using existing EmbeddingService.
    5. Retrieves top-K context chunks from existing Chroma Cloud collection.
    6. Assembles grounded prompt combining conversation history + retrieved context + question.
    7. Generates natural language answer via Hugging Face LLM (Qwen/Qwen2.5-7B-Instruct).
    8. Records user turn and assistant answer into conversation memory.
    9. Returns answer, conversation_id, and source citations.
    """
    question = request.question.strip()
    if not question:
        logger.warning("Chat request received empty/whitespace question.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question must not be empty or whitespace-only."
        )

    # 1. Resolve or create conversation session
    conversation_id = request.conversation_id.strip() if request.conversation_id else ""
    if not conversation_id:
        conversation_id = conversation_service.create_conversation_id()
        logger.info(f"Initialized new conversation session '{conversation_id}'.")

    # 2. Retrieve past conversation history for this session
    past_history = conversation_service.get_history(conversation_id)
    logger.info(
        f"Processing chat question for conversation '{conversation_id}' "
        f"(history_messages={len(past_history)}, top_k={request.top_k})."
    )

    # 3. Generate query embedding using existing EmbeddingService
    try:
        embeddings, _ = embedding_service.generate_embeddings([question])
        query_embedding = embeddings[0]
    except EmbeddingError as ee:
        logger.error(f"Chat query embedding error: {str(ee)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(ee)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during query embedding: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during query embedding: {str(e)}"
        )

    # 4. Retrieve top-K relevant chunks from existing Chroma Cloud collection
    try:
        chunks = vector_store_service.search(
            query_embedding=query_embedding,
            top_k=request.top_k,
        )
    except VectorStoreError as ve:
        logger.error(f"Chat vector retrieval error: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector store retrieval failed: {str(ve)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during similarity search: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during similarity search: {str(e)}"
        )

    logger.info(f"Retrieved {len(chunks)} context chunk(s) for chat question.")

    # 5. Extract unique source file & page number pairs
    sources: List[Dict[str, Any]] = []
    seen_sources = set()

    for chunk in chunks:
        src = chunk.get("source", "unknown")
        page = chunk.get("page_number", -1)
        key = (src, page)
        if key not in seen_sources:
            seen_sources.add(key)
            sources.append({
                "source": src,
                "page_number": page,
            })

    # 6. Generate answer via Hugging Face LLM with conversation history
    try:
        logger.info(
            f"Invoking LLM to generate answer with {len(chunks)} chunk(s) "
            f"and {len(past_history)} history turn(s)..."
        )
        answer = llm_service.generate_answer(
            question=question,
            chunks=chunks,
            chat_history=past_history,
        )
    except LLMError as le:
        logger.error(f"LLM answer generation failed: {str(le)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM generation failed: {str(le)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during LLM answer generation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during LLM answer generation: {str(e)}"
        )

    # 7. Update conversation memory with current turn
    conversation_service.add_message(conversation_id=conversation_id, role="user", content=question)
    conversation_service.add_message(conversation_id=conversation_id, role="assistant", content=answer)

    logger.info(
        f"Generated answer for conversation '{conversation_id}' with {len(sources)} source citation(s)."
    )

    return {
        "conversation_id": conversation_id,
        "answer": answer,
        "sources": sources,
    }


@router.delete("/{conversation_id}")
def clear_conversation(conversation_id: str):
    """Resets/clears the stored history for a specific conversation session."""
    cleared = conversation_service.clear_history(conversation_id)
    return {
        "conversation_id": conversation_id,
        "cleared": cleared,
        "message": f"Conversation '{conversation_id}' history cleared." if cleared else f"No active history for '{conversation_id}'."
    }
