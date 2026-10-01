import io
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from main import app
from app.services.vector_store_service import VectorStoreError
from app.services.llm_service import LLMError
from app.services.embedding_service import EmbeddingError
from app.services.conversation_service import conversation_service

client = TestClient(app)


def test_root_health_check():
    """Test root health check GET / endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "PDF Chatbot API is running."}


def test_upload_pdf_invalid_extension():
    """Test upload route with invalid file extension."""
    files = {"file": ("test.txt", b"Plain text content", "text/plain")}
    response = client.post("/api/documents/upload", files=files)
    assert response.status_code == 400
    assert "Only PDF files (.pdf) are allowed" in response.json()["detail"]


def test_upload_pdf_invalid_magic_bytes():
    """Test upload route with non-PDF binary data."""
    files = {"file": ("fake.pdf", b"NOT_A_PDF_HEADER", "application/pdf")}
    response = client.post("/api/documents/upload", files=files)
    assert response.status_code == 400
    assert "not a valid PDF file structure" in response.json()["detail"]


@patch("app.routes.document_routes.extract_pdf_text")
@patch("app.routes.document_routes.chunk_text")
@patch("app.routes.document_routes.embedding_service.generate_embeddings")
@patch("app.routes.document_routes.vector_store_service.store_chunks")
def test_upload_pdf_success(mock_store, mock_embed, mock_chunk, mock_extract):
    """Test successful PDF upload processing pipeline."""
    mock_extract.return_value = {
        "total_pages": 1,
        "pages": [{"page_number": 1, "text": "Sample text."}]
    }
    mock_chunk.return_value = [
        {"chunk_id": 1, "page_number": 1, "source": "sample.pdf", "text": "Sample text."}
    ]
    mock_embed.return_value = ([[0.1] * 384], 384)
    mock_store.return_value = 1

    pdf_content = b"%PDF-1.4 sample pdf binary data"
    files = {"file": ("sample.pdf", pdf_content, "application/pdf")}

    response = client.post("/api/documents/upload", files=files)
    assert response.status_code == 200
    data = response.json()

    assert data["filename"] == "sample.pdf"
    assert data["total_pages"] == 1
    assert data["total_chunks"] == 1
    assert data["vectors_stored"] == 1
    assert data["vector_store_status"] == "ok"


def test_semantic_search_empty_query():
    """Test semantic search route with empty query."""
    response = client.post("/api/search", json={"query": "   ", "top_k": 3})
    assert response.status_code == 400
    assert "Query must not be empty" in response.json()["detail"]


@patch("app.routes.search_routes.embedding_service.generate_embeddings")
@patch("app.routes.search_routes.vector_store_service.search")
def test_semantic_search_success(mock_search, mock_embed):
    """Test successful semantic search route."""
    mock_embed.return_value = ([[0.1] * 384], 384)
    mock_search.return_value = [
        {
            "chunk_id": "doc__page_1__chunk_1",
            "text": "Supervised learning info.",
            "source": "ml.pdf",
            "page_number": 1,
            "distance": 0.05,
        }
    ]

    response = client.post("/api/search", json={"query": "What is supervised learning?", "top_k": 1})
    assert response.status_code == 200
    data = response.json()

    assert data["query"] == "What is supervised learning?"
    assert data["total_results"] == 1
    assert data["results"][0]["source"] == "ml.pdf"


def test_chat_with_pdf_empty_question():
    """Test RAG chat route with empty question string."""
    response = client.post("/api/chat", json={"question": "   ", "document_filename": "doc.pdf", "top_k": 3})
    assert response.status_code == 400
    assert "Question must not be empty" in response.json()["detail"]


def test_chat_with_pdf_missing_document_filename():
    """Test RAG chat route rejects requests without active document_filename."""
    response = client.post("/api/chat", json={"question": "What is in the PDF?", "top_k": 3})
    assert response.status_code == 400
    assert "Active document_filename is required" in response.json()["detail"]


@patch("app.routes.chat_routes.embedding_service.generate_embeddings")
@patch("app.routes.chat_routes.vector_store_service.search")
@patch("app.routes.chat_routes.llm_service.generate_answer")
def test_chat_with_pdf_success(mock_answer, mock_search, mock_embed):
    """Test successful RAG chat Q&A endpoint with document_filename filter."""
    mock_embed.return_value = ([[0.1] * 384], 384)
    mock_search.return_value = [
        {
            "chunk_id": "doc__page_1__chunk_1",
            "text": "Machine learning details.",
            "source": "doc.pdf",
            "page_number": 1,
            "distance": 0.1,
        }
    ]
    mock_answer.return_value = "Machine learning is a field of AI."

    response = client.post("/api/chat", json={"question": "What is machine learning?", "document_filename": "doc.pdf", "top_k": 3})
    assert response.status_code == 200
    data = response.json()

    assert data["answer"] == "Machine learning is a field of AI."
    assert len(data["sources"]) == 1
    assert data["sources"][0] == {"source": "doc.pdf", "page_number": 1}
    assert "conversation_id" in data

    # Verify vector_store_service.search was called with metadata filter for doc.pdf
    mock_search.assert_called_once()
    _, kwargs = mock_search.call_args
    assert kwargs.get("where") == {"source": "doc.pdf"}


@patch("app.routes.chat_routes.embedding_service.generate_embeddings")
@patch("app.routes.chat_routes.vector_store_service.search")
@patch("app.routes.chat_routes.llm_service.generate_answer")
def test_chat_with_pdf_multi_document_isolation(mock_answer, mock_search, mock_embed):
    """Test that querying docA.pdf isolates Chroma search strictly to docA.pdf and excludes docB.pdf."""
    mock_embed.return_value = ([[0.2] * 384], 384)

    # When searching for docA.pdf, return only docA chunks
    mock_search.return_value = [
        {"chunk_id": "docA__p1_c0", "source": "docA.pdf", "page_number": 1, "text": "Content from Document A"}
    ]
    mock_answer.return_value = "Response based on Document A."

    response_a = client.post("/api/chat", json={"question": "Summarize doc", "document_filename": "docA.pdf"})
    assert response_a.status_code == 200
    data_a = response_a.json()
    assert data_a["sources"][0]["source"] == "docA.pdf"
    assert mock_search.call_args[1]["where"] == {"source": "docA.pdf"}

    mock_search.reset_mock()
    # When searching for docB.pdf, return only docB chunks
    mock_search.return_value = [
        {"chunk_id": "docB__p2_c1", "source": "docB.pdf", "page_number": 2, "text": "Content from Document B"}
    ]
    mock_answer.return_value = "Response based on Document B."

    response_b = client.post("/api/chat", json={"question": "Summarize doc", "document_filename": "docB.pdf"})
    assert response_b.status_code == 200
    data_b = response_b.json()
    assert data_b["sources"][0]["source"] == "docB.pdf"
    assert mock_search.call_args[1]["where"] == {"source": "docB.pdf"}


@patch("app.routes.chat_routes.embedding_service.generate_embeddings")
@patch("app.routes.chat_routes.vector_store_service.search")
@patch("app.routes.chat_routes.llm_service.generate_answer")
def test_chat_with_pdf_empty_retrieval(mock_answer, mock_search, mock_embed):
    """Test chat flow when vector store returns 0 matching chunks for active document."""
    mock_embed.return_value = ([[0.1] * 384], 384)
    mock_search.return_value = []
    mock_answer.return_value = "The requested information was not found in the uploaded document."

    response = client.post("/api/chat", json={"question": "Where is the secret key?", "document_filename": "doc.pdf", "top_k": 3})
    assert response.status_code == 200
    data = response.json()

    assert data["answer"] == "The requested information was not found in the uploaded document."
    assert data["sources"] == []


@patch("app.routes.chat_routes.embedding_service.generate_embeddings")
@patch("app.routes.chat_routes.vector_store_service.search")
@patch("app.routes.chat_routes.llm_service.generate_answer")
def test_chat_with_pdf_source_deduplication(mock_answer, mock_search, mock_embed):
    """Test that multiple chunks from the same page produce deduplicated source items."""
    mock_embed.return_value = ([[0.1] * 384], 384)
    mock_search.return_value = [
        {"chunk_id": "doc__p1_c1", "source": "guide.pdf", "page_number": 1, "text": "Chunk 1"},
        {"chunk_id": "doc__p1_c2", "source": "guide.pdf", "page_number": 1, "text": "Chunk 2"},
        {"chunk_id": "doc__p2_c3", "source": "guide.pdf", "page_number": 2, "text": "Chunk 3"},
    ]
    mock_answer.return_value = "Deduplicated response."

    response = client.post("/api/chat", json={"question": "Overview", "document_filename": "guide.pdf", "top_k": 3})
    assert response.status_code == 200
    data = response.json()

    assert len(data["sources"]) == 2
    assert data["sources"][0] == {"source": "guide.pdf", "page_number": 1}
    assert data["sources"][1] == {"source": "guide.pdf", "page_number": 2}


@patch("app.routes.chat_routes.embedding_service.generate_embeddings")
@patch("app.routes.chat_routes.vector_store_service.search", side_effect=VectorStoreError("Chroma offline"))
def test_chat_with_pdf_vector_store_failure(mock_search, mock_embed):
    """Test 503 error handling when vector store search fails."""
    mock_embed.return_value = ([[0.1] * 384], 384)
    response = client.post("/api/chat", json={"question": "What is AI?", "document_filename": "doc.pdf"})
    assert response.status_code == 503
    assert "Vector store retrieval failed" in response.json()["detail"]


@patch("app.routes.chat_routes.embedding_service.generate_embeddings")
@patch("app.routes.chat_routes.vector_store_service.search", return_value=[])
@patch("app.routes.chat_routes.llm_service.generate_answer", side_effect=LLMError("API token invalid"))
def test_chat_with_pdf_llm_failure(mock_answer, mock_search, mock_embed):
    """Test 500 error handling when LLM answer generation fails."""
    mock_embed.return_value = ([[0.1] * 384], 384)
    response = client.post("/api/chat", json={"question": "What is AI?", "document_filename": "doc.pdf"})
    assert response.status_code == 500
    assert "LLM generation failed" in response.json()["detail"]


def test_clear_conversation_endpoint():
    """Test DELETE /api/chat/{conversation_id} endpoint."""
    cid = "test_del_session"
    conversation_service.add_message(cid, "user", "Message to delete")

    response = client.delete(f"/api/chat/{cid}")
    assert response.status_code == 200
    assert response.json()["cleared"] is True
    assert conversation_service.get_history(cid) == []
