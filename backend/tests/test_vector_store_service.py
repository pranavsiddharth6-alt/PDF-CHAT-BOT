import os
import pytest
from unittest.mock import MagicMock, patch
from app.services.vector_store_service import (
    VectorStoreService,
    VectorStoreError,
    COLLECTION_NAME,
    vector_store_service,
    _sanitize_filename,
    _make_chunk_id,
)


def test_vector_store_service_singleton_and_constants():
    """Test that VectorStoreService acts as a singleton and uses expected collection name."""
    service1 = VectorStoreService()
    service2 = VectorStoreService()
    assert service1 is service2
    assert COLLECTION_NAME == "pdf_chatbot_chunks"
    assert isinstance(vector_store_service, VectorStoreService)


def test_sanitize_filename():
    """Test filename sanitization for Chroma IDs."""
    assert _sanitize_filename("sample_doc.pdf") == "sample_doc"
    assert _sanitize_filename("My File (2024)!.pdf") == "My_File__2024__"
    assert _sanitize_filename("deep/nested/path/document.pdf") == "document" or "deep" in _sanitize_filename("deep/nested/path/document.pdf")


def test_make_chunk_id():
    """Test deterministic chunk ID generation."""
    chunk_id = _make_chunk_id("report.pdf", 3, 5)
    assert chunk_id == "report__page_3__chunk_5"


def test_missing_credentials_raises_error(monkeypatch):
    """Test that missing Chroma environment variables raise VectorStoreError."""
    monkeypatch.delenv("CHROMA_API_KEY", raising=False)
    monkeypatch.delenv("CHROMA_TENANT", raising=False)
    monkeypatch.delenv("CHROMA_DATABASE", raising=False)

    service = VectorStoreService()
    with pytest.raises(VectorStoreError, match="Missing required Chroma Cloud environment variable"):
        service._load_credentials()


def test_missing_single_credential_raises_error(monkeypatch):
    """Test that missing one specific Chroma credential identifies the missing variable."""
    monkeypatch.setenv("CHROMA_API_KEY", "test_key")
    monkeypatch.setenv("CHROMA_TENANT", "test_tenant")
    monkeypatch.delenv("CHROMA_DATABASE", raising=False)

    service = VectorStoreService()
    with pytest.raises(VectorStoreError, match="CHROMA_DATABASE"):
        service._load_credentials()


def test_connect_client_and_collection_creation():
    """Test client and collection initialization with mock chromadb."""
    service = VectorStoreService()
    service._client = None
    service._collection = None

    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.count.return_value = 10
    mock_client.get_or_create_collection.return_value = mock_collection

    with patch("chromadb.CloudClient", return_value=mock_client):
        with patch.object(service, "_load_credentials", return_value=("fake_key", "fake_tenant", "fake_db")):
            service._connect()

            assert service._client is mock_client
            assert service._collection is mock_collection
            mock_client.get_or_create_collection.assert_called_once_with(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )


def test_connect_client_failure():
    """Test connection error handling when chromadb.CloudClient fails."""
    service = VectorStoreService()
    service._client = None
    service._collection = None

    with patch("chromadb.CloudClient", side_effect=Exception("Network connection failed")):
        with patch.object(service, "_load_credentials", return_value=("fake_key", "fake_tenant", "fake_db")):
            with pytest.raises(VectorStoreError, match="Failed to connect to Chroma Cloud"):
                service._connect()


def test_connect_collection_failure():
    """Test error handling when get_or_create_collection fails."""
    service = VectorStoreService()
    service._client = None
    service._collection = None

    mock_client = MagicMock()
    mock_client.get_or_create_collection.side_effect = Exception("Collection quota exceeded")

    with patch("chromadb.CloudClient", return_value=mock_client):
        with patch.object(service, "_load_credentials", return_value=("fake_key", "fake_tenant", "fake_db")):
            with pytest.raises(VectorStoreError, match="Failed to get/create Chroma collection"):
                service._connect()


def test_store_chunks_validation():
    """Test parameter validation in store_chunks."""
    service = VectorStoreService()

    with pytest.raises(VectorStoreError, match="No chunks provided"):
        service.store_chunks([], [])

    chunks = [{"text": "t", "page_number": 1, "source": "s.pdf", "chunk_id": 1}]
    embeddings = [[0.1] * 384, [0.2] * 384]

    with pytest.raises(VectorStoreError, match="Chunk count .* does not match embedding count"):
        service.store_chunks(chunks, embeddings)


def test_store_chunks_upsert_failure():
    """Test error handling when collection.upsert raises an exception."""
    service = VectorStoreService()
    mock_collection = MagicMock()
    mock_collection.upsert.side_effect = Exception("Upsert rate limit exceeded")

    service._client = MagicMock()
    service._collection = mock_collection

    chunks = [{"text": "test content", "page_number": 1, "source": "sample.pdf", "chunk_id": 1}]
    embeddings = [[0.05] * 384]

    with pytest.raises(VectorStoreError, match="Failed to store vectors in Chroma Cloud"):
        service.store_chunks(chunks, embeddings)


def test_search_validation():
    """Test parameter validation in search."""
    service = VectorStoreService()

    with pytest.raises(VectorStoreError, match="Empty query embedding provided"):
        service.search([], top_k=3)

    with pytest.raises(VectorStoreError, match="top_k must be at least 1"):
        service.search([0.1] * 384, top_k=0)


def test_search_failure():
    """Test error handling when collection.query raises an exception."""
    service = VectorStoreService()
    mock_collection = MagicMock()
    mock_collection.query.side_effect = Exception("Query timeout")

    service._client = MagicMock()
    service._collection = mock_collection

    with pytest.raises(VectorStoreError, match="Chroma similarity search failed"):
        service.search([0.1] * 384, top_k=3)


def test_search_empty_results():
    """Test search returning empty result list when no matches are found."""
    service = VectorStoreService()
    mock_collection = MagicMock()
    mock_collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    service._client = MagicMock()
    service._collection = mock_collection

    results = service.search([0.1] * 384, top_k=3)
    assert results == []


def test_store_chunks_and_search_with_mock_and_filter():
    """Test full store_chunks and search with 384-dimensional embeddings, metadata, and where filter."""
    service = VectorStoreService()
    mock_collection = MagicMock()
    mock_collection.upsert.return_value = None
    mock_collection.query.return_value = {
        "ids": [["doc__page_1__chunk_1", "doc__page_2__chunk_2"]],
        "documents": [["Chunk 1 text content.", "Chunk 2 text content."]],
        "metadatas": [[
            {"source": "doc.pdf", "page_number": 1, "chunk_id": 1},
            {"source": "doc.pdf", "page_number": 2, "chunk_id": 2},
        ]],
        "distances": [[0.051234, 0.129876]],
    }

    service._client = MagicMock()
    service._collection = mock_collection

    chunks = [
        {"text": "Chunk 1 text content.", "page_number": 1, "source": "doc.pdf", "chunk_id": 1},
        {"text": "Chunk 2 text content.", "page_number": 2, "source": "doc.pdf", "chunk_id": 2},
    ]
    # 384 dimensions matching sentence-transformers/all-MiniLM-L6-v2
    embeddings = [[0.01] * 384, [0.02] * 384]

    stored_count = service.store_chunks(chunks, embeddings)
    assert stored_count == 2
    mock_collection.upsert.assert_called_once_with(
        ids=["doc__page_1__chunk_1", "doc__page_2__chunk_2"],
        documents=["Chunk 1 text content.", "Chunk 2 text content."],
        embeddings=embeddings,
        metadatas=[
            {"source": "doc.pdf", "page_number": 1, "chunk_id": 1},
            {"source": "doc.pdf", "page_number": 2, "chunk_id": 2},
        ],
    )

    results = service.search([0.01] * 384, top_k=2, where={"source": "doc.pdf"})
    assert len(results) == 2
    assert results[0]["chunk_id"] == "doc__page_1__chunk_1"
    assert results[0]["text"] == "Chunk 1 text content."
    assert results[0]["source"] == "doc.pdf"
    assert results[0]["page_number"] == 1
    assert results[0]["distance"] == 0.051234

    assert results[1]["chunk_id"] == "doc__page_2__chunk_2"
    assert results[1]["page_number"] == 2
    assert results[1]["distance"] == 0.129876


def test_collection_count():
    """Test getting vector collection count."""
    service = VectorStoreService()
    mock_collection = MagicMock()
    mock_collection.count.return_value = 42

    service._client = MagicMock()
    service._collection = mock_collection

    count = service.collection_count()
    assert count == 42


def test_collection_count_failure():
    """Test error handling when collection.count fails."""
    service = VectorStoreService()
    mock_collection = MagicMock()
    mock_collection.count.side_effect = Exception("Collection read error")

    service._client = MagicMock()
    service._collection = mock_collection

    with pytest.raises(VectorStoreError, match="Failed to get collection count"):
        service.collection_count()
