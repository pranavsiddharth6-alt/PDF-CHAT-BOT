import os
import pytest
from unittest.mock import MagicMock, patch
from app.services.embedding_service import (
    EmbeddingService,
    EmbeddingError,
    DEFAULT_MODEL_NAME,
    REQUIRED_DIMENSION,
    embedding_service,
)


def test_embedding_service_singleton_and_model_name():
    """Test that EmbeddingService acts as a singleton and uses the default model name."""
    service1 = EmbeddingService()
    service2 = EmbeddingService()
    assert service1 is service2
    assert service1._model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert DEFAULT_MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2"
    assert REQUIRED_DIMENSION == 384
    assert isinstance(embedding_service, EmbeddingService)


def test_embedding_service_empty_input():
    """Test that providing an empty text list raises EmbeddingError."""
    service = EmbeddingService()
    with pytest.raises(EmbeddingError, match="No text inputs provided"):
        service.generate_embeddings([])


def test_embedding_service_whitespace_input():
    """Test that whitespace-only strings raise EmbeddingError."""
    service = EmbeddingService()
    with pytest.raises(EmbeddingError, match="All provided text chunks were empty or whitespace-only"):
        service.generate_embeddings(["   ", "\n\t", "  "])


def test_embedding_service_dimension():
    """Test retrieving model embedding dimension (must be 384)."""
    service = EmbeddingService()
    dim = service.get_embedding_dimension()
    assert dim == 384


def test_1_successful_embedding_mocked():
    """Test 1 — Successful embedding: Service calls HF InferenceClient and returns valid 384-d vectors."""
    service = EmbeddingService()
    mock_client = MagicMock()
    # Create fake 384-dimensional vector list for 2 texts
    fake_vectors = [[0.1] * 384, [0.2] * 384]
    mock_client.feature_extraction.return_value = fake_vectors

    original_client = service._client
    service._client = mock_client
    try:
        texts = ["Retrieval-Augmented Generation", "Artificial Intelligence"]
        embeddings, dim = service.generate_embeddings(texts)

        assert dim == 384
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384
        assert len(embeddings[1]) == 384
        assert isinstance(embeddings[0][0], float)
        mock_client.feature_extraction.assert_called_once_with(
            texts,
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
    finally:
        service._client = original_client


def test_2_dimension_validation():
    """Test 2 — Dimension validation: Verify that a valid embedding contains exactly 384 values."""
    service = EmbeddingService()
    mock_client = MagicMock()
    fake_vector = [[0.05] * 384]
    mock_client.feature_extraction.return_value = fake_vector

    original_client = service._client
    service._client = mock_client
    try:
        embeddings, dim = service.generate_embeddings(["Test sentence for dimension check"])
        assert dim == 384
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384
    finally:
        service._client = original_client


def test_3_invalid_dimension_rejection():
    """Test 3 — Invalid dimension: Verify that an embedding with wrong dimension (e.g. 512) is rejected."""
    service = EmbeddingService()
    mock_client = MagicMock()
    # Return 512 dimensions instead of expected 384
    invalid_vector = [[0.1] * 512]
    mock_client.feature_extraction.return_value = invalid_vector

    original_client = service._client
    service._client = mock_client
    try:
        with pytest.raises(EmbeddingError, match="Invalid embedding dimension: expected 384, got 512"):
            service.generate_embeddings(["Sample text with bad dimension"])
    finally:
        service._client = original_client


def test_4_missing_api_token():
    """Test 4 — Missing API token: Verify missing/unconfigured token raises clear EmbeddingError."""
    service = EmbeddingService()
    original_client = service._client
    service._client = None
    try:
        with patch.dict(os.environ, {"HUGGINGFACEHUB_API_TOKEN": "", "HF_TOKEN": ""}, clear=True):
            with pytest.raises(EmbeddingError, match="Hugging Face API token is missing or unconfigured"):
                service.generate_embeddings(["Sample text"])
    finally:
        service._client = original_client


def test_5_huggingface_api_failure():
    """Test 5 — Hugging Face API failure: Verify API failure is handled without exposing secrets."""
    service = EmbeddingService()
    mock_client = MagicMock()
    mock_client.feature_extraction.side_effect = RuntimeError("Hugging Face API 503 Service Unavailable")

    original_client = service._client
    service._client = mock_client
    try:
        with pytest.raises(EmbeddingError, match="Hugging Face Inference API embedding error"):
            service.generate_embeddings(["Sample text during outage"])
    finally:
        service._client = original_client


def test_embedding_filters_empty_strings_among_valid():
    """Test that empty or whitespace strings among valid texts are cleanly filtered out."""
    service = EmbeddingService()
    mock_client = MagicMock()
    mock_client.feature_extraction.return_value = [[0.1] * 384, [0.2] * 384]

    original_client = service._client
    service._client = mock_client
    try:
        texts = ["Valid text one", "", "   ", "Valid text two"]
        embeddings, dim = service.generate_embeddings(texts)

        assert dim == 384
        assert len(embeddings) == 2
        mock_client.feature_extraction.assert_called_once_with(
            ["Valid text one", "Valid text two"],
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
    finally:
        service._client = original_client

