import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from app.services.embedding_service import (
    EmbeddingService,
    EmbeddingError,
    DEFAULT_MODEL_NAME,
    embedding_service,
)


def test_embedding_service_singleton_and_model_name():
    """Test that EmbeddingService acts as a singleton and uses the default model name."""
    service1 = EmbeddingService()
    service2 = EmbeddingService()
    assert service1 is service2
    assert service1._model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert DEFAULT_MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2"
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


def test_embedding_service_filters_empty_strings_among_valid():
    """Test that empty or whitespace strings among valid texts are cleanly filtered out."""
    service = EmbeddingService()
    texts = ["Valid text one", "", "   ", "Valid text two"]
    embeddings, dim = service.generate_embeddings(texts)

    assert dim == 384
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384


def test_embedding_service_dimension():
    """Test retrieving model embedding dimension for all-MiniLM-L6-v2."""
    service = EmbeddingService()
    dim = service.get_embedding_dimension()
    assert dim == 384


def test_embedding_service_generate():
    """Test generating embeddings for real sample texts."""
    service = EmbeddingService()
    texts = ["Retrieval-Augmented Generation", "Artificial Intelligence and Machine Learning"]
    embeddings, dim = service.generate_embeddings(texts)

    assert dim == 384
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384
    assert isinstance(embeddings[0][0], float)


def test_embedding_service_batch_multiple_chunks():
    """Test generating embeddings for a larger batch of text chunks (> 32)."""
    service = EmbeddingService()
    texts = [f"This is sample test chunk number {i} describing AI and RAG concepts." for i in range(40)]
    embeddings, dim = service.generate_embeddings(texts)

    assert dim == 384
    assert len(embeddings) == 40
    for emb in embeddings:
        assert len(emb) == 384


def test_embedding_service_model_load_failure():
    """Test error handling when SentenceTransformer model fails to load."""
    with patch("app.services.embedding_service.SentenceTransformer", side_effect=Exception("Model file not found")):
        service = EmbeddingService()
        # Temporarily clear model to test lazy loading failure
        original_model = service._model
        service._model = None
        try:
            with pytest.raises(EmbeddingError, match="Failed to load embedding model"):
                service._load_model()
        finally:
            service._model = original_model


def test_embedding_service_dimension_failure():
    """Test error handling when get_sentence_embedding_dimension fails."""
    service = EmbeddingService()
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.side_effect = Exception("Dimension error")

    original_model = service._model
    service._model = mock_model
    try:
        with pytest.raises(EmbeddingError, match="Failed to get embedding dimension"):
            service.get_embedding_dimension()
    finally:
        service._model = original_model


def test_embedding_service_encode_failure():
    """Test error handling when model.encode raises an unexpected exception."""
    service = EmbeddingService()
    mock_model = MagicMock()
    mock_model.encode.side_effect = RuntimeError("CUDA/CPU out of memory")
    mock_model.get_sentence_embedding_dimension.return_value = 384

    original_model = service._model
    service._model = mock_model
    try:
        with pytest.raises(EmbeddingError, match="Error generating embeddings"):
            service.generate_embeddings(["Some valid text"])
    finally:
        service._model = original_model


def test_embedding_service_mocked_call_parameters():
    """Test that model.encode is called with expected batch_size and normalization."""
    service = EmbeddingService()
    mock_model = MagicMock()
    fake_vectors = np.ones((2, 384), dtype=np.float32)
    mock_model.encode.return_value = fake_vectors
    mock_model.get_sentence_embedding_dimension.return_value = 384

    original_model = service._model
    service._model = mock_model
    try:
        texts = ["Text A", "Text B"]
        embeddings, dim = service.generate_embeddings(texts)

        mock_model.encode.assert_called_once_with(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        assert dim == 384
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384
    finally:
        service._model = original_model
