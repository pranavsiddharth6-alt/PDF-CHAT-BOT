import pytest
from unittest.mock import MagicMock, patch
from app.services.llm_service import (
    LLMService,
    LLMError,
    DEFAULT_MODEL,
    llm_service,
)


def test_llm_service_initialization():
    """Test LLM service default model name and instance creation."""
    service = LLMService()
    assert service.model_name == "Qwen/Qwen2.5-7B-Instruct"
    assert DEFAULT_MODEL == "Qwen/Qwen2.5-7B-Instruct"
    assert isinstance(llm_service, LLMService)


def test_llm_service_token_resolution(monkeypatch):
    """Test token resolution from both HUGGINGFACEHUB_API_TOKEN and HF_TOKEN."""
    service = LLMService()

    # Case 1: HUGGINGFACEHUB_API_TOKEN set
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "hf_test_token_123")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert service._get_token() == "hf_test_token_123"

    # Case 2: HF_TOKEN fallback
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_456")
    assert service._get_token() == "hf_test_token_456"


def test_llm_service_missing_or_placeholder_token(monkeypatch):
    """Test that missing or unconfigured placeholder tokens raise LLMError."""
    service = LLMService()

    # Missing
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(LLMError, match="Hugging Face API token is missing or unconfigured"):
        service._get_token()

    # Placeholder string
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "your_huggingface_api_token_here")
    with pytest.raises(LLMError, match="Hugging Face API token is missing or unconfigured"):
        service._get_token()


def test_get_client_initialization_and_failure(monkeypatch):
    """Test client initialization and exception handling."""
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "valid_token")
    service = LLMService()
    service._client = None

    with patch("app.services.llm_service.InferenceClient", side_effect=Exception("Initialization failed")):
        with pytest.raises(LLMError, match="Failed to initialize Hugging Face InferenceClient"):
            service._get_client()


def test_build_prompt_context_empty():
    """Test prompt context creation when no chunks are retrieved."""
    service = LLMService()
    context = service.build_prompt_context([])
    assert context == "No relevant document context found."


def test_build_prompt_context_with_multiple_chunks_and_metadata():
    """Test formatting multiple retrieved chunks into structured context with page numbers and sources."""
    service = LLMService()
    chunks = [
        {"source": "ml_guide.pdf", "page_number": 3, "text": "Supervised learning requires labeled data."},
        {"source": "ai_book.pdf", "page_number": 12, "text": "Neural networks use backpropagation."},
        {"text": "Anonymous chunk with missing metadata."}
    ]
    context = service.build_prompt_context(chunks)

    assert "[Source: ml_guide.pdf | Page: 3]" in context
    assert "Supervised learning requires labeled data." in context
    assert "[Source: ai_book.pdf | Page: 12]" in context
    assert "Neural networks use backpropagation." in context
    assert "[Source: unknown | Page: N/A]" in context
    assert "Anonymous chunk with missing metadata." in context


def test_llm_service_empty_question():
    """Test empty and whitespace-only question validation."""
    service = LLMService()
    with pytest.raises(LLMError, match="Question cannot be empty or whitespace-only"):
        service.generate_answer("", [])

    with pytest.raises(LLMError, match="Question cannot be empty or whitespace-only"):
        service.generate_answer("   \n\t  ", [])


def test_llm_service_generate_answer_prompt_and_params():
    """Test prompt structure, messages format, and hyperparameter passing to InferenceClient."""
    service = LLMService()
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Grounded answer from context."
    mock_response = MagicMock(choices=[mock_choice])
    mock_client.chat_completion.return_value = mock_response

    service._client = mock_client

    chunks = [{"source": "paper.pdf", "page_number": 1, "text": "Quantum computing uses qubits."}]
    question = "What does quantum computing use?"
    answer = service.generate_answer(question, chunks)

    assert answer == "Grounded answer from context."
    mock_client.chat_completion.assert_called_once()
    call_kwargs = mock_client.chat_completion.call_args[1]

    assert call_kwargs["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert call_kwargs["max_tokens"] == 512
    assert call_kwargs["temperature"] == 0.2

    messages = call_kwargs["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "You are a PDF question-answering assistant." in messages[0]["content"]

    assert messages[1]["role"] == "user"
    assert "CONTEXT:" in messages[1]["content"]
    assert "[Source: paper.pdf | Page: 1]" in messages[1]["content"]
    assert "Quantum computing uses qubits." in messages[1]["content"]
    assert "QUESTION:\n\nWhat does quantum computing use?" in messages[1]["content"]


def test_llm_service_generate_answer_empty_choices():
    """Test error handling when API returns response with no choices."""
    service = LLMService()
    mock_client = MagicMock()
    mock_client.chat_completion.return_value = MagicMock(choices=[])
    service._client = mock_client

    with pytest.raises(LLMError, match="No response choices returned by Hugging Face API"):
        service.generate_answer("Any question", [{"source": "doc.pdf", "page_number": 1, "text": "Context"}])


def test_llm_service_generate_answer_empty_content():
    """Test error handling when API choice message content is empty."""
    service = LLMService()
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = ""
    mock_client.chat_completion.return_value = MagicMock(choices=[mock_choice])
    service._client = mock_client

    with pytest.raises(LLMError, match="Received empty response content from Hugging Face API"):
        service.generate_answer("Any question", [{"source": "doc.pdf", "page_number": 1, "text": "Context"}])


def test_llm_service_api_network_exception():
    """Test error handling when InferenceClient raises network or timeout exceptions."""
    service = LLMService()
    mock_client = MagicMock()
    mock_client.chat_completion.side_effect = TimeoutError("Connection to Hugging Face timed out")
    service._client = mock_client

    with pytest.raises(LLMError, match="Hugging Face Inference API error"):
        service.generate_answer("Any question", [])
