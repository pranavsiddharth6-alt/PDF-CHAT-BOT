import pytest
import io
import fitz
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app
from app.services.conversation_service import conversation_service

client = TestClient(app)


def create_sample_pdf_bytes(title: str, body: str) -> bytes:
    """Helper to create valid PDF bytes."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), f"{title}\n\n{body}")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestPhase8UIBackendIntegration:

    def setup_method(self):
        conversation_service._conversations.clear()

    @patch("app.routes.document_routes.vector_store_service.store_chunks")
    def test_ui_scenario_1_valid_pdf_upload(self, mock_store):
        """TEST 1: Upload a valid PDF -> Success with document metadata."""
        mock_store.return_value = 2
        pdf_bytes = create_sample_pdf_bytes(
            "Introduction to Deep Learning",
            "Deep learning is a subset of machine learning based on artificial neural networks."
        )

        response = client.post(
            "/api/documents/upload",
            files={"file": ("deep_learning.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "deep_learning.pdf"
        assert data["total_pages"] == 1
        assert data["total_chunks"] >= 1
        assert data["vector_store_status"] == "ok"
        assert data["embedding_dimension"] == 384

    @patch("app.routes.chat_routes.vector_store_service.search")
    @patch("app.routes.chat_routes.llm_service.generate_answer")
    def test_ui_scenario_2_ask_question_answered_by_pdf(self, mock_llm, mock_search):
        """TEST 2: Ask a question answered by the PDF -> Answer + source/page displayed."""
        mock_search.return_value = [
            {
                "chunk_id": "dl_chunk_0",
                "text": "Deep learning is a subset of machine learning based on artificial neural networks.",
                "page_number": 1,
                "source": "deep_learning.pdf",
                "distance": 0.12,
            }
        ]
        mock_llm.return_value = (
            "Deep learning is a specialized branch of machine learning utilizing artificial neural networks."
        )

        response = client.post(
            "/api/chat",
            json={"question": "What is deep learning?", "conversation_id": None, "document_filename": "deep_learning.pdf"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "Deep learning" in data["answer"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["source"] == "deep_learning.pdf"
        assert data["sources"][0]["page_number"] == 1
        assert data["conversation_id"] is not None

    @patch("app.routes.chat_routes.vector_store_service.search")
    @patch("app.routes.chat_routes.llm_service.generate_answer")
    def test_ui_scenario_3_follow_up_question_memory(self, mock_llm, mock_search):
        """TEST 3: Ask a follow-up question -> Conversation memory works across turns."""
        mock_search.return_value = [
            {
                "chunk_id": "dl_chunk_1",
                "text": "Types of neural networks include CNNs and RNNs.",
                "page_number": 1,
                "source": "deep_learning.pdf",
                "distance": 0.15,
            }
        ]
        mock_llm.return_value = "Its primary types include Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs)."

        # Turn 1
        turn1 = client.post(
            "/api/chat",
            json={"question": "What is deep learning?", "conversation_id": None, "document_filename": "deep_learning.pdf"}
        )
        conv_id = turn1.json()["conversation_id"]

        # Turn 2: Follow-up question using the session ID
        turn2 = client.post(
            "/api/chat",
            json={"question": "What are its types?", "conversation_id": conv_id, "document_filename": "deep_learning.pdf"}
        )

        assert turn2.status_code == 200
        data = turn2.json()
        assert data["conversation_id"] == conv_id
        # Verify LLM was called with history preserved in call kwargs
        call_kwargs = mock_llm.call_args.kwargs
        assert call_kwargs["question"] == "What are its types?"
        passed_history = call_kwargs.get("chat_history", [])
        assert len(passed_history) >= 2
        assert any("What is deep learning?" in msg["content"] for msg in passed_history)

    @patch("app.routes.chat_routes.vector_store_service.search")
    @patch("app.routes.chat_routes.llm_service.generate_answer")
    def test_ui_scenario_4_unrelated_question_grounding(self, mock_llm, mock_search):
        """TEST 4: Ask an unrelated question -> Grounded no-answer behavior."""
        mock_search.return_value = []
        mock_llm.return_value = (
            "I could not find any relevant information in the provided document to answer your question."
        )

        response = client.post(
            "/api/chat",
            json={"question": "What is the capital of Mars?", "conversation_id": None, "document_filename": "deep_learning.pdf"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "not find any relevant information" in data["answer"].lower()
        assert data["sources"] == []

    @patch("app.routes.chat_routes.vector_store_service.search")
    @patch("app.routes.chat_routes.llm_service.generate_answer")
    def test_ui_scenario_5_new_conversation_memory_isolation(self, mock_llm, mock_search):
        """TEST 5: Start a new conversation -> Previous conversation memory is not reused."""
        mock_search.return_value = []
        mock_llm.return_value = "Answer 1"

        # Conversation A
        res_a = client.post("/api/chat", json={"question": "Question A", "conversation_id": None, "document_filename": "deep_learning.pdf"})
        conv_a = res_a.json()["conversation_id"]

        # Reset / New Conversation B
        res_b = client.post("/api/chat", json={"question": "Question B", "conversation_id": None, "document_filename": "deep_learning.pdf"})
        conv_b = res_b.json()["conversation_id"]

        assert conv_a != conv_b
        # Verify history for conv_b only contains Question B
        history_b = conversation_service.get_history(conv_b)
        assert len(history_b) == 2  # 1 user + 1 assistant
        assert history_b[0]["content"] == "Question B"

    def test_ui_scenario_6_invalid_pdf_handling(self):
        """TEST 6: Try an invalid PDF -> Friendly error returned."""
        # Non-pdf file extension
        res1 = client.post(
            "/api/documents/upload",
            files={"file": ("virus.exe", io.BytesIO(b"malicious payload"), "application/octet-stream")}
        )
        assert res1.status_code == 400
        assert "Only PDF files" in res1.json()["detail"]

        # Corrupted PDF content with .pdf extension
        res2 = client.post(
            "/api/documents/upload",
            files={"file": ("corrupt.pdf", io.BytesIO(b"NOT A REAL PDF FILE CONTENT"), "application/pdf")}
        )
        assert res2.status_code == 400
        assert "valid PDF" in res2.json()["detail"]

    def test_ui_scenario_7_empty_question_validation(self):
        """TEST 7: Try submitting an empty question -> Validation message."""
        response = client.post(
            "/api/chat",
            json={"question": "   ", "conversation_id": None, "document_filename": "deep_learning.pdf"}
        )
        assert response.status_code == 400
        assert "must not be empty" in response.json()["detail"]

    def test_ui_scenario_8_backend_health_and_restart(self):
        """TEST 8: Health check endpoint responds accurately for frontend status pill."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "PDF Chatbot API" in data["message"]
