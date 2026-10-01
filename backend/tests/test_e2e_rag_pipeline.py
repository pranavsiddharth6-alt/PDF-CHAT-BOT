import fitz
import numpy as np
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from main import app
from app.services.vector_store_service import vector_store_service
from app.services.llm_service import llm_service
from app.services.conversation_service import conversation_service

client = TestClient(app)


def generate_sample_pdf_bytes() -> bytes:
    """Creates an in-memory 2-page PDF document about Machine Learning."""
    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text(
        (50, 50),
        "Machine Learning Overview.\n\n"
        "Supervised learning is an approach where models are trained using labeled datasets.\n"
        "Its main types are classification and regression algorithms.\n"
        "Classification predicts discrete categories, while regression predicts continuous numbers."
    )

    page2 = doc.new_page()
    page2.insert_text(
        (50, 50),
        "Unsupervised and Reinforcement Learning.\n\n"
        "Unsupervised learning deals with unlabeled data to discover hidden patterns and clusters.\n"
        "Reinforcement learning trains agents via rewards and penalties in interactive environments."
    )

    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


class InMemoryChromaCollection:
    """
    Lightweight in-memory vector collection to test end-to-end vector storage
    and cosine similarity retrieval without external network calls.
    """

    def __init__(self):
        self.store = {}

    def upsert(self, ids, documents, embeddings, metadatas):
        for i, doc_id in enumerate(ids):
            self.store[doc_id] = {
                "document": documents[i],
                "embedding": np.array(embeddings[i], dtype=np.float32),
                "metadata": metadatas[i],
            }

    def query(self, query_embeddings, n_results, include, where=None):
        if not self.store:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        q_vec = np.array(query_embeddings[0], dtype=np.float32)
        scored = []

        for doc_id, entry in self.store.items():
            doc_vec = entry["embedding"]
            cosine_dist = float(max(0.0, 1.0 - np.dot(q_vec, doc_vec)))
            scored.append((cosine_dist, doc_id, entry["document"], entry["metadata"]))

        scored.sort(key=lambda x: x[0])
        top_matches = scored[:n_results]

        return {
            "ids": [[m[1] for m in top_matches]],
            "documents": [[m[2] for m in top_matches]],
            "metadatas": [[m[3] for m in top_matches]],
            "distances": [[m[0] for m in top_matches]],
        }

    def count(self):
        return len(self.store)


@pytest.fixture(autouse=True)
def setup_in_memory_vector_store():
    """Sets up an in-memory vector store on vector_store_service before each test."""
    in_memory_col = InMemoryChromaCollection()
    original_client = vector_store_service._client
    original_collection = vector_store_service._collection

    vector_store_service._client = MagicMock()
    vector_store_service._collection = in_memory_col

    yield in_memory_col

    vector_store_service._client = original_client
    vector_store_service._collection = original_collection


def test_full_e2e_rag_pipeline_upload_to_chat(setup_in_memory_vector_store):
    """
    End-to-End Test: Upload PDF -> Semantic Search -> Grounded RAG Chat.
    """
    pdf_bytes = generate_sample_pdf_bytes()
    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("ml_handbook.pdf", pdf_bytes, "application/pdf")}
    )

    assert upload_response.status_code == 200
    upload_data = upload_response.json()
    assert upload_data["filename"] == "ml_handbook.pdf"
    assert upload_data["total_pages"] == 2
    assert upload_data["vectors_stored"] >= 2

    # Semantic Search
    search_response = client.post(
        "/api/search",
        json={"query": "What is supervised learning?", "top_k": 2}
    )
    assert search_response.status_code == 200
    assert len(search_response.json()["results"]) == 2

    # RAG Chat
    mock_hf_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Supervised learning uses labeled datasets."
    mock_hf_client.chat_completion.return_value = MagicMock(choices=[mock_choice])

    original_llm_client = llm_service._client
    llm_service._client = mock_hf_client

    try:
        chat_response = client.post(
            "/api/chat",
            json={"question": "What is supervised learning?", "top_k": 2, "document_filename": "ml_handbook.pdf"}
        )
        assert chat_response.status_code == 200
        chat_data = chat_response.json()

        assert chat_data["answer"] == "Supervised learning uses labeled datasets."
        assert len(chat_data["sources"]) >= 1
        assert chat_data["sources"][0]["source"] == "ml_handbook.pdf"
        assert chat_data["sources"][0]["page_number"] == 1
        assert "conversation_id" in chat_data
    finally:
        llm_service._client = original_llm_client


def test_phase7_test1_basic_conversation_followup(setup_in_memory_vector_store):
    """
    TEST 1 — Basic multi-turn conversation:
    Turn 1: 'What is supervised learning?'
    Turn 2: 'What are its types?'
    Verify second question passes previous conversation history to LLM.
    """
    pdf_bytes = generate_sample_pdf_bytes()
    client.post("/api/documents/upload", files={"file": ("ml.pdf", pdf_bytes, "application/pdf")})

    mock_hf_client = MagicMock()
    original_llm_client = llm_service._client
    llm_service._client = mock_hf_client

    try:
        # Turn 1
        choice1 = MagicMock()
        choice1.message.content = "Supervised learning trains models on labeled data."
        mock_hf_client.chat_completion.return_value = MagicMock(choices=[choice1])

        resp1 = client.post("/api/chat", json={"question": "What is supervised learning?", "document_filename": "ml.pdf"})
        assert resp1.status_code == 200
        cid = resp1.json()["conversation_id"]

        # Turn 2 (Follow-up using the same conversation_id)
        choice2 = MagicMock()
        choice2.message.content = "Its main types are classification and regression."
        mock_hf_client.chat_completion.return_value = MagicMock(choices=[choice2])

        resp2 = client.post("/api/chat", json={"conversation_id": cid, "question": "What are its types?", "document_filename": "ml.pdf"})
        assert resp2.status_code == 200
        assert resp2.json()["conversation_id"] == cid
        assert resp2.json()["answer"] == "Its main types are classification and regression."

        # Verify that Turn 2's LLM prompt contained the history from Turn 1
        call_messages = mock_hf_client.chat_completion.call_args[1]["messages"]

        roles = [m["role"] for m in call_messages]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles

        # Check history injection
        assert call_messages[1]["role"] == "user"
        assert call_messages[1]["content"] == "What is supervised learning?"
        assert call_messages[2]["role"] == "assistant"
        assert call_messages[2]["content"] == "Supervised learning trains models on labeled data."

        # Check current turn with retrieved context
        assert call_messages[3]["role"] == "user"
        assert "What are its types?" in call_messages[3]["content"]

    finally:
        llm_service._client = original_llm_client


def test_phase7_test2_three_turn_conversation(setup_in_memory_vector_store):
    """
    TEST 2 — Three-turn conversation maintaining continuous dialogue history:
    Turn 1: 'What is machine learning?'
    Turn 2: 'What are its types?'
    Turn 3: 'Which one uses labeled data?'
    """
    pdf_bytes = generate_sample_pdf_bytes()
    client.post("/api/documents/upload", files={"file": ("ml.pdf", pdf_bytes, "application/pdf")})

    mock_hf_client = MagicMock()
    original_llm_client = llm_service._client
    llm_service._client = mock_hf_client

    try:
        # Turn 1
        choice1 = MagicMock()
        choice1.message.content = "Machine learning is a field of AI."
        mock_hf_client.chat_completion.return_value = MagicMock(choices=[choice1])
        r1 = client.post("/api/chat", json={"question": "What is machine learning?", "document_filename": "ml.pdf"})
        cid = r1.json()["conversation_id"]

        # Turn 2
        choice2 = MagicMock()
        choice2.message.content = "Types include supervised, unsupervised, and reinforcement."
        mock_hf_client.chat_completion.return_value = MagicMock(choices=[choice2])
        r2 = client.post("/api/chat", json={"conversation_id": cid, "question": "What are its types?", "document_filename": "ml.pdf"})

        # Turn 3
        choice3 = MagicMock()
        choice3.message.content = "Supervised learning uses labeled data."
        mock_hf_client.chat_completion.return_value = MagicMock(choices=[choice3])
        r3 = client.post("/api/chat", json={"conversation_id": cid, "question": "Which one uses labeled data?", "document_filename": "ml.pdf"})

        assert r3.status_code == 200
        assert r3.json()["answer"] == "Supervised learning uses labeled data."

        # Verify all previous turns exist in the Turn 3 prompt
        turn3_messages = mock_hf_client.chat_completion.call_args[1]["messages"]
        # System + Turn 1 (user+assistant) + Turn 2 (user+assistant) + Turn 3 (user) = 6 messages
        assert len(turn3_messages) == 6
        assert turn3_messages[1]["content"] == "What is machine learning?"
        assert turn3_messages[2]["content"] == "Machine learning is a field of AI."
        assert turn3_messages[3]["content"] == "What are its types?"
        assert turn3_messages[4]["content"] == "Types include supervised, unsupervised, and reinforcement."
        assert "Which one uses labeled data?" in turn3_messages[5]["content"]

    finally:
        llm_service._client = original_llm_client


def test_phase7_test3_new_conversation_isolation(setup_in_memory_vector_store):
    """
    TEST 3 — New conversation does NOT use history from a previous conversation ID.
    """
    pdf_bytes = generate_sample_pdf_bytes()
    client.post("/api/documents/upload", files={"file": ("ml.pdf", pdf_bytes, "application/pdf")})

    mock_hf_client = MagicMock()
    original_llm_client = llm_service._client
    llm_service._client = mock_hf_client

    try:
        # Session A
        choice_a = MagicMock()
        choice_a.message.content = "Quantum computing details."
        mock_hf_client.chat_completion.return_value = MagicMock(choices=[choice_a])
        r_a = client.post("/api/chat", json={"question": "Tell me about quantum computing.", "document_filename": "ml.pdf"})
        cid_a = r_a.json()["conversation_id"]

        # Session B (new conversation without conversation_id)
        choice_b = MagicMock()
        choice_b.message.content = "Classification and regression."
        mock_hf_client.chat_completion.return_value = MagicMock(choices=[choice_b])
        r_b = client.post("/api/chat", json={"question": "What are its types?", "document_filename": "ml.pdf"})
        cid_b = r_b.json()["conversation_id"]

        assert cid_a != cid_b

        # Verify Session B did NOT receive Session A's messages in prompt
        session_b_messages = mock_hf_client.chat_completion.call_args[1]["messages"]
        # System + current Turn = 2 messages
        assert len(session_b_messages) == 2
        assert "quantum computing" not in session_b_messages[1]["content"].lower()

    finally:
        llm_service._client = original_llm_client


def test_phase7_test4_pdf_grounding_and_sources(setup_in_memory_vector_store):
    """
    TEST 4 — PDF grounding: Question answered by PDF returns answer + source + page.
    """
    pdf_bytes = generate_sample_pdf_bytes()
    client.post("/api/documents/upload", files={"file": ("textbook.pdf", pdf_bytes, "application/pdf")})

    mock_hf_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Regression predicts continuous numbers."
    mock_hf_client.chat_completion.return_value = MagicMock(choices=[mock_choice])

    original_llm_client = llm_service._client
    llm_service._client = mock_hf_client

    try:
        resp = client.post("/api/chat", json={"question": "What does regression predict?", "document_filename": "textbook.pdf"})
        assert resp.status_code == 200
        data = resp.json()

        assert data["answer"] == "Regression predicts continuous numbers."
        assert len(data["sources"]) >= 1
        assert data["sources"][0]["source"] == "textbook.pdf"
        assert data["sources"][0]["page_number"] == 1
    finally:
        llm_service._client = original_llm_client


def test_phase7_test5_unanswerable_question_no_hallucination(setup_in_memory_vector_store):
    """
    TEST 5 — Unanswerable question: Model correctly reports information is not found in PDF.
    """
    mock_hf_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "The requested information was not found in the uploaded document."
    mock_hf_client.chat_completion.return_value = MagicMock(choices=[mock_choice])

    original_llm_client = llm_service._client
    llm_service._client = mock_hf_client

    try:
        resp = client.post("/api/chat", json={"question": "What is the secret recipe for chocolate cake?", "document_filename": "ml.pdf"})
        assert resp.status_code == 200
        data = resp.json()

        assert data["answer"] == "The requested information was not found in the uploaded document."
        assert data["sources"] == []
    finally:
        llm_service._client = original_llm_client


def test_phase7_test6_history_limit(setup_in_memory_vector_store):
    """
    TEST 6 — History limit: Adding many turns caps the history passed to LLM.
    """
    pdf_bytes = generate_sample_pdf_bytes()
    client.post("/api/documents/upload", files={"file": ("ml.pdf", pdf_bytes, "application/pdf")})

    mock_hf_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Answer content."
    mock_hf_client.chat_completion.return_value = MagicMock(choices=[mock_choice])

    original_llm_client = llm_service._client
    llm_service._client = mock_hf_client

    try:
        cid = "capped_session"
        # Run 8 questions in the same session
        for i in range(1, 9):
            client.post("/api/chat", json={"conversation_id": cid, "question": f"Question {i}", "document_filename": "ml.pdf"})

        # Check prompt for the 8th turn: should have system (1) + max 6 history messages + current user message (1) = 8 messages total
        last_call_messages = mock_hf_client.chat_completion.call_args[1]["messages"]
        assert len(last_call_messages) <= 8

    finally:
        llm_service._client = original_llm_client
