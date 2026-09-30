import pytest
from app.services.conversation_service import (
    ConversationService,
    conversation_service,
    DEFAULT_MAX_HISTORY_MESSAGES,
)


def test_conversation_service_create_id():
    """Test generating a unique conversation ID."""
    service = ConversationService()
    id1 = service.create_conversation_id()
    id2 = service.create_conversation_id()
    assert isinstance(id1, str)
    assert len(id1) >= 8
    assert id1 != id2


def test_conversation_service_add_and_get_history():
    """Test adding user and assistant messages and retrieving history."""
    service = ConversationService()
    cid = service.create_conversation_id()

    service.add_message(cid, "user", "What is supervised learning?")
    service.add_message(cid, "assistant", "Supervised learning uses labeled data.")

    history = service.get_history(cid)
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "What is supervised learning?"}
    assert history[1] == {"role": "assistant", "content": "Supervised learning uses labeled data."}


def test_conversation_service_history_limit():
    """Test that get_history respects the message limit."""
    service = ConversationService(max_history_messages=4)
    cid = service.create_conversation_id()

    # Add 6 messages (3 full turns)
    for i in range(1, 4):
        service.add_message(cid, "user", f"Question {i}")
        service.add_message(cid, "assistant", f"Answer {i}")

    # Should only return the 4 most recent messages
    recent_history = service.get_history(cid)
    assert len(recent_history) == 4
    assert recent_history[0] == {"role": "user", "content": "Question 2"}
    assert recent_history[1] == {"role": "assistant", "content": "Answer 2"}
    assert recent_history[2] == {"role": "user", "content": "Question 3"}
    assert recent_history[3] == {"role": "assistant", "content": "Answer 3"}


def test_conversation_service_session_isolation():
    """Test that two conversation IDs maintain separate independent histories."""
    service = ConversationService()
    cid_a = "session_A"
    cid_b = "session_B"

    service.add_message(cid_a, "user", "Topic A question")
    service.add_message(cid_a, "assistant", "Topic A answer")

    service.add_message(cid_b, "user", "Topic B question")

    history_a = service.get_history(cid_a)
    history_b = service.get_history(cid_b)

    assert len(history_a) == 2
    assert history_a[0]["content"] == "Topic A question"

    assert len(history_b) == 1
    assert history_b[0]["content"] == "Topic B question"


def test_conversation_service_clear_history():
    """Test clearing conversation history."""
    service = ConversationService()
    cid = service.create_conversation_id()

    service.add_message(cid, "user", "Hello")
    assert service.conversation_exists(cid) is True

    cleared = service.clear_history(cid)
    assert cleared is True
    assert service.get_history(cid) == []
    assert service.conversation_exists(cid) is False

    # Clearing again returns False
    assert service.clear_history(cid) is False


def test_conversation_service_empty_queries():
    """Test get_history on non-existent conversation returns empty list."""
    service = ConversationService()
    assert service.get_history("non_existent_id") == []
    assert service.get_history("") == []


def test_conversation_service_retention_cap():
    """Test that in-memory storage caps total messages to prevent memory leaks."""
    service = ConversationService(max_history_messages=4)
    cid = service.create_conversation_id()

    for i in range(50):
        service.add_message(cid, "user", f"Message {i}")

    # Internal storage should be capped and not grow to 50
    assert len(service._conversations[cid]) <= 20
