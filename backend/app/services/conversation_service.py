import logging
import uuid
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Default maximum number of messages (user + assistant) to retain per conversation
DEFAULT_MAX_HISTORY_MESSAGES = 6


class ConversationService:
    """
    In-memory conversation history service for managing multi-turn dialogue context.
    Maps conversation_id -> list of message dicts: [{"role": "user"|"assistant", "content": str}].
    """

    def __init__(self, max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES):
        self._conversations: Dict[str, List[Dict[str, str]]] = {}
        self.max_history_messages = max_history_messages

    def create_conversation_id(self) -> str:
        """Generates a unique, URL-safe conversation ID."""
        return uuid.uuid4().hex[:12]

    def get_history(self, conversation_id: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Retrieves the most recent message history for a conversation up to the specified limit.

        :param conversation_id: The unique conversation identifier.
        :param limit: Maximum number of recent messages to return (defaults to self.max_history_messages).
        :return: List of message dictionaries.
        """
        if not conversation_id or conversation_id not in self._conversations:
            return []

        history = self._conversations[conversation_id]
        max_limit = limit if limit is not None else self.max_history_messages
        return history[-max_limit:] if max_limit > 0 else []

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        """
        Appends a message to the conversation history.

        :param conversation_id: The unique conversation identifier.
        :param role: 'user' or 'assistant'.
        :param content: Text content of the message.
        """
        if not conversation_id:
            return

        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = []

        self._conversations[conversation_id].append({
            "role": role,
            "content": content.strip()
        })

        # Cap memory to avoid unbounded growth (keep up to 2x max_history_messages)
        retention_cap = max(self.max_history_messages * 4, 20)
        if len(self._conversations[conversation_id]) > retention_cap:
            self._conversations[conversation_id] = self._conversations[conversation_id][-retention_cap:]

        logger.info(
            f"Added {role} message to conversation '{conversation_id}'. "
            f"Total turns in memory: {len(self._conversations[conversation_id])}."
        )

    def clear_history(self, conversation_id: str) -> bool:
        """
        Clears the stored history for a given conversation.

        :param conversation_id: The conversation identifier to clear.
        :return: True if history was found and cleared, False otherwise.
        """
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            logger.info(f"Cleared history for conversation '{conversation_id}'.")
            return True
        return False

    def conversation_exists(self, conversation_id: str) -> bool:
        """Checks whether a conversation ID exists in memory."""
        return conversation_id in self._conversations


# Reusable singleton instance
conversation_service = ConversationService()
