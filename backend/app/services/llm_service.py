import os
import logging
from typing import List, Dict, Any, Optional
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class LLMError(Exception):
    """Custom exception raised when LLM answer generation fails."""
    pass


class LLMService:
    """
    Service for generating natural language answers using Hugging Face Inference API
    with the instruction-following model Qwen/Qwen2.5-7B-Instruct, with support for
    conversation history and grounded document context.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._client = None

    def _get_token(self) -> str:
        """
        Loads Hugging Face API token from environment variables.
        Supports both HUGGINGFACEHUB_API_TOKEN and HF_TOKEN keys.
        """
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
        if token:
            token = token.strip()

        if not token or token == "your_huggingface_api_token_here":
            raise LLMError(
                "Hugging Face API token is missing or unconfigured. "
                "Please set a valid HUGGINGFACEHUB_API_TOKEN or HF_TOKEN in backend/.env"
            )
        return token

    def _get_client(self) -> InferenceClient:
        """Lazily initializes the Hugging Face InferenceClient."""
        if self._client is None:
            token = self._get_token()
            try:
                self._client = InferenceClient(api_key=token)
            except Exception as e:
                raise LLMError(f"Failed to initialize Hugging Face InferenceClient: {str(e)}")
        return self._client

    def build_prompt_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved vector store chunks into a clean, structured context string.

        Format:
        [Source: <source> | Page: <page_number>]

        <chunk text>
        """
        if not chunks:
            return "No relevant document context found."

        context_parts = []
        for chunk in chunks:
            source = chunk.get("source", "unknown")
            page_num = chunk.get("page_number", "N/A")
            text = chunk.get("text", "").strip()
            context_parts.append(f"[Source: {source} | Page: {page_num}]\n\n{text}")

        return "\n\n".join(context_parts)

    def generate_answer(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Generates a grounded natural language answer based on retrieved context chunks
        and previous conversation history.

        :param question: The user's question string.
        :param chunks: List of retrieved context chunk dictionaries from Chroma Cloud.
        :param chat_history: Optional list of past conversation message dicts [{"role": "user"|"assistant", "content": "..."}].
        :return: Natural language answer string.
        :raises LLMError: On API authentication failure, timeout, or model errors.
        """
        if not question or not question.strip():
            raise LLMError("Question cannot be empty or whitespace-only.")

        context_str = self.build_prompt_context(chunks)

        system_instruction = (
            "You are a PDF question-answering assistant.\n"
            "Answer the user's question using only the supplied document context and conversation history.\n"
            "Conversation history provides dialogue context for follow-up questions, while document context provides factual evidence from the uploaded PDF.\n"
            "If the answer is not present in the supplied document context, clearly state that the information was not found in the uploaded document.\n"
            "Do not invent information or use unrelated outside knowledge.\n"
            "Give a concise and understandable answer.\n"
            "Do not mention internal implementation details."
        )

        messages = [
            {"role": "system", "content": system_instruction}
        ]

        # Inject previous conversation turns into LLM message history
        if chat_history:
            for msg in chat_history:
                role = msg.get("role", "user")
                content = msg.get("content", "").strip()
                if content and role in ("user", "assistant"):
                    messages.append({"role": role, "content": content})

        # Inject retrieved document context along with current user question
        current_turn_content = f"CONTEXT:\n\n{context_str}\n\nQUESTION:\n\n{question.strip()}"
        messages.append({"role": "user", "content": current_turn_content})

        client = self._get_client()

        try:
            history_count = len(chat_history) if chat_history else 0
            logger.info(
                f"Calling Hugging Face Inference API with model '{self.model_name}' "
                f"({len(chunks)} chunk(s), {history_count} previous message(s))..."
            )
            response = client.chat_completion(
                messages=messages,
                model=self.model_name,
                max_tokens=512,
                temperature=0.2,
            )

            if not response or not response.choices:
                raise LLMError("No response choices returned by Hugging Face API.")

            answer = response.choices[0].message.content
            if not answer:
                raise LLMError("Received empty response content from Hugging Face API.")

            return answer.strip()
        except LLMError:
            raise
        except Exception as e:
            logger.error(f"Error calling Hugging Face model {self.model_name}: {str(e)}")
            raise LLMError(f"Hugging Face Inference API error: {str(e)}")


# Singleton instance export
llm_service = LLMService()
