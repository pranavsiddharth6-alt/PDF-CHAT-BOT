import os
import math
import logging
from typing import List, Tuple, Any
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
REQUIRED_DIMENSION = 384


class EmbeddingError(Exception):
    """Custom exception raised when embedding generation or API call fails."""
    pass


class EmbeddingService:
    """
    Singleton service class for generating 384-dimensional vector embeddings
    using Hugging Face Inference API for model 'sentence-transformers/all-MiniLM-L6-v2'.
    """
    _instance = None
    _client = None
    _model_name = DEFAULT_MODEL_NAME

    def __new__(cls, model_name: str = DEFAULT_MODEL_NAME):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._model_name = model_name
        return cls._instance

    def _get_token(self) -> str:
        """
        Loads Hugging Face API token from environment variables.
        Supports both HUGGINGFACEHUB_API_TOKEN and HF_TOKEN keys.
        """
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
        if token:
            token = token.strip()

        if not token or token == "your_huggingface_api_token_here":
            raise EmbeddingError(
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
                raise EmbeddingError(f"Failed to initialize Hugging Face InferenceClient: {str(e)}")
        return self._client

    def get_embedding_dimension(self) -> int:
        """Returns the vector dimension of the embedding model (384)."""
        return REQUIRED_DIMENSION

    def _validate_and_normalize(self, raw_data: Any, expected_count: int) -> List[List[float]]:
        """
        Validates structure and dimension (384) of raw embeddings from HF API,
        and applies L2 normalization.
        """
        if hasattr(raw_data, "tolist"):
            data = raw_data.tolist()
        else:
            data = raw_data

        if not isinstance(data, list) or len(data) == 0:
            raise EmbeddingError("Received empty or invalid response from Hugging Face Inference API.")

        # Handle 1D vector return when expected_count == 1
        if expected_count == 1 and isinstance(data[0], (int, float)):
            data = [data]

        # Handle 3D tensor return (batch, seq_len, dim) via mean pooling over tokens
        if isinstance(data[0], list) and len(data[0]) > 0 and isinstance(data[0][0], list):
            pooled = []
            for token_matrix in data:
                seq_len = len(token_matrix)
                dim = len(token_matrix[0])
                mean_vec = [
                    sum(token_matrix[t][d] for t in range(seq_len)) / seq_len
                    for d in range(dim)
                ]
                pooled.append(mean_vec)
            data = pooled

        if len(data) != expected_count:
            raise EmbeddingError(
                f"Expected {expected_count} embedding vector(s) from API, got {len(data)}."
            )

        embeddings: List[List[float]] = []
        for vec in data:
            if not isinstance(vec, list) or len(vec) != REQUIRED_DIMENSION:
                got_dim = len(vec) if isinstance(vec, list) else type(vec)
                raise EmbeddingError(
                    f"Invalid embedding dimension: expected {REQUIRED_DIMENSION}, got {got_dim}."
                )

            float_vec = [float(x) for x in vec]
            norm = math.sqrt(sum(x * x for x in float_vec))
            if norm > 0:
                float_vec = [x / norm for x in float_vec]
            embeddings.append(float_vec)

        return embeddings

    def generate_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """
        Generates 384-dimensional vector embeddings for a batch of text strings
        using Hugging Face Inference API.

        :param texts: List of text strings to encode.
        :return: Tuple of (list of embedding float lists, embedding vector dimension 384).
        :raises EmbeddingError: If inputs are invalid or API call fails.
        """
        if not texts:
            logger.warning("Embedding generation called with empty text list.")
            raise EmbeddingError("No text inputs provided for embedding generation.")

        valid_texts = [t.strip() for t in texts if t and t.strip()]
        if not valid_texts:
            logger.warning("All text inputs provided for embedding generation were empty or whitespace.")
            raise EmbeddingError("All provided text chunks were empty or whitespace-only.")

        client = self._get_client()

        try:
            logger.info(
                f"Calling Hugging Face Inference API for {len(valid_texts)} text string(s) "
                f"using model '{self._model_name}'..."
            )
            raw_response = client.feature_extraction(
                valid_texts,
                model=self._model_name
            )

            embeddings = self._validate_and_normalize(raw_response, expected_count=len(valid_texts))
            dimension = self.get_embedding_dimension()
            logger.info(
                f"Successfully generated {len(embeddings)} embedding vector(s) "
                f"(dimension={dimension})."
            )
            return embeddings, dimension
        except EmbeddingError:
            raise
        except Exception as e:
            err_msg = str(e)
            logger.error(f"Error calling Hugging Face Inference API for embeddings: {err_msg}")
            raise EmbeddingError(f"Hugging Face Inference API embedding error: {err_msg}")


# Export a default reusable service instance
embedding_service = EmbeddingService()

