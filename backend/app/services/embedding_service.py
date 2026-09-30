import logging
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingError(Exception):
    """Custom exception raised when embedding generation or model loading fails."""
    pass


class EmbeddingService:
    """
    Singleton service class for managing Hugging Face SentenceTransformer models
    and generating vector embeddings in batches.
    """
    _instance = None
    _model = None
    _model_name = DEFAULT_MODEL_NAME

    def __new__(cls, model_name: str = DEFAULT_MODEL_NAME):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._model_name = model_name
        return cls._instance

    def _load_model(self):
        """Loads the SentenceTransformer model lazily if not already initialized."""
        if self._model is None:
            try:
                logger.info(f"Loading SentenceTransformer model '{self._model_name}'...")
                self._model = SentenceTransformer(self._model_name)
                logger.info(f"Successfully loaded '{self._model_name}'.")
            except Exception as e:
                logger.error(f"Failed to load embedding model '{self._model_name}': {str(e)}")
                raise EmbeddingError(f"Failed to load embedding model '{self._model_name}': {str(e)}")

    def get_embedding_dimension(self) -> int:
        """Returns the vector dimension of the loaded embedding model."""
        self._load_model()
        try:
            dim = self._model.get_sentence_embedding_dimension()
            return dim
        except Exception as e:
            logger.error(f"Failed to get embedding dimension: {str(e)}")
            raise EmbeddingError(f"Failed to get embedding dimension: {str(e)}")

    def generate_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """
        Generates vector embeddings for a batch of text strings.

        :param texts: List of text strings to encode.
        :return: Tuple of (list of embedding float lists, embedding vector dimension).
        :raises EmbeddingError: If inputs are invalid or encoding fails.
        """
        if not texts:
            logger.warning("Embedding generation called with empty text list.")
            raise EmbeddingError("No text inputs provided for embedding generation.")

        # Filter out empty or whitespace-only strings safely
        valid_texts = [t.strip() for t in texts if t and t.strip()]
        if not valid_texts:
            logger.warning("All text inputs provided for embedding generation were empty or whitespace.")
            raise EmbeddingError("All provided text chunks were empty or whitespace-only.")

        self._load_model()

        try:
            logger.info(f"Encoding batch of {len(valid_texts)} text string(s)...")
            # Batch encode texts using sentence-transformers
            embeddings_ndarray = self._model.encode(
                valid_texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            # Convert numpy array to Python list of lists for serializability
            embeddings_list = embeddings_ndarray.tolist()
            dimension = self.get_embedding_dimension()
            logger.info(f"Successfully encoded {len(embeddings_list)} vector(s) (dimension={dimension}).")
            return embeddings_list, dimension
        except Exception as e:
            logger.error(f"Error during embedding generation: {str(e)}")
            raise EmbeddingError(f"Error generating embeddings: {str(e)}")


# Export a default reusable service instance
embedding_service = EmbeddingService()
