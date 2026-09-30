import logging
import os
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

COLLECTION_NAME = "pdf_chatbot_chunks"


class VectorStoreError(Exception):
    """Custom exception raised when vector store operations fail."""
    pass


def _sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename for use as part of a Chroma document ID.
    Replaces any character that is not alphanumeric, hyphen, or underscore with underscore.
    """
    name = os.path.splitext(filename)[0]  # strip extension
    name = re.sub(r"[^a-zA-Z0-9\-]", "_", name)
    return name


def _make_chunk_id(source_filename: str, page_number: int, chunk_id: int) -> str:
    """
    Create a deterministic, unique chunk ID for Chroma from document metadata.

    Format: <sanitized_filename>__page_<page>__chunk_<id>
    """
    safe_name = _sanitize_filename(source_filename)
    return f"{safe_name}__page_{page_number}__chunk_{chunk_id}"


class VectorStoreService:
    """
    Service for connecting to Chroma Cloud, storing document chunk embeddings,
    and performing vector similarity search.

    Credentials are loaded from environment variables:
        CHROMA_API_KEY  - Chroma Cloud API key
        CHROMA_TENANT   - Chroma Cloud tenant name
        CHROMA_DATABASE - Chroma Cloud database name

    All environment variables must be set in backend/.env (never hard-coded).
    """

    _instance = None
    _client = None
    _collection = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_credentials(self) -> tuple[str, str, str]:
        """
        Load Chroma Cloud credentials from environment variables.
        Raises VectorStoreError if any required credential is missing.
        """
        api_key = os.getenv("CHROMA_API_KEY", "").strip()
        tenant = os.getenv("CHROMA_TENANT", "").strip()
        database = os.getenv("CHROMA_DATABASE", "").strip()

        missing = []
        if not api_key:
            missing.append("CHROMA_API_KEY")
        if not tenant:
            missing.append("CHROMA_TENANT")
        if not database:
            missing.append("CHROMA_DATABASE")

        if missing:
            raise VectorStoreError(
                f"Missing required Chroma Cloud environment variable(s): {', '.join(missing)}. "
                f"Set them in backend/.env (see .env.example)."
            )

        return api_key, tenant, database

    def _connect(self):
        """
        Lazily initialize the Chroma Cloud client and get/create the collection.
        The collection uses no built-in embedding function — embeddings are
        always supplied pre-computed from embedding_service.py.
        """
        if self._client is not None and self._collection is not None:
            return

        try:
            import chromadb
        except ImportError:
            raise VectorStoreError(
                "chromadb package is not installed. Run: pip install chromadb"
            )

        api_key, tenant, database = self._load_credentials()

        logger.info(
            f"Connecting to Chroma Cloud (tenant='{tenant}', database='{database}')..."
        )
        try:
            self._client = chromadb.CloudClient(
                tenant=tenant,
                database=database,
                api_key=api_key,
            )
        except Exception as e:
            raise VectorStoreError(
                f"Failed to connect to Chroma Cloud: {str(e)}"
            )

        # Get or create the collection — no embedding_function means we supply
        # pre-computed embeddings explicitly on every add/query call.
        try:
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"Chroma collection '{COLLECTION_NAME}' ready "
                f"({self._collection.count()} existing vectors)."
            )
        except Exception as e:
            raise VectorStoreError(
                f"Failed to get/create Chroma collection '{COLLECTION_NAME}': {str(e)}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:
        """
        Store document chunks with their pre-computed embeddings in Chroma Cloud.

        :param chunks: List of chunk dicts (text, page_number, source, chunk_id).
        :param embeddings: Parallel list of 384-dimensional embedding vectors.
        :return: Number of vectors successfully stored.
        :raises VectorStoreError: On any storage failure.
        """
        if not chunks:
            raise VectorStoreError("No chunks provided to store.")
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Chunk count ({len(chunks)}) does not match embedding count ({len(embeddings)})."
            )

        self._connect()

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        embedding_list: List[List[float]] = []

        for chunk, embedding in zip(chunks, embeddings):
            chunk_id_str = _make_chunk_id(
                source_filename=chunk["source"],
                page_number=chunk["page_number"],
                chunk_id=chunk["chunk_id"],
            )
            ids.append(chunk_id_str)
            documents.append(chunk["text"])
            metadatas.append({
                "source": chunk["source"],
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
            })
            embedding_list.append(embedding)

        try:
            self._collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embedding_list,
                metadatas=metadatas,
            )
            logger.info(f"Stored {len(ids)} vectors in Chroma Cloud collection '{COLLECTION_NAME}'.")
            return len(ids)
        except Exception as e:
            raise VectorStoreError(f"Failed to store vectors in Chroma Cloud: {str(e)}")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform a vector similarity search against Chroma Cloud.

        :param query_embedding: 384-dimensional query vector.
        :param top_k: Number of top results to return.
        :param where: Optional Chroma metadata filter dict.
        :return: List of result dicts with text, metadata, and distance.
        :raises VectorStoreError: On any search failure.
        """
        if not query_embedding:
            raise VectorStoreError("Empty query embedding provided.")
        if top_k < 1:
            raise VectorStoreError("top_k must be at least 1.")

        self._connect()

        try:
            query_kwargs: Dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                query_kwargs["where"] = where

            raw = self._collection.query(**query_kwargs)
        except Exception as e:
            raise VectorStoreError(f"Chroma similarity search failed: {str(e)}")

        results: List[Dict[str, Any]] = []
        if not raw or not raw.get("ids"):
            return results

        ids = raw["ids"][0]
        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]

        for i, doc_id in enumerate(ids):
            results.append({
                "chunk_id": doc_id,
                "text": documents[i],
                "source": metadatas[i].get("source", "unknown"),
                "page_number": metadatas[i].get("page_number", -1),
                "distance": round(distances[i], 6),
            })

        return results

    def collection_count(self) -> int:
        """Return the total number of vectors currently in the collection."""
        self._connect()
        try:
            return self._collection.count()
        except Exception as e:
            raise VectorStoreError(f"Failed to get collection count: {str(e)}")


# Reusable singleton instance
vector_store_service = VectorStoreService()
