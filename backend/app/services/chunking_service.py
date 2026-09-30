import logging

logger = logging.getLogger(__name__)


class ChunkingError(Exception):
    """Custom exception raised when text chunking fails."""
    pass


def chunk_text(pages: list[dict], source_filename: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
    """
    Splits page-wise PDF text into smaller overlapping text chunks while preserving metadata.

    :param pages: List of dicts containing 'page_number' and 'text'.
    :param source_filename: Name of the source PDF document.
    :param chunk_size: Target maximum character length for each chunk.
    :param chunk_overlap: Overlapping character length between consecutive chunks.
    :return: List of chunk dictionaries containing text, page_number, and source filename.
    """
    if not pages:
        logger.warning(f"Chunking failed for '{source_filename}': no page content provided.")
        raise ChunkingError("No page content provided for chunking.")

    if chunk_size <= 0:
        logger.warning(f"Invalid chunk_size {chunk_size} provided.")
        raise ChunkingError("chunk_size must be a positive integer.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        logger.warning(f"Invalid chunk_overlap {chunk_overlap} for chunk_size {chunk_size}.")
        raise ChunkingError("chunk_overlap must be non-negative and strictly less than chunk_size.")

    logger.info(
        f"Chunking text from '{source_filename}' ({len(pages)} page(s), "
        f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap})..."
    )

    chunks = []
    chunk_index = 1

    for page in pages:
        page_num = page.get("page_number", 1)
        text = page.get("text", "").strip()

        if not text:
            continue

        # If page text is within chunk limit, save as a single chunk
        if len(text) <= chunk_size:
            chunks.append({
                "chunk_id": chunk_index,
                "text": text,
                "page_number": page_num,
                "source": source_filename
            })
            chunk_index += 1
            continue

        # Otherwise, split page text into overlapping windows
        start = 0
        text_length = len(text)
        step = chunk_size - chunk_overlap

        while start < text_length:
            end = start + chunk_size
            chunk_str = text[start:end]

            # Try to snap to the nearest space or newline if not at the very end of text
            if end < text_length:
                last_space = chunk_str.rfind(" ")
                if last_space > chunk_size // 2:
                    chunk_str = chunk_str[:last_space]

            cleaned_chunk = chunk_str.strip()
            if cleaned_chunk:
                chunks.append({
                    "chunk_id": chunk_index,
                    "text": cleaned_chunk,
                    "page_number": page_num,
                    "source": source_filename
                })
                chunk_index += 1

            start += len(chunk_str) if len(chunk_str) > 0 else step
            # Account for overlap back-track
            if start < text_length:
                start = max(start - chunk_overlap, 0)
                # Avoid infinite loops if start does not advance
                if start >= (end - chunk_overlap):
                    start = end

    logger.info(f"Generated {len(chunks)} chunk(s) for '{source_filename}'.")
    return chunks
