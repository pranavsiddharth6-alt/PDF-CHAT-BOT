import pytest
from app.services.chunking_service import chunk_text, ChunkingError


def test_chunk_text_no_pages():
    """Test that empty pages list raises ChunkingError."""
    with pytest.raises(ChunkingError, match="No page content provided"):
        chunk_text([], "sample.pdf")


def test_chunk_text_invalid_chunk_size():
    """Test that chunk_size <= 0 raises ChunkingError."""
    pages = [{"page_number": 1, "text": "Some text"}]
    with pytest.raises(ChunkingError, match="chunk_size must be a positive integer"):
        chunk_text(pages, "sample.pdf", chunk_size=0)


def test_chunk_text_invalid_overlap():
    """Test that overlap >= chunk_size raises ChunkingError."""
    pages = [{"page_number": 1, "text": "Some text"}]
    with pytest.raises(ChunkingError, match="chunk_overlap must be non-negative and strictly less than chunk_size"):
        chunk_text(pages, "sample.pdf", chunk_size=100, chunk_overlap=100)


def test_chunk_text_negative_overlap():
    """Test that overlap < 0 raises ChunkingError."""
    pages = [{"page_number": 1, "text": "Some text"}]
    with pytest.raises(ChunkingError, match="chunk_overlap must be non-negative and strictly less than chunk_size"):
        chunk_text(pages, "sample.pdf", chunk_size=100, chunk_overlap=-10)


def test_chunk_text_empty_and_whitespace_pages():
    """Test that pages with only whitespace or empty text return empty chunk list."""
    pages = [
        {"page_number": 1, "text": ""},
        {"page_number": 2, "text": "   \n\t  "},
    ]
    chunks = chunk_text(pages, "sample.pdf", chunk_size=100, chunk_overlap=20)
    assert chunks == []


def test_chunk_text_short_text():
    """Test chunking text shorter than chunk_size."""
    pages = [
        {"page_number": 1, "text": "Short page content for test."},
        {"page_number": 2, "text": "Another short page content."}
    ]
    chunks = chunk_text(pages, "sample.pdf", chunk_size=100, chunk_overlap=20)

    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == 1
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["source"] == "sample.pdf"
    assert chunks[0]["text"] == "Short page content for test."

    assert chunks[1]["chunk_id"] == 2
    assert chunks[1]["page_number"] == 2
    assert chunks[1]["source"] == "sample.pdf"


def test_chunk_text_exact_boundary():
    """Test chunking text whose length exactly matches chunk_size."""
    exact_text = "A" * 100
    pages = [{"page_number": 1, "text": exact_text}]
    chunks = chunk_text(pages, "boundary.pdf", chunk_size=100, chunk_overlap=20)

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == 1
    assert chunks[0]["text"] == exact_text
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["source"] == "boundary.pdf"


def test_chunk_text_overlap_content_verification():
    """Test that consecutive chunks from the same page share overlapping text content."""
    sentence = "The quick brown fox jumps over the lazy dog repeatedly until the page limit is reached. " * 5
    pages = [{"page_number": 1, "text": sentence}]

    chunks = chunk_text(pages, "overlap.pdf", chunk_size=150, chunk_overlap=40)

    assert len(chunks) > 1
    for i in range(len(chunks) - 1):
        curr_text = chunks[i]["text"]
        next_text = chunks[i + 1]["text"]
        # Check that there is shared substring between end of curr and start of next
        words_in_curr = curr_text.split()[-3:]
        joined_words = " ".join(words_in_curr)
        assert any(word in next_text for word in words_in_curr), f"No overlap found between chunk {i+1} and {i+2}"


def test_chunk_text_multipage_sequential_ids():
    """Test that chunk IDs increment sequentially across multiple pages."""
    pages = [
        {"page_number": 1, "text": "Long page one text " * 20},
        {"page_number": 2, "text": "Long page two text " * 20},
        {"page_number": 3, "text": "Short page three text."},
    ]

    chunks = chunk_text(pages, "multi.pdf", chunk_size=100, chunk_overlap=20)

    assert len(chunks) >= 3
    # Check continuous 1-based indexing
    expected_ids = list(range(1, len(chunks) + 1))
    actual_ids = [c["chunk_id"] for c in chunks]
    assert actual_ids == expected_ids

    # Check that sources and page numbers are correctly attached
    page_numbers = [c["page_number"] for c in chunks]
    assert 1 in page_numbers
    assert 2 in page_numbers
    assert 3 in page_numbers
    assert all(c["source"] == "multi.pdf" for c in chunks)
