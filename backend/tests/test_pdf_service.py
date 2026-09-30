import fitz
import pytest
from app.services.pdf_service import extract_pdf_text, PDFProcessingError
from app.services.chunking_service import chunk_text


def create_sample_pdf_bytes(text_pages: list[str]) -> bytes:
    """Helper to generate in-memory PDF byte stream for testing."""
    doc = fitz.open()
    for text in text_pages:
        page = doc.new_page()
        if text:
            page.insert_text((50, 50), text)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def test_extract_pdf_text_empty_bytes():
    """Test that empty file bytes raise PDFProcessingError."""
    with pytest.raises(PDFProcessingError, match="Uploaded file is empty"):
        extract_pdf_text(b"")


def test_extract_pdf_text_invalid_pdf_bytes():
    """Test that non-PDF bytes raise PDFProcessingError."""
    with pytest.raises(PDFProcessingError, match="Failed to parse PDF document"):
        extract_pdf_text(b"This is not a PDF file")


def test_extract_pdf_text_valid_pdf():
    """Test successful text extraction from a valid PDF."""
    sample_text_1 = "Hello, world! This is page 1 of the test PDF."
    sample_text_2 = "Welcome to page 2 of the test document."

    pdf_bytes = create_sample_pdf_bytes([sample_text_1, sample_text_2])
    result = extract_pdf_text(pdf_bytes)

    assert result["total_pages"] == 2
    assert len(result["pages"]) == 2

    assert result["pages"][0]["page_number"] == 1
    assert "Hello, world!" in result["pages"][0]["text"]

    assert result["pages"][1]["page_number"] == 2
    assert "Welcome to page 2" in result["pages"][1]["text"]


def test_extract_pdf_text_multipage_with_blank_page():
    """Test extraction with multiple pages including an empty page."""
    pdf_bytes = create_sample_pdf_bytes(["Page 1 content", "", "Page 3 content"])
    result = extract_pdf_text(pdf_bytes)

    assert result["total_pages"] == 3
    assert len(result["pages"]) == 3
    assert result["pages"][0]["text"] == "Page 1 content"
    assert result["pages"][1]["text"] == ""
    assert result["pages"][2]["text"] == "Page 3 content"


def test_extract_pdf_text_special_characters_and_newlines():
    """Test extraction of Unicode symbols, numbers, and newlines."""
    complex_text = "Line 1: Special characters: ©, ®, €, →, α, β.\nLine 2: Math equations: x^2 + y^2 = z^2."
    pdf_bytes = create_sample_pdf_bytes([complex_text])
    result = extract_pdf_text(pdf_bytes)

    assert result["total_pages"] == 1
    assert "Special characters" in result["pages"][0]["text"]
    assert "Line 2: Math equations" in result["pages"][0]["text"]


def test_pdf_extraction_to_chunking_pipeline():
    """Test the integrated pipeline from raw PDF bytes to chunked metadata."""
    page_1_text = "Artificial Intelligence is transforming industries across the globe."
    page_2_text = "Retrieval-Augmented Generation enhances large language models with external knowledge."
    page_3_text = "Vector databases store embeddings for high-dimensional semantic search."

    pdf_bytes = create_sample_pdf_bytes([page_1_text, page_2_text, page_3_text])

    # 1. Extraction
    extraction = extract_pdf_text(pdf_bytes)
    assert extraction["total_pages"] == 3
    assert len(extraction["pages"]) == 3

    # 2. Chunking
    filename = "ai_overview.pdf"
    chunks = chunk_text(pages=extraction["pages"], source_filename=filename, chunk_size=200, chunk_overlap=30)

    assert len(chunks) == 3
    assert [c["page_number"] for c in chunks] == [1, 2, 3]
    assert [c["chunk_id"] for c in chunks] == [1, 2, 3]
    assert all(c["source"] == filename for c in chunks)
    assert "Artificial Intelligence" in chunks[0]["text"]
    assert "Retrieval-Augmented Generation" in chunks[1]["text"]
    assert "Vector databases" in chunks[2]["text"]
