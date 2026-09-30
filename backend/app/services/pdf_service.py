import logging
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PDFProcessingError(Exception):
    """Custom exception raised when PDF processing fails."""
    pass


def extract_pdf_text(file_bytes: bytes) -> dict:
    """
    Extract text page-by-page from a PDF byte stream using PyMuPDF.

    :param file_bytes: Raw bytes of the uploaded PDF file.
    :return: Dictionary containing total_pages and a list of page dicts.
    :raises PDFProcessingError: If the file cannot be opened or parsed as a PDF.
    """
    if not file_bytes:
        logger.warning("PDF extraction failed: file bytes are empty.")
        raise PDFProcessingError("Uploaded file is empty.")

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        logger.error(f"PyMuPDF failed to parse PDF stream: {str(e)}")
        raise PDFProcessingError(f"Failed to parse PDF document: {str(e)}")

    if doc.is_encrypted:
        try:
            # Try to authenticate with empty password (for standard restricted PDFs)
            doc.authenticate("")
        except Exception:
            logger.warning("PDF is encrypted/password-protected and could not be unlocked.")
            raise PDFProcessingError("PDF is password-protected and cannot be extracted.")

    total_pages = doc.page_count
    if total_pages == 0:
        logger.warning("Parsed PDF contains 0 pages.")
        raise PDFProcessingError("PDF document contains no pages.")

    logger.info(f"Extracting text from PDF with {total_pages} page(s)...")
    pages_data = []
    for page_index in range(total_pages):
        page = doc.load_page(page_index)
        text = page.get_text()
        pages_data.append({
            "page_number": page_index + 1,
            "text": text.strip()
        })

    doc.close()
    logger.info(f"Successfully extracted text from all {total_pages} page(s).")

    return {
        "total_pages": total_pages,
        "pages": pages_data
    }
