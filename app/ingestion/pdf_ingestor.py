from dataclasses import dataclass
from io import BytesIO
import time
from typing import Iterable

from pypdf import PdfReader

from app.ingestion.chunker import TextSlice, chunk_document


def _is_retryable_ocr_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "status 502" in message
        or "status 503" in message
        or "status 504" in message
        or "timeout" in message
        or "temporar" in message
    )


@dataclass
class PageText:
    page_number: int
    text: str


def extract_pdf_pages(
    file_bytes: bytes,
    *,
    ocr_fallback_enabled: bool = False,
    mistral_api_key: str | None = None,
    mistral_ocr_model: str = "mistral-ocr-latest",
) -> list[PageText]:
    reader = PdfReader(BytesIO(file_bytes))

    if reader.is_encrypted:
        decrypt_result = reader.decrypt("")
        if decrypt_result == 0:
            raise ValueError("Encrypted PDF is not supported without a password.")

    pages: list[PageText] = []
    for idx, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        pages.append(PageText(page_number=idx, text=extracted))

    if not pages:
        raise ValueError("No pages were found in the PDF.")

    has_text = any(page.text.strip() for page in pages)
    if has_text:
        return pages

    if not ocr_fallback_enabled:
        return pages

    return extract_pdf_pages_with_ocr(
        file_bytes,
        mistral_api_key=mistral_api_key,
        mistral_ocr_model=mistral_ocr_model,
    )


def extract_pdf_pages_with_ocr(
    file_bytes: bytes,
    *,
    mistral_api_key: str | None,
    mistral_ocr_model: str,
) -> list[PageText]:
    if not mistral_api_key:
        raise ValueError(
            "The PDF does not contain extractable text and OCR fallback is unavailable. "
            "Set MISTRAL_API_KEY to enable OCR fallback."
        )

    try:
        from mistralai import Mistral
    except ImportError as exc:
        raise ValueError(
            "The PDF does not contain extractable text and OCR fallback is unavailable. "
            "Install the mistralai package to enable OCR fallback."
        ) from exc

    client = Mistral(api_key=mistral_api_key)
    uploaded_file_id: str | None = None

    try:
        uploaded_file = client.files.upload(
            file={
                "file_name": "ingestion.pdf",
                "content": file_bytes,
            },
            purpose="ocr",
        )
        uploaded_file_id = uploaded_file.id

        ocr_response = None
        for attempt in range(3):
            try:
                ocr_response = client.ocr.process(
                    document={
                        "type": "file",
                        "file_id": uploaded_file_id,
                    },
                    model=mistral_ocr_model,
                    include_image_base64=False,
                )
                break
            except Exception as exc:
                if attempt == 2 or not _is_retryable_ocr_error(exc):
                    raise
                time.sleep(2**attempt)

        if ocr_response is None:
            raise ValueError("OCR did not return a response.")

        pages: list[PageText] = []
        for page in ocr_response.pages:
            pages.append(PageText(page_number=page.index + 1, text=page.markdown or ""))

        if not pages or not any(page.text.strip() for page in pages):
            raise ValueError("OCR did not return extractable text for this PDF.")

        pages.sort(key=lambda item: item.page_number)
        return pages
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"OCR fallback failed: {exc}") from exc
    finally:
        if uploaded_file_id:
            try:
                client.files.delete(file_id=uploaded_file_id)
            except Exception:
                pass


def build_chunks(
    pages: Iterable[PageText],
    *,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
    min_chunk_chars: int,
) -> list[TextSlice]:
    all_chunks = chunk_document(
        list(pages),
        chunk_size_chars=chunk_size_chars,
        chunk_overlap_chars=chunk_overlap_chars,
        min_chunk_chars=min_chunk_chars,
    )

    if not all_chunks:
        raise ValueError("The PDF does not contain extractable text.")

    return all_chunks
