import re
from dataclasses import dataclass
from typing import Protocol


class _PageLike(Protocol):
    page_number: int
    text: str


@dataclass
class TextSlice:
    text: str
    page_start: int
    page_end: int


@dataclass
class _PageSpan:
    page_number: int
    start: int  # inclusive char index in concatenated document string
    end: int    # exclusive char index


def _normalize_whitespace(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"[\t\r\f]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def _sentence_break_index(text: str, max_len: int) -> int:
    if len(text) <= max_len:
        return len(text)
    window = text[: max_len + 1]
    punctuation = [window.rfind(ch) for ch in ".!?\n"]
    cut = max(punctuation)
    if cut >= int(max_len * 0.6):
        return cut + 1
    space_cut = window.rfind(" ")
    if space_cut >= int(max_len * 0.6):
        return space_cut + 1
    return max_len


def _resolve_pages(
    chunk_start: int, chunk_end: int, spans: list[_PageSpan]
) -> tuple[int, int]:
    page_start = page_end = None
    for span in spans:
        if span.start < chunk_end and span.end > chunk_start:
            if page_start is None:
                page_start = span.page_number
            page_end = span.page_number
    return (page_start or 1), (page_end or 1)


def chunk_document(
    pages: list[_PageLike],
    *,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
    min_chunk_chars: int,
) -> list[TextSlice]:
    # Phase 1: concatenate all pages into one string, tracking which character
    # ranges came from which page.
    parts: list[str] = []
    spans: list[_PageSpan] = []
    offset = 0
    prev_normalized: str | None = None
    for page in pages:
        normalized = _normalize_whitespace(page.text)
        if not normalized:
            continue
        if prev_normalized is not None:
            sep = "\n\n" if prev_normalized[-1] in ".!?" else " "
            parts.append(sep)
            offset += len(sep)
        spans.append(_PageSpan(page_number=page.page_number, start=offset, end=offset + len(normalized)))
        parts.append(normalized)
        offset += len(normalized)
        prev_normalized = normalized

    full_text = "".join(parts)
    if not full_text:
        return []

    # Phase 2: split into paragraphs, recording each paragraph's position
    # in full_text so we can map any chunk range back to page numbers.
    paras: list[tuple[str, int, int]] = []  # (text, start, end)
    scan = 0
    for raw in re.split(r"\n\s*\n", full_text):
        stripped = raw.strip()
        if not stripped:
            continue
        idx = full_text.find(stripped, scan)
        paras.append((stripped, idx, idx + len(stripped)))
        scan = idx + len(stripped)

    if not paras:
        return []

    # Phase 3: paragraph-accumulation chunker, tracking document offsets.
    chunks: list[TextSlice] = []
    buffer = ""
    buf_doc_start = 0
    buf_doc_end = 0

    def emit(text: str, doc_start: int, doc_end: int) -> None:
        ps, pe = _resolve_pages(doc_start, doc_end, spans)
        chunks.append(TextSlice(text=text, page_start=ps, page_end=pe))

    for para_text, para_start, para_end in paras:
        candidate = f"{buffer}\n\n{para_text}".strip() if buffer else para_text

        if len(candidate) <= chunk_size_chars:
            if not buffer:
                buf_doc_start = para_start
            buffer = candidate
            buf_doc_end = para_end
            continue

        if buffer and len(buffer) >= min_chunk_chars:
            emit(buffer, buf_doc_start, buf_doc_end)
            overlap = buffer[-chunk_overlap_chars:] if chunk_overlap_chars > 0 else ""
            overlap_doc_start = max(buf_doc_start, buf_doc_end - chunk_overlap_chars)
            buffer = f"{overlap} {para_text}".strip() if overlap else para_text
            buf_doc_start = overlap_doc_start if overlap else para_start
            buf_doc_end = para_end
        else:
            # Long paragraph (or tiny buffer + large para) that overflows on its own.
            long_text = candidate
            long_doc_start = buf_doc_start if buffer else para_start
            long_doc_end = para_end
            while len(long_text) > chunk_size_chars:
                cut = _sentence_break_index(long_text, chunk_size_chars)
                segment = long_text[:cut].strip()
                # Approximate the doc-end for this segment proportionally.
                ratio = cut / max(len(long_text), 1)
                seg_doc_end = long_doc_start + int((long_doc_end - long_doc_start) * ratio)
                if segment:
                    emit(segment, long_doc_start, seg_doc_end)
                overlap = segment[-chunk_overlap_chars:] if (chunk_overlap_chars > 0 and segment) else ""
                long_doc_start = max(long_doc_start, seg_doc_end - chunk_overlap_chars)
                long_text = f"{overlap} {long_text[cut:]}".strip() if overlap else long_text[cut:].strip()
            buffer = long_text
            buf_doc_start = long_doc_start
            buf_doc_end = long_doc_end

    if buffer and (len(buffer) >= min_chunk_chars or not chunks):
        emit(buffer, buf_doc_start, buf_doc_end)

    return chunks
