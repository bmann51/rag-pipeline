import re
from dataclasses import dataclass


@dataclass
class TextSlice:
    text: str
    page_start: int
    page_end: int


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\t\r\f]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def _paragraph_split(text: str) -> list[str]:
    pieces = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return pieces or ([text.strip()] if text.strip() else [])


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


def chunk_page_text(
    page_text: str,
    page_number: int,
    *,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
    min_chunk_chars: int,
) -> list[TextSlice]:
    cleaned = _normalize_whitespace(page_text)
    if not cleaned:
        return []

    paragraphs = _paragraph_split(cleaned)
    chunks: list[TextSlice] = []
    buffer = ""

    for para in paragraphs:
        candidate = f"{buffer}\n\n{para}".strip() if buffer else para
        if len(candidate) <= chunk_size_chars:
            buffer = candidate
            continue

        if buffer and len(buffer) >= min_chunk_chars:
            chunks.append(TextSlice(text=buffer, page_start=page_number, page_end=page_number))
            overlap = buffer[-chunk_overlap_chars:] if chunk_overlap_chars > 0 else ""
            buffer = f"{overlap} {para}".strip()
        else:
            long_text = candidate
            while len(long_text) > chunk_size_chars:
                cut = _sentence_break_index(long_text, chunk_size_chars)
                segment = long_text[:cut].strip()
                if segment:
                    chunks.append(TextSlice(text=segment, page_start=page_number, page_end=page_number))
                overlap = segment[-chunk_overlap_chars:] if chunk_overlap_chars > 0 else ""
                long_text = f"{overlap} {long_text[cut:]}".strip()
            buffer = long_text

    if buffer and (len(buffer) >= min_chunk_chars or not chunks):
        chunks.append(TextSlice(text=buffer, page_start=page_number, page_end=page_number))

    return chunks
