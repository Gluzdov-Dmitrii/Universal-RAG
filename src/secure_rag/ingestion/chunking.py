from __future__ import annotations

import uuid

from ..domain.models import ChunkRecord, TextLocation

CHUNK_NAMESPACE = uuid.UUID("cc895e0d-5b1b-4a7f-b3b0-4c26b5915509")


def source_location_for_span(
    text_locations: tuple[TextLocation, ...],
    start: int,
    end: int,
) -> tuple[str, str | None, str | None]:
    overlapping_locations = [
        location
        for location in text_locations
        if location.start < end and location.end > start
    ]
    if not overlapping_locations:
        return "", None, None
    first = overlapping_locations[0]
    last = overlapping_locations[-1]
    if any(location.kind != first.kind for location in overlapping_locations):
        return "", None, None
    return (
        first.kind,
        first.value,
        last.value if last.value != first.value else None,
    )


def _choose_end(text: str, start: int, target: int, minimum: int) -> int:
    if target >= len(text):
        return len(text)
    lower = max(start + minimum, start + int((target - start) * 0.65))
    for separator in ("\n\n", ". ", "\n", "; ", ", ", " "):
        position = text.rfind(separator, lower, target)
        if position >= lower:
            return position + len(separator)
    return target


def chunk_text(
    text: str,
    document_id: str,
    revision: str,
    chunk_chars: int,
    overlap_chars: int,
    min_chunk_chars: int,
    access_group: str = "pilot",
    text_locations: tuple[TextLocation, ...] = (),
) -> list[ChunkRecord]:
    if chunk_chars <= 0 or overlap_chars < 0 or overlap_chars >= chunk_chars:
        raise ValueError("Invalid chunk size or overlap")
    if not text:
        return []

    chunks: list[ChunkRecord] = []
    start = 0
    ordinal = 0
    while start < len(text):
        while start < len(text) and text[start].isspace():
            start += 1
        if start >= len(text):
            break
        target = min(start + chunk_chars, len(text))
        end = _choose_end(text, start, target, min_chunk_chars)
        while end > start and text[end - 1].isspace():
            end -= 1
        if end <= start:
            break
        identity = f"{document_id}:{revision}:{ordinal}:{start}:{end}"
        chunk_id = str(uuid.uuid5(CHUNK_NAMESPACE, identity))
        location_kind, location_start, location_end = source_location_for_span(
            text_locations,
            start,
            end,
        )
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                document_id=document_id,
                revision=revision,
                ordinal=ordinal,
                start=start,
                end=end,
                text=text[start:end],
                access_group=access_group,
                location_kind=location_kind,
                location_start=location_start,
                location_end=location_end,
            )
        )
        if end >= len(text):
            break
        next_start = max(end - overlap_chars, start + 1)
        start = next_start
        ordinal += 1
    return chunks
