from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import AppConfig
from ..domain.models import RetrievalHit
from ..ingestion.chunking import chunk_text
from ..ingestion.extractors import extract_document, file_sha256, stable_document_id
from ..orchestration.events import EventCallback, timed_stage
from .embeddings import Embedder


def resolve_attachment_path(config: AppConfig, value: str | Path) -> Path:
    raw = str(value).strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
        raw = raw[1:-1].strip()
    if not raw:
        raise ValueError("Attachment path is empty")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = config.paths.source_root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError("Attachment file does not exist or is unavailable") from exc
    if not resolved.is_relative_to(config.paths.source_root):
        raise ValueError("Attachment must be inside the configured source root")
    if not resolved.is_file():
        raise ValueError("Attachment must be a regular file")
    if resolved.suffix.lower() not in config.ingestion.supported_extensions:
        raise ValueError("Attachment file type is not supported")
    return resolved


def attachment_hits(
    config: AppConfig,
    embedder: Embedder,
    question: str,
    path: Path,
    *,
    top_k: int,
    on_event: EventCallback | None = None,
) -> list[RetrievalHit]:
    with timed_stage(
        on_event,
        "attachment.read",
        "Чтение и проверка локального файла",
        {"extension": path.suffix.lower(), "size_bytes": path.stat().st_size},
    ) as details:
        before = path.stat()
        extracted = extract_document(path, config.ingestion)
        revision = file_sha256(path)
        after = path.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise ValueError("Attachment changed while it was being read")
        relative = path.relative_to(config.paths.source_root).as_posix()
        document_id = stable_document_id(relative)
        details["extracted_chars"] = len(extracted.text)
        details["truncated"] = extracted.truncated

    with timed_stage(
        on_event,
        "attachment.chunk",
        "Разбиение локального файла на фрагменты",
    ) as details:
        chunks = chunk_text(
            extracted.text,
            document_id,
            revision,
            config.ingestion.chunk_chars,
            config.ingestion.chunk_overlap_chars,
            config.ingestion.min_chunk_chars,
            access_group=config.retrieval.access_group,
        )
        details["chunks"] = len(chunks)
        details["embedding_candidates"] = len(chunks)

    if not chunks:
        return []
    with timed_stage(
        on_event,
        "attachment.embedding",
        "Токенизация и embedding фрагментов локального файла",
        {"candidate_count": len(chunks)},
    ) as details:
        query_vector = embedder.embed_query(question)
        vectors = embedder.embed_passages([chunk.text for chunk in chunks])
        scores = vectors @ query_vector
        selected_indexes = np.argsort(scores)[::-1][:top_k]
        details["selected"] = len(selected_indexes)

    return [
        RetrievalHit(
            chunk_id=chunks[int(index)].chunk_id,
            document_id=document_id,
            revision=revision,
            score=float(scores[int(index)]),
            start=chunks[int(index)].start,
            end=chunks[int(index)].end,
            text=chunks[int(index)].text,
            source_name=path.name,
            source_path=path,
            source_type=path.suffix.lower().lstrip("."),
            ordinal=chunks[int(index)].ordinal,
            build_id="direct-attachment",
            embedding_version=embedder.model_version,
            index_signature="direct-attachment",
        )
        for index in selected_indexes
    ]
