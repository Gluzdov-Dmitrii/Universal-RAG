from __future__ import annotations

import hashlib
import json

from .config import AppConfig


def compute_index_signature(config: AppConfig, embedding_version: str) -> str:
    payload = {
        "schema": config.schema_version,
        "extractor": "extractors-v1",
        "embedding": embedding_version,
        "chunk_chars": config.ingestion.chunk_chars,
        "chunk_overlap": config.ingestion.chunk_overlap_chars,
        "min_chunk_chars": config.ingestion.min_chunk_chars,
        "pdf_max_pages": config.ingestion.pdf_max_pages,
        "max_extracted_chars": config.ingestion.max_extracted_chars,
        "access_group": config.retrieval.access_group,
        "goz": False,
        "is_final": True,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
