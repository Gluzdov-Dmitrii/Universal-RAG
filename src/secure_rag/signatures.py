from __future__ import annotations

import hashlib
import json

from .config import AppConfig
from .extractors import EXTRACTOR_VERSION


def compute_index_signature(config: AppConfig, embedding_version: str) -> str:
    qdrant_target = {
        "mode": config.qdrant.mode,
        "url": config.qdrant.url if config.qdrant.mode == "server" else None,
        "collection": config.qdrant.collection_name,
    }
    payload = {
        "schema": config.schema_version,
        "extractor": EXTRACTOR_VERSION,
        "supported_extensions": sorted(config.ingestion.supported_extensions),
        "embedding": embedding_version,
        "embedding_policy": {
            "dimension": config.embedding.dimension,
            "normalize_embeddings": True,
            "query_prefix": config.embedding.query_prefix,
            "passage_prefix": config.embedding.passage_prefix,
        },
        "chunk_chars": config.ingestion.chunk_chars,
        "chunk_overlap": config.ingestion.chunk_overlap_chars,
        "min_chunk_chars": config.ingestion.min_chunk_chars,
        "pdf_max_pages": config.ingestion.pdf_max_pages,
        "max_extracted_chars": config.ingestion.max_extracted_chars,
        "access_group": config.retrieval.access_group,
        "goz": False,
        "is_final": True,
        # Manifest is shared by the pilot backends. This target identity forces a
        # re-index when moving from embedded storage to another Qdrant server.
        "qdrant_target": qdrant_target,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
