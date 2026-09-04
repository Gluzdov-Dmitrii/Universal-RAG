from __future__ import annotations

import os

from .config import AppConfig
from .retrieval.embeddings import SentenceTransformerEmbedder
from .retrieval.vector_store import QdrantStore
from .sanitization.core import PrivacyGateway
from .sanitization.ner import EnsembleDetector, TransformersNerDetector
from .sanitization.regex import RegexDetector


def create_embedder(config: AppConfig) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        config.embedding,
        config.model_cache_path / "embeddings",
    )


def create_gateway(config: AppConfig, mode: str = "all") -> PrivacyGateway:
    valid_modes = {"all", "regex", "legal", "collection3"}
    if mode not in valid_modes:
        raise ValueError(f"NER mode must be one of: {', '.join(sorted(valid_modes))}")

    detectors = []
    if config.sanitization.regex_enabled:
        detectors.append(RegexDetector())
    if mode != "regex":
        selected_names = {"legal", "collection3"} if mode == "all" else {mode}
        for model_config in config.sanitization.models:
            if not model_config.enabled or model_config.name not in selected_names:
                continue
            if model_config.name == "gliner":
                raise ValueError("GLiNER is intentionally outside the baseline environment")
            detectors.append(
                TransformersNerDetector(
                    model_config,
                    config.model_cache_path / "ner",
                )
            )
    if not detectors:
        raise ValueError("No sanitizer detectors are enabled")
    return PrivacyGateway(EnsembleDetector(detectors))


def create_vector_store(config: AppConfig, dimension: int) -> QdrantStore:
    return QdrantStore(
        config.qdrant_path,
        config.qdrant.collection_name,
        dimension,
        url=config.qdrant.url if config.qdrant.mode == "server" else None,
        api_key=os.getenv("SECURE_RAG_QDRANT_API_KEY") or None,
        timeout_seconds=config.qdrant.timeout_seconds,
        write_max_attempts=config.qdrant.write_max_attempts,
        retry_backoff_seconds=config.qdrant.retry_backoff_seconds,
    )
