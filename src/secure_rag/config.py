from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class PathsConfig:
    source_root: Path
    runtime_root: Path


@dataclass(frozen=True, slots=True)
class IngestionConfig:
    supported_extensions: tuple[str, ...]
    chunk_chars: int
    chunk_overlap_chars: int
    min_chunk_chars: int
    pdf_max_pages: int
    max_file_mb: int
    max_extracted_chars: int


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    model_id: str
    revision: str
    dimension: int
    batch_size: int
    device: str
    query_prefix: str
    passage_prefix: str


@dataclass(frozen=True, slots=True)
class QdrantConfig:
    mode: str
    collection_name: str
    url: str | None


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    top_k: int
    access_group: str
    score_threshold: float | None


@dataclass(frozen=True, slots=True)
class NerModelConfig:
    name: str
    model_id: str
    revision: str
    local_path: Path | None
    enabled: bool
    threshold: float
    device: str
    priority: int


@dataclass(frozen=True, slots=True)
class SanitizationConfig:
    marker_pattern_version: int
    fail_on_unknown_marker: bool
    regex_enabled: bool
    models: tuple[NerModelConfig, ...]


@dataclass(frozen=True, slots=True)
class BridgeConfig:
    max_output_chars: int


@dataclass(frozen=True, slots=True)
class AppConfig:
    repo_root: Path
    schema_version: int
    paths: PathsConfig
    ingestion: IngestionConfig
    embedding: EmbeddingConfig
    qdrant: QdrantConfig
    retrieval: RetrievalConfig
    sanitization: SanitizationConfig
    bridge: BridgeConfig

    @property
    def manifest_path(self) -> Path:
        return self.paths.runtime_root / "manifest" / "documents.sqlite"

    @property
    def qdrant_path(self) -> Path:
        return self.paths.runtime_root / "qdrant"

    @property
    def requests_path(self) -> Path:
        return self.paths.runtime_root / "requests"

    @property
    def marker_vault_path(self) -> Path:
        return self.paths.runtime_root / "marker-vault"

    @property
    def model_cache_path(self) -> Path:
        return self.paths.runtime_root / "model-cache"

    def ensure_runtime(self) -> None:
        for path in (
            self.manifest_path.parent,
            self.qdrant_path,
            self.requests_path,
            self.marker_vault_path,
            self.model_cache_path / "embeddings",
            self.model_cache_path / "ner",
            self.paths.runtime_root / "reports",
        ):
            path.mkdir(parents=True, exist_ok=True)


def _resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _need(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"Missing configuration key: {key}")
    return data[key]


def load_config(config_path: str | Path | None = None) -> AppConfig:
    default_repo = Path(__file__).resolve().parents[2]
    raw_path = config_path or os.getenv("SECURE_RAG_CONFIG", "config/pilot.yaml")
    path = Path(raw_path)
    if not path.is_absolute():
        path = default_repo / path
    path = path.resolve()
    repo_root = path.parent.parent

    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    paths = _need(raw, "paths")
    source_override = os.getenv("SECURE_RAG_SOURCE_ROOT")
    source_value = source_override or _need(paths, "source_root")
    runtime_value = _need(paths, "runtime_root")
    ingestion = _need(raw, "ingestion")
    embedding = _need(raw, "embedding")
    retrieval = _need(raw, "retrieval")
    sanitization = _need(raw, "sanitization")

    config = AppConfig(
        repo_root=repo_root,
        schema_version=int(_need(raw, "schema_version")),
        paths=PathsConfig(
            source_root=_resolve_path(str(source_value), repo_root),
            runtime_root=_resolve_path(str(runtime_value), repo_root),
        ),
        ingestion=IngestionConfig(
            supported_extensions=tuple(
                str(item).lower() for item in _need(ingestion, "supported_extensions")
            ),
            chunk_chars=int(_need(ingestion, "chunk_chars")),
            chunk_overlap_chars=int(_need(ingestion, "chunk_overlap_chars")),
            min_chunk_chars=int(_need(ingestion, "min_chunk_chars")),
            pdf_max_pages=int(_need(ingestion, "pdf_max_pages")),
            max_file_mb=int(_need(ingestion, "max_file_mb")),
            max_extracted_chars=int(_need(ingestion, "max_extracted_chars")),
        ),
        embedding=EmbeddingConfig(
            model_id=str(_need(embedding, "model_id")),
            revision=str(_need(embedding, "revision")),
            dimension=int(_need(embedding, "dimension")),
            batch_size=int(_need(embedding, "batch_size")),
            device=str(_need(embedding, "device")),
            query_prefix=str(embedding.get("query_prefix", "")),
            passage_prefix=str(embedding.get("passage_prefix", "")),
        ),
        qdrant=QdrantConfig(
            mode=str(_need(_need(raw, "qdrant"), "mode")),
            collection_name=str(_need(_need(raw, "qdrant"), "collection_name")),
            url=(
                os.getenv("SECURE_RAG_QDRANT_URL")
                or _need(raw, "qdrant").get("url")
                or None
            ),
        ),
        retrieval=RetrievalConfig(
            top_k=int(_need(retrieval, "top_k")),
            access_group=str(_need(retrieval, "access_group")),
            score_threshold=(
                None
                if retrieval.get("score_threshold") is None
                else float(retrieval["score_threshold"])
            ),
        ),
        sanitization=SanitizationConfig(
            marker_pattern_version=int(_need(sanitization, "marker_pattern_version")),
            fail_on_unknown_marker=bool(_need(sanitization, "fail_on_unknown_marker")),
            regex_enabled=bool(_need(sanitization, "regex_enabled")),
            models=tuple(
                NerModelConfig(
                    name=str(_need(item, "name")),
                    model_id=str(item.get("model_id", "")),
                    revision=str(item.get("revision", "")),
                    local_path=(
                        None
                        if not item.get("local_path")
                        else _resolve_path(str(item["local_path"]), repo_root)
                    ),
                    enabled=bool(item.get("enabled", False)),
                    threshold=float(item.get("threshold", 0.5)),
                    device=str(item.get("device", "auto")),
                    priority=int(item.get("priority", 0)),
                )
                for item in sanitization.get("models", [])
            ),
        ),
        bridge=BridgeConfig(
            max_output_chars=int(_need(_need(raw, "bridge"), "max_output_chars"))
        ),
    )
    if config.ingestion.chunk_overlap_chars >= config.ingestion.chunk_chars:
        raise ValueError("chunk_overlap_chars must be smaller than chunk_chars")
    if config.qdrant.mode not in {"embedded", "server"}:
        raise ValueError("qdrant.mode must be embedded or server")
    if config.qdrant.mode == "server" and not config.qdrant.url:
        raise ValueError("qdrant.url is required in server mode")
    source = config.paths.source_root
    runtime = config.paths.runtime_root
    if source == runtime or source.is_relative_to(runtime) or runtime.is_relative_to(source):
        raise ValueError("source_root and runtime_root must not overlap")
    return config
