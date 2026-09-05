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
    persist_extracted_text_cache: bool


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
    timeout_seconds: int
    write_max_attempts: int
    retry_backoff_seconds: float


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    top_k: int
    access_group: str
    score_threshold: float | None
    hnsw_ef: int | None = None
    exact_search: bool = False
    iterative_enabled: bool = True
    max_iterations: int = 3
    max_queries_per_iteration: int = 3
    max_contexts: int = 24
    rewrite_min_similarity: float = 0.55
    adjacent_chunk_radius: int = 1
    parent_context_enabled: bool = True
    whole_document_max_chars: int = 80_000
    logical_parent_max_chars: int = 30_000
    total_context_max_chars: int = 180_000


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
    allowed_labels: tuple[str, ...] = ()
    min_chars: int = 1


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
class GenerationConfig:
    agent_workspace_root: Path | None
    instruction_files: tuple[str, ...]
    max_instruction_chars: int


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
    generation: GenerationConfig
    bridge: BridgeConfig

    @property
    def data_path(self) -> Path:
        """Durable application data that must survive process restarts."""

        return self.paths.runtime_root / "data"

    @property
    def state_path(self) -> Path:
        """Private workflow state with an application-defined lifetime."""

        return self.paths.runtime_root / "state"

    @property
    def cache_path(self) -> Path:
        """Rebuildable local artifacts."""

        return self.paths.runtime_root / "cache"

    @property
    def diagnostics_path(self) -> Path:
        return self.paths.runtime_root / "diagnostics"

    @property
    def run_path(self) -> Path:
        """Ephemeral process coordination state."""

        return self.paths.runtime_root / "run"

    @property
    def temp_path(self) -> Path:
        return self.paths.runtime_root / "tmp"

    @property
    def manifest_path(self) -> Path:
        return self.data_path / "manifest" / "documents.sqlite"

    @property
    def qdrant_path(self) -> Path:
        return self.data_path / "qdrant"

    @property
    def chat_state_path(self) -> Path:
        return self.data_path / "sessions" / "chat-state.sqlite"

    @property
    def requests_path(self) -> Path:
        return self.state_path / "requests"

    @property
    def marker_vault_path(self) -> Path:
        return self.state_path / "marker-vault"

    @property
    def model_cache_path(self) -> Path:
        return self.cache_path / "models"

    @property
    def extracted_text_cache_path(self) -> Path:
        return self.cache_path / "extracted-text"

    @property
    def reports_path(self) -> Path:
        return self.diagnostics_path / "reports"

    @property
    def logs_path(self) -> Path:
        return self.diagnostics_path / "logs"

    @property
    def locks_path(self) -> Path:
        return self.run_path / "locks"

    @property
    def pids_path(self) -> Path:
        return self.run_path / "pids"

    @property
    def codex_sandbox_path(self) -> Path:
        return self.temp_path / "codex-sandbox"

    def ensure_runtime(self) -> None:
        for path in (
            self.manifest_path.parent,
            self.qdrant_path,
            self.chat_state_path.parent,
            self.requests_path,
            self.marker_vault_path,
            self.model_cache_path / "embeddings",
            self.model_cache_path / "ner",
            self.reports_path,
            self.logs_path,
            self.locks_path,
            self.pids_path,
            self.temp_path,
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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean environment value: {name}")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or not raw.strip() else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or not raw.strip() else float(raw)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    default_repo = Path(__file__).resolve().parents[2]
    raw_path = config_path or os.getenv("SECURE_RAG_CONFIG", "config/app.yaml")
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
    qdrant = _need(raw, "qdrant")
    retrieval = _need(raw, "retrieval")
    sanitization = _need(raw, "sanitization")
    generation = raw.get("llm", {})
    workspace_override = os.getenv("SECURE_RAG_AGENT_WORKSPACE_ROOT")
    workspace_value = workspace_override or generation.get("agent_workspace_root")

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
            persist_extracted_text_cache=_env_bool(
                "SECURE_RAG_PERSIST_EXTRACTED_CACHE",
                bool(ingestion.get("persist_extracted_text_cache", False)),
            ),
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
            mode=str(
                os.getenv("SECURE_RAG_QDRANT_MODE")
                or _need(qdrant, "mode")
            ),
            collection_name=str(_need(qdrant, "collection_name")),
            url=(
                os.getenv("SECURE_RAG_QDRANT_URL")
                or qdrant.get("url")
                or None
            ),
            timeout_seconds=_env_int(
                "SECURE_RAG_QDRANT_TIMEOUT_SECONDS",
                int(qdrant.get("timeout_seconds", 60)),
            ),
            write_max_attempts=_env_int(
                "SECURE_RAG_QDRANT_WRITE_MAX_ATTEMPTS",
                int(qdrant.get("write_max_attempts", 3)),
            ),
            retry_backoff_seconds=_env_float(
                "SECURE_RAG_QDRANT_RETRY_BACKOFF_SECONDS",
                float(qdrant.get("retry_backoff_seconds", 0.25)),
            ),
        ),
        retrieval=RetrievalConfig(
            top_k=_env_int("SECURE_RAG_RETRIEVAL_TOP_K", int(_need(retrieval, "top_k"))),
            access_group=str(_need(retrieval, "access_group")),
            score_threshold=(
                None
                if retrieval.get("score_threshold") is None
                else float(retrieval["score_threshold"])
            ),
            hnsw_ef=(
                None
                if retrieval.get("hnsw_ef") is None
                else _env_int("SECURE_RAG_RETRIEVAL_HNSW_EF", int(retrieval["hnsw_ef"]))
            ),
            exact_search=_env_bool(
                "SECURE_RAG_RETRIEVAL_EXACT",
                bool(retrieval.get("exact_search", False)),
            ),
            iterative_enabled=_env_bool(
                "SECURE_RAG_ITERATIVE_ENABLED",
                bool(retrieval.get("iterative_enabled", True)),
            ),
            max_iterations=_env_int(
                "SECURE_RAG_MAX_ITERATIONS",
                int(retrieval.get("max_iterations", 3)),
            ),
            max_queries_per_iteration=_env_int(
                "SECURE_RAG_MAX_QUERIES_PER_ITERATION",
                int(retrieval.get("max_queries_per_iteration", 3)),
            ),
            max_contexts=_env_int(
                "SECURE_RAG_MAX_CONTEXTS",
                int(retrieval.get("max_contexts", 24)),
            ),
            rewrite_min_similarity=_env_float(
                "SECURE_RAG_REWRITE_MIN_SIMILARITY",
                float(retrieval.get("rewrite_min_similarity", 0.55)),
            ),
            adjacent_chunk_radius=_env_int(
                "SECURE_RAG_ADJACENT_CHUNK_RADIUS",
                int(retrieval.get("adjacent_chunk_radius", 1)),
            ),
            parent_context_enabled=_env_bool(
                "SECURE_RAG_PARENT_CONTEXT_ENABLED",
                bool(retrieval.get("parent_context_enabled", True)),
            ),
            whole_document_max_chars=_env_int(
                "SECURE_RAG_WHOLE_DOCUMENT_MAX_CHARS",
                int(retrieval.get("whole_document_max_chars", 80_000)),
            ),
            logical_parent_max_chars=_env_int(
                "SECURE_RAG_LOGICAL_PARENT_MAX_CHARS",
                int(retrieval.get("logical_parent_max_chars", 30_000)),
            ),
            total_context_max_chars=_env_int(
                "SECURE_RAG_TOTAL_CONTEXT_MAX_CHARS",
                int(retrieval.get("total_context_max_chars", 180_000)),
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
                    allowed_labels=tuple(
                        str(label).upper().strip()
                        for label in item.get("allowed_labels", [])
                    ),
                    min_chars=int(item.get("min_chars", 1)),
                )
                for item in sanitization.get("models", [])
            ),
        ),
        generation=GenerationConfig(
            agent_workspace_root=(
                None
                if not workspace_value
                else _resolve_path(str(workspace_value), repo_root)
            ),
            instruction_files=tuple(
                str(item).replace("\\", "/").strip()
                for item in generation.get("instruction_files", [])
                if str(item).strip()
            ),
            max_instruction_chars=int(generation.get("max_instruction_chars", 20_000)),
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
    if config.qdrant.timeout_seconds <= 0:
        raise ValueError("qdrant.timeout_seconds must be positive")
    if not 1 <= config.qdrant.write_max_attempts <= 10:
        raise ValueError("qdrant.write_max_attempts must be between 1 and 10")
    if config.qdrant.retry_backoff_seconds < 0:
        raise ValueError("qdrant.retry_backoff_seconds must not be negative")
    if not 1 <= config.retrieval.top_k <= 100:
        raise ValueError("retrieval.top_k must be between 1 and 100")
    if config.retrieval.hnsw_ef is not None and config.retrieval.hnsw_ef < 1:
        raise ValueError("retrieval.hnsw_ef must be positive")
    if not 1 <= config.retrieval.max_iterations <= 5:
        raise ValueError("retrieval.max_iterations must be between 1 and 5")
    if not 1 <= config.retrieval.max_queries_per_iteration <= 5:
        raise ValueError("retrieval.max_queries_per_iteration must be between 1 and 5")
    if not config.retrieval.top_k <= config.retrieval.max_contexts <= 100:
        raise ValueError("retrieval.max_contexts must be between top_k and 100")
    if not 0.0 <= config.retrieval.rewrite_min_similarity <= 1.0:
        raise ValueError("retrieval.rewrite_min_similarity must be between 0 and 1")
    if not 0 <= config.retrieval.adjacent_chunk_radius <= 3:
        raise ValueError("retrieval.adjacent_chunk_radius must be between 0 and 3")
    if config.retrieval.whole_document_max_chars < 1_000:
        raise ValueError("retrieval.whole_document_max_chars must be at least 1000")
    if config.retrieval.logical_parent_max_chars < 1_000:
        raise ValueError("retrieval.logical_parent_max_chars must be at least 1000")
    if (
        config.retrieval.total_context_max_chars
        < config.retrieval.logical_parent_max_chars
    ):
        raise ValueError(
            "retrieval.total_context_max_chars must be at least logical_parent_max_chars"
        )
    if not 1 <= config.generation.max_instruction_chars <= 100_000:
        raise ValueError("llm.max_instruction_chars must be between 1 and 100000")
    for model in config.sanitization.models:
        if not 0.0 <= model.threshold <= 1.0:
            raise ValueError(f"NER threshold must be between 0 and 1: {model.name}")
        if model.min_chars < 1:
            raise ValueError(f"NER min_chars must be positive: {model.name}")
    source = config.paths.source_root
    runtime = config.paths.runtime_root
    if source == runtime or source.is_relative_to(runtime) or runtime.is_relative_to(source):
        raise ValueError("source_root and runtime_root must not overlap")
    return config
