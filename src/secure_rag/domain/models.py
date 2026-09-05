from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    document_id: str
    source_path: Path
    relative_path: str
    extension: str
    size: int
    mtime_ns: int
    revision: str
    status: str = "discovered"
    indexed_revision: str | None = None
    index_signature: str | None = None


@dataclass(frozen=True, slots=True)
class TextLocation:
    """A location-bearing interval in normalized extracted text."""

    start: int
    end: int
    kind: str
    value: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"Invalid text location: {self.start}:{self.end}")
        if not self.kind or not self.value:
            raise ValueError("Text location kind and value must not be empty")


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    revision: str
    ordinal: int
    start: int
    end: int
    text: str
    access_group: str = "employees"
    goz: bool = False
    is_final: bool = True
    location_kind: str = ""
    location_start: str | None = None
    location_end: str | None = None


@dataclass(frozen=True, slots=True)
class ChunkLocation:
    chunk_id: str
    document_id: str
    revision: str
    ordinal: int
    start: int
    end: int
    location_kind: str = ""
    location_start: str | None = None
    location_end: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    chunk_id: str
    document_id: str
    revision: str
    score: float
    start: int
    end: int
    text: str
    source_name: str
    source_path: Path | None = None
    source_type: str = ""
    ordinal: int = -1
    build_id: str = ""
    embedding_version: str = ""
    index_signature: str = ""
    location_kind: str = ""
    location_start: str | None = None
    location_end: str | None = None
    context_scope: str = "retrieved_chunk"


@dataclass(frozen=True, slots=True)
class EntitySpan:
    start: int
    end: int
    label: str
    score: float
    source: str
    priority: int = 0

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"Invalid entity span: {self.start}:{self.end}")


@dataclass(slots=True)
class MarkerState:
    marker_to_value: dict[str, str] = field(default_factory=dict)
    value_to_marker: dict[tuple[str, str], str] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    marker_to_aliases: dict[str, set[str]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SanitizedField:
    name: str
    text: str
    detected_count: int


@dataclass(frozen=True, slots=True)
class CitationLocation:
    citation_ref: str
    kind: str
    start: str
    end: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentSource:
    citation_refs: tuple[str, ...]
    document_id: str
    path: Path
    file_type: str
    best_score: float
    locations: tuple[CitationLocation, ...] = ()


@dataclass(frozen=True, slots=True)
class BridgeResult:
    request_id: str
    request_dir: Path
    codex_input: Path
    codex_output: Path
    restored_output: Path
    retrieved_count: int
    marker_count: int
    provider: str
    sources_path: Path
    sources: tuple[DocumentSource, ...] = ()
    iterations: int = 1


@dataclass(frozen=True, slots=True)
class IndexReport:
    build_id: str
    discovered: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    chunks: int = 0
    unsupported: int = 0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IndexProgress:
    """Aggregate, log-safe progress for a running index build."""

    build_id: str
    phase: str
    discovered: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    chunks: int = 0
    elapsed_seconds: float = 0.0
    files_per_second: float = 0.0
