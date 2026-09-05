from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal

type EventStatus = Literal["started", "completed", "failed", "info"]
type EventValue = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    """A log-safe progress event. Details must never contain source text or paths."""

    stage: str
    label: str
    status: EventStatus
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    duration_ms: float | None = None
    details: Mapping[str, EventValue] = field(default_factory=dict)


type EventCallback = Callable[[PipelineEvent], None]

_RUN_ID_RE = re.compile(r"^[0-9a-f-]{36}$")
_STAGE_RE = re.compile(r"^[a-z0-9_.-]{1,80}$")
_PERSISTED_DETAIL_KEYS = {
    "accepted_hits",
    "accepted_queries",
    "adjacent_radius",
    "attachment_extension",
    "attachment_provided",
    "automatic",
    "candidate_count",
    "candidates",
    "chunks",
    "context_count",
    "conversation_context_chars",
    "detected_spans",
    "documents_read",
    "embedding_candidates",
    "embedding_dimension",
    "error_type",
    "extracted_cache_bypassed",
    "extracted_cache_hits",
    "extracted_cache_misses",
    "extracted_cache_write_failures",
    "extracted_cache_writes",
    "extractions",
    "extension",
    "exact_search",
    "extracted_chars",
    "fields",
    "index_hits",
    "iteration",
    "hnsw_ef",
    "marker_count",
    "ner_mode",
    "output_chars",
    "provider",
    "qdrant_mode",
    "query_normalized",
    "query_count",
    "question_chars",
    "selected",
    "expand_count",
    "size_bytes",
    "source_hashes",
    "top_k",
    "truncated",
    "retrieval_requested",
}


class JsonlEventLog:
    """Append-only, log-safe event sink for one UI run."""

    def __init__(self, root: Path, run_id: str) -> None:
        if not _RUN_ID_RE.fullmatch(run_id):
            raise ValueError("Invalid event log run ID")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = (self.root / f"{run_id}.jsonl").resolve()
        if not self.path.is_relative_to(self.root):
            raise ValueError("Event log path escaped its root")

    def __call__(self, event: PipelineEvent) -> None:
        details = {
            key: value
            for key, value in event.details.items()
            if key in _PERSISTED_DETAIL_KEYS
        }
        record = {
            "timestamp": event.timestamp.isoformat(),
            "stage": event.stage if _STAGE_RE.fullmatch(event.stage) else "invalid_stage",
            "status": event.status,
            "duration_ms": event.duration_ms,
            "details": details,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def emit_event(callback: EventCallback | None, event: PipelineEvent) -> None:
    if callback is None:
        return
    try:
        callback(event)
    except Exception:
        # Observability must not change the privacy pipeline result.
        return


@contextmanager
def timed_stage(
    callback: EventCallback | None,
    stage: str,
    label: str,
    details: Mapping[str, EventValue] | None = None,
) -> Iterator[dict[str, EventValue]]:
    safe_details = dict(details or {})
    started = perf_counter()
    emit_event(
        callback,
        PipelineEvent(stage=stage, label=label, status="started", details=safe_details),
    )
    try:
        yield safe_details
    except Exception as exc:
        emit_event(
            callback,
            PipelineEvent(
                stage=stage,
                label=label,
                status="failed",
                duration_ms=(perf_counter() - started) * 1000,
                details={**safe_details, "error_type": type(exc).__name__},
            ),
        )
        raise
    emit_event(
        callback,
        PipelineEvent(
            stage=stage,
            label=label,
            status="completed",
            duration_ms=(perf_counter() - started) * 1000,
            details=safe_details,
        ),
    )
