from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from pathlib import Path
from threading import RLock

from ..domain.models import (
    CitationLocation,
    DocumentSource,
    EntitySpan,
    MarkerState,
    RetrievalHit,
)
from ..sanitization.core import PrivacyGateway, merge_spans
from ..sanitization.normalization import canonical_marker_value
from .events import EventCallback, timed_stage

_FILE_TYPE_RE = re.compile(r"^[a-z0-9]{1,10}$")


class ProcessMemorySpanCache:
    """Bounded text-free cache of NER coordinates for immutable document revisions."""

    def __init__(self, *, max_entries: int = 64, max_total_spans: int = 250_000) -> None:
        if max_entries < 1 or max_total_spans < 1:
            raise ValueError("span_cache_limits_must_be_positive")
        self.max_entries = max_entries
        self.max_total_spans = max_total_spans
        self._entries: OrderedDict[tuple[str, ...], tuple[EntitySpan, ...]] = OrderedDict()
        self._total_spans = 0
        self._lock = RLock()

    @staticmethod
    def _key(hit: RetrievalHit) -> tuple[str, ...]:
        digest = hashlib.sha256(hit.text.encode("utf-8")).hexdigest()
        return (
            hit.document_id,
            hit.revision,
            hit.context_scope,
            str(hit.start),
            str(hit.end),
            digest,
        )

    def load(self, hit: RetrievalHit) -> tuple[EntitySpan, ...] | None:
        key = self._key(hit)
        with self._lock:
            spans = self._entries.get(key)
            if spans is not None:
                self._entries.move_to_end(key)
            return spans

    def store(self, hit: RetrievalHit, spans: list[EntitySpan]) -> None:
        if len(spans) > self.max_total_spans:
            return
        key = self._key(hit)
        value = tuple(spans)
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._total_spans -= len(previous)
            while self._entries and (
                len(self._entries) >= self.max_entries
                or self._total_spans + len(value) > self.max_total_spans
            ):
                _, evicted = self._entries.popitem(last=False)
                self._total_spans -= len(evicted)
            self._entries[key] = value
            self._total_spans += len(value)


class ContextAssembler:
    """Build one marker namespace and local provenance across retrieval iterations."""

    def __init__(
        self,
        gateway: PrivacyGateway,
        span_cache: ProcessMemorySpanCache | None = None,
    ) -> None:
        self.gateway = gateway
        self.span_cache = span_cache

    def sanitize(
        self,
        question: str,
        hits: list[RetrievalHit],
        state: MarkerState,
        on_event: EventCallback | None,
    ) -> tuple[str, list[dict[str, object]]]:
        raw_fields = [question, *[hit.text for hit in hits]]
        with timed_stage(
            on_event,
            "sanitizer.detect",
            "NER и regex: поиск чувствительных сущностей",
            {"fields": len(raw_fields)},
        ) as details:
            detected: list[list[EntitySpan] | None] = [None] * len(raw_fields)
            missing_indices = [0]
            missing_texts = [question]
            cache_hits = 0
            cache_misses = 0
            for index, hit in enumerate(hits, start=1):
                cached = self.span_cache.load(hit) if self.span_cache is not None else None
                if cached is None:
                    cache_misses += 1
                    missing_indices.append(index)
                    missing_texts.append(hit.text)
                else:
                    cache_hits += 1
                    detected[index] = list(cached)
            for index, spans in zip(
                missing_indices,
                self.gateway.detect_many(missing_texts),
                strict=True,
            ):
                detected[index] = spans
                if index > 0 and self.span_cache is not None:
                    self.span_cache.store(hits[index - 1], spans)
            resolved_detected = [spans if spans is not None else [] for spans in detected]
            detected = resolved_detected
            details["detected_spans"] = sum(len(spans) for spans in detected)
            details["span_cache_hits"] = cache_hits
            details["span_cache_misses"] = cache_misses

        with timed_stage(
            on_event,
            "sanitizer.mark",
            "Распространение сущностей и установка маркеров",
        ) as details:
            known: list[tuple[str, str, int]] = []
            for text, spans in zip(raw_fields, detected, strict=True):
                for span in spans:
                    value = text[span.start : span.end]
                    known.append((span.label, value, span.priority))
                    canonical = canonical_marker_value(span.label, value)
                    if canonical.casefold() != value.strip().casefold():
                        known.append((span.label, canonical, -1))
            completed_spans = [
                merge_spans([*spans, *self.gateway.propagate_known(text, known)])
                for text, spans in zip(raw_fields, detected, strict=True)
            ]
            sanitized_question = self.gateway.sanitize_with_spans(
                "question", question, state, completed_spans[0]
            ).text
            contexts: list[dict[str, object]] = []
            for index, hit in enumerate(hits, start=1):
                sanitized_text = self.gateway.sanitize_with_spans(
                    f"chunk:{hit.chunk_id}",
                    hit.text,
                    state,
                    completed_spans[index],
                ).text
                file_type = hit.source_type.lower().lstrip(".")
                if not _FILE_TYPE_RE.fullmatch(file_type):
                    file_type = "unknown"
                contexts.append(
                    {
                        "document_id": hit.document_id,
                        "chunk_id": hit.chunk_id,
                        "citation_ref": f"R{index:03d}",
                        "score": hit.score,
                        "file_type": file_type,
                        "context_scope": hit.context_scope,
                        "text": sanitized_text,
                    }
                )
            # A later field can teach the shared marker state a normalized alias that was
            # absent when an earlier field was processed. Re-apply all learned aliases so
            # multi-turn history cannot retain a known raw variant.
            sanitized_question = self.gateway.propagate_state_markers(
                sanitized_question,
                state,
            )
            for context in contexts:
                context["text"] = self.gateway.propagate_state_markers(
                    str(context["text"]),
                    state,
                )
            details["marker_count"] = len(state.marker_to_value)
        return sanitized_question, contexts

    @staticmethod
    def sources(hits: list[RetrievalHit]) -> tuple[DocumentSource, ...]:
        grouped: dict[tuple[str, Path], dict[str, object]] = {}
        for index, hit in enumerate(hits, start=1):
            if hit.source_path is None:
                continue
            path = hit.source_path.resolve()
            key = (hit.document_id, path)
            item = grouped.setdefault(
                key,
                {
                    "refs": [],
                    "locations": [],
                    "file_type": hit.source_type.lower().lstrip(".") or "unknown",
                    "score": hit.score,
                },
            )
            refs = item["refs"]
            assert isinstance(refs, list)
            refs.append(f"R{index:03d}")
            if hit.location_kind and hit.location_start is not None:
                locations = item["locations"]
                assert isinstance(locations, list)
                locations.append(
                    CitationLocation(
                        citation_ref=f"R{index:03d}",
                        kind=hit.location_kind,
                        start=hit.location_start,
                        end=hit.location_end,
                    )
                )
            item["score"] = max(float(item["score"]), hit.score)
        return tuple(
            DocumentSource(
                citation_refs=tuple(item["refs"]),
                document_id=document_id,
                path=path,
                file_type=str(item["file_type"]),
                best_score=float(item["score"]),
                locations=tuple(item["locations"]),
            )
            for (document_id, path), item in grouped.items()
        )
