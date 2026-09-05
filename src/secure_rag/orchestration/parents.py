from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

from ..config import AppConfig
from ..domain.models import RetrievalHit, TextLocation
from ..ingestion.extractors import (
    ExtractedDocument,
    ExtractionError,
    extract_document,
    file_sha256,
)
from .events import EventCallback, timed_stage

_TABLE_TYPES = frozenset({"csv", "xls", "xlsx"})
_SECTION_SEPARATOR = "\n\n[... другая логическая часть документа ...]\n\n"
_TABLE_SEPARATOR = "\n\n[... строки между заголовком и найденным фрагментом опущены ...]\n\n"
_TABLE_HEADER_CHARS = 3_000


class ParentContextBuilder:
    """Turn retrieval chunks into bounded, source-verified document context."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._cache: dict[tuple[Path, str], ExtractedDocument | None] = {}

    def build(
        self,
        hits: list[RetrievalHit],
        *,
        on_event: EventCallback | None = None,
    ) -> list[RetrievalHit]:
        if not self.config.retrieval.parent_context_enabled:
            return hits

        groups: OrderedDict[tuple[str, str], list[RetrievalHit]] = OrderedDict()
        for hit in hits:
            groups.setdefault((hit.document_id, hit.revision), []).append(hit)

        result: list[RetrievalHit] = []
        remaining = self.config.retrieval.total_context_max_chars
        with timed_stage(
            on_event,
            "retrieval.parent_context",
            "Сборка полноразмерного контекста документов",
            {"anchor_chunks": len(hits), "documents": len(groups)},
        ) as details:
            whole_documents = 0
            logical_parents = 0
            fallback_chunks = 0
            for group in groups.values():
                if remaining <= 0:
                    break
                best = group[0]
                extracted = self._load_verified(best)
                if extracted is None or not extracted.text:
                    for hit in group:
                        if remaining <= 0:
                            break
                        text = hit.text[:remaining]
                        if not text:
                            continue
                        result.append(replace(hit, text=text, context_scope="retrieved_chunk"))
                        remaining -= len(text)
                        fallback_chunks += 1
                    continue

                if (
                    not extracted.truncated
                    and len(extracted.text)
                    <= self.config.retrieval.whole_document_max_chars
                    and len(extracted.text) <= remaining
                ):
                    result.append(
                        replace(
                            best,
                            start=0,
                            end=len(extracted.text),
                            text=extracted.text,
                            context_scope="whole_document",
                        )
                    )
                    remaining -= len(extracted.text)
                    whole_documents += 1
                    continue

                budget = min(
                    self.config.retrieval.logical_parent_max_chars,
                    remaining,
                )
                text, start, end = self._logical_parent(extracted, group, budget)
                if not text:
                    continue
                result.append(
                    replace(
                        best,
                        start=start,
                        end=end,
                        text=text,
                        context_scope="logical_parent",
                    )
                )
                remaining -= len(text)
                logical_parents += 1

            details["contexts"] = len(result)
            details["whole_documents"] = whole_documents
            details["logical_parents"] = logical_parents
            details["fallback_chunks"] = fallback_chunks
            details["context_chars"] = sum(len(hit.text) for hit in result)
        return result

    def _load_verified(self, hit: RetrievalHit) -> ExtractedDocument | None:
        if hit.source_path is None:
            return None
        try:
            path = hit.source_path.resolve(strict=True)
            if not path.is_relative_to(self.config.paths.source_root):
                return None
            key = (path, hit.revision)
            if key in self._cache:
                return self._cache[key]
            before = path.stat()
            if file_sha256(path) != hit.revision:
                self._cache[key] = None
                return None
            extracted = extract_document(path, self.config.ingestion)
            after = path.stat()
            if (
                after.st_size != before.st_size
                or after.st_mtime_ns != before.st_mtime_ns
                or file_sha256(path) != hit.revision
            ):
                self._cache[key] = None
                return None
            self._cache[key] = extracted
            return extracted
        except (OSError, ExtractionError):
            return None

    def _logical_parent(
        self,
        extracted: ExtractedDocument,
        hits: list[RetrievalHit],
        budget: int,
    ) -> tuple[str, int, int]:
        text = extracted.text
        best = hits[0]
        relevant = self._relevant_locations(extracted.locations, hits)
        if relevant:
            selected: list[TextLocation] = []
            used = 0
            for location in relevant:
                separator_chars = len(_SECTION_SEPARATOR) if selected else 0
                location_chars = location.end - location.start
                if used + separator_chars + location_chars > budget:
                    continue
                selected.append(location)
                used += separator_chars + location_chars
            if selected:
                return (
                    _SECTION_SEPARATOR.join(text[item.start : item.end] for item in selected),
                    min(item.start for item in selected),
                    max(item.end for item in selected),
                )
            bounds_start, bounds_end = relevant[0].start, relevant[0].end
        else:
            bounds_start, bounds_end = 0, len(text)

        if best.source_type.lower().lstrip(".") in _TABLE_TYPES:
            return self._table_excerpt(
                text,
                bounds_start,
                bounds_end,
                best.start,
                best.end,
                budget,
            )
        return self._window(
            text,
            bounds_start,
            bounds_end,
            min(hit.start for hit in hits),
            max(hit.end for hit in hits),
            budget,
        )

    @staticmethod
    def _relevant_locations(
        locations: tuple[TextLocation, ...],
        hits: list[RetrievalHit],
    ) -> list[TextLocation]:
        result: list[TextLocation] = []
        seen: set[tuple[int, int]] = set()
        for hit in hits:
            for location in locations:
                if location.end <= hit.start or location.start >= hit.end:
                    continue
                key = (location.start, location.end)
                if key not in seen:
                    seen.add(key)
                    result.append(location)
        return result

    @classmethod
    def _table_excerpt(
        cls,
        text: str,
        bounds_start: int,
        bounds_end: int,
        focus_start: int,
        focus_end: int,
        budget: int,
    ) -> tuple[str, int, int]:
        if bounds_end - bounds_start <= budget:
            return text[bounds_start:bounds_end], bounds_start, bounds_end
        header_chars = min(_TABLE_HEADER_CHARS, max(1, budget // 3))
        header_end = min(bounds_end, bounds_start + header_chars)
        if focus_start <= header_end:
            end = min(bounds_end, bounds_start + budget)
            return text[bounds_start:end], bounds_start, end
        match_budget = max(1, budget - header_chars - len(_TABLE_SEPARATOR))
        match, match_start, match_end = cls._window(
            text,
            bounds_start,
            bounds_end,
            focus_start,
            focus_end,
            match_budget,
        )
        combined = text[bounds_start:header_end] + _TABLE_SEPARATOR + match
        return combined[:budget], bounds_start, match_end

    @staticmethod
    def _window(
        text: str,
        bounds_start: int,
        bounds_end: int,
        focus_start: int,
        focus_end: int,
        budget: int,
    ) -> tuple[str, int, int]:
        bounds_start = max(0, bounds_start)
        bounds_end = min(len(text), bounds_end)
        focus_start = min(max(focus_start, bounds_start), bounds_end)
        focus_end = min(max(focus_end, focus_start), bounds_end)
        if bounds_end - bounds_start <= budget:
            return text[bounds_start:bounds_end], bounds_start, bounds_end
        focus_chars = focus_end - focus_start
        if focus_chars >= budget:
            end = min(bounds_end, focus_start + budget)
            return text[focus_start:end], focus_start, end
        remaining = budget - focus_chars
        before = min(focus_start - bounds_start, remaining // 2)
        after = min(bounds_end - focus_end, remaining - before)
        before = min(focus_start - bounds_start, remaining - after)
        start = focus_start - before
        end = focus_end + after
        return text[start:end], start, end
