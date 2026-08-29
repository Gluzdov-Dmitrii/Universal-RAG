from __future__ import annotations

import re
from collections.abc import Iterable

from ..models import EntitySpan, MarkerState, SanitizedField
from .ner import SpanDetector

MARKER_RE = re.compile(r"\[\[[A-Z][A-Z0-9_]*_\d{4,}\]\]")
MARKER_LIKE_RE = re.compile(r"\[\[[^\[\]\r\n]{1,80}\]\]")
SAFE_LABEL_RE = re.compile(r"[^A-Z0-9_]+")


def merge_spans(spans: Iterable[EntitySpan]) -> list[EntitySpan]:
    candidates = sorted(
        spans,
        key=lambda item: (
            -item.priority,
            -item.score,
            -(item.end - item.start),
            item.start,
            item.end,
        ),
    )
    accepted: list[EntitySpan] = []
    for candidate in candidates:
        if any(
            candidate.start < existing.end and existing.start < candidate.end
            for existing in accepted
        ):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda item: (item.start, item.end))


class PrivacyGateway:
    def __init__(self, detector: SpanDetector) -> None:
        self.detector = detector

    @staticmethod
    def _safe_label(label: str) -> str:
        normalized = SAFE_LABEL_RE.sub("_", label.upper()).strip("_")
        return normalized or "ENTITY"

    def mark_literal(self, value: str, label: str, state: MarkerState) -> str:
        safe_label = self._safe_label(label)
        key = (safe_label, value)
        existing = state.value_to_marker.get(key)
        if existing is not None:
            return existing
        number = state.counters.get(safe_label, 0) + 1
        state.counters[safe_label] = number
        marker = f"[[{safe_label}_{number:04d}]]"
        state.value_to_marker[key] = marker
        state.marker_to_value[marker] = value
        return marker

    def sanitize_field(
        self,
        name: str,
        text: str,
        state: MarkerState,
        extra_spans: Iterable[EntitySpan] = (),
    ) -> SanitizedField:
        spans = self.detect_spans(text, extra_spans)
        return self.sanitize_with_spans(name, text, state, spans)

    def detect_spans(
        self,
        text: str,
        extra_spans: Iterable[EntitySpan] = (),
    ) -> list[EntitySpan]:
        return merge_spans([*self.detector.detect(text), *extra_spans])

    def propagate_known(
        self,
        text: str,
        known: Iterable[tuple[str, str, int]],
    ) -> list[EntitySpan]:
        spans: list[EntitySpan] = []
        seen: set[tuple[int, int, str]] = set()
        for label, value, priority in known:
            if len(value.strip()) < 3:
                continue
            for match in re.finditer(re.escape(value), text, flags=re.IGNORECASE):
                key = (match.start(), match.end(), label)
                if key in seen:
                    continue
                seen.add(key)
                spans.append(
                    EntitySpan(
                        start=match.start(),
                        end=match.end(),
                        label=label,
                        score=1.0,
                        source="known-value-propagation",
                        priority=priority,
                    )
                )
        return spans

    def sanitize_with_spans(
        self,
        name: str,
        text: str,
        state: MarkerState,
        spans: Iterable[EntitySpan],
    ) -> SanitizedField:
        spans = merge_spans(spans)
        pieces: list[str] = []
        cursor = 0
        for span in spans:
            if span.start < cursor or span.end > len(text):
                continue
            pieces.append(text[cursor : span.start])
            value = text[span.start : span.end]
            pieces.append(self.mark_literal(value, span.label, state))
            cursor = span.end
        pieces.append(text[cursor:])
        return SanitizedField(name=name, text="".join(pieces), detected_count=len(spans))

    def sanitize_filename(self, filename: str, state: MarkerState) -> str:
        return self.mark_literal(filename, "FILE", state)

    @staticmethod
    def validate_outbound(text: str, state: MarkerState) -> None:
        folded = text.casefold()
        leaked = [
            marker
            for marker, value in state.marker_to_value.items()
            if len(value.strip()) >= 3 and value.casefold() in folded
        ]
        if leaked:
            raise ValueError("Outbound validation found a known unmarked value")

    @staticmethod
    def restore(text: str, state: MarkerState, fail_on_unknown: bool = True) -> str:
        unknown = {
            marker
            for marker in MARKER_LIKE_RE.findall(text)
            if marker not in state.marker_to_value
        }
        if unknown and fail_on_unknown:
            raise ValueError("Response contains an unknown or foreign marker")
        return MARKER_RE.sub(
            lambda match: state.marker_to_value.get(match.group(0), match.group(0)),
            text,
        )
