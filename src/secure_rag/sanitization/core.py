from __future__ import annotations

import re
from collections.abc import Iterable

from ..domain.models import EntitySpan, MarkerState, SanitizedField
from .ner import SpanDetector
from .normalization import (
    canonical_marker_value,
    marker_display_value,
    marker_identity_key,
)

MARKER_RE = re.compile(r"\[\[[A-Z][A-Z0-9_]*_\d{4,}\]\]")
MARKER_LIKE_RE = re.compile(r"\[\[[^\[\]\r\n]{1,80}\]\]")
SAFE_LABEL_RE = re.compile(r"[^A-Z0-9_]+")


def _known_value_pattern(value: str) -> re.Pattern[str] | None:
    candidate = value.strip()
    if len(candidate) < 3:
        return None
    prefix = r"(?<!\w)" if candidate[0].isalpha() else ""
    suffix = r"(?!\w)" if candidate[-1].isalpha() else ""
    return re.compile(
        f"{prefix}{re.escape(candidate)}{suffix}",
        flags=re.IGNORECASE,
    )


def _contains_known_value(text: str, value: str) -> bool:
    """Match a known value without treating it as part of a larger word."""

    pattern = _known_value_pattern(value)
    return pattern is not None and pattern.search(text) is not None


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
        canonical_value = canonical_marker_value(safe_label, value)
        display_value = marker_display_value(safe_label, value)
        _, identity_value = marker_identity_key(safe_label, value)
        key = (safe_label, identity_value)
        existing = state.value_to_marker.get(key)
        if existing is not None:
            state.marker_to_aliases.setdefault(existing, set()).update(
                {value, canonical_value, display_value}
            )
            return existing
        number = state.counters.get(safe_label, 0) + 1
        state.counters[safe_label] = number
        marker = f"[[{safe_label}_{number:04d}]]"
        state.value_to_marker[key] = marker
        state.marker_to_value[marker] = display_value
        state.marker_to_aliases[marker] = {value, canonical_value, display_value}
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
            pattern = _known_value_pattern(value)
            if pattern is None:
                continue
            for match in pattern.finditer(text):
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
    def propagate_state_markers(text: str, state: MarkerState) -> str:
        """Replace aliases learned from other fields without rewriting existing markers."""

        marker_ranges = [(match.start(), match.end()) for match in MARKER_RE.finditer(text)]
        candidates: list[tuple[int, int, str]] = []
        for marker, value in state.marker_to_value.items():
            for alias in state.marker_to_aliases.get(marker, {value}):
                pattern = _known_value_pattern(alias)
                if pattern is None:
                    continue
                for match in pattern.finditer(text):
                    if any(
                        match.start() < end and start < match.end()
                        for start, end in marker_ranges
                    ):
                        continue
                    candidates.append((match.start(), match.end(), marker))
        accepted: list[tuple[int, int, str]] = []
        occupied_until = -1
        for start, end, marker in sorted(
            candidates,
            key=lambda item: (item[0], -(item[1] - item[0]), item[2]),
        ):
            if start < occupied_until:
                continue
            accepted.append((start, end, marker))
            occupied_until = end
        if not accepted:
            return text
        pieces: list[str] = []
        cursor = 0
        for start, end, marker in accepted:
            pieces.extend((text[cursor:start], marker))
            cursor = end
        pieces.append(text[cursor:])
        return "".join(pieces)

    @staticmethod
    def validate_outbound(text: str, state: MarkerState) -> None:
        # Marker counters can coincidentally equal a detected numeric value (for example
        # raw "0001" inside [[ID_0001]]). Validate only text outside known marker tokens.
        validation_text = MARKER_RE.sub("", text)
        leaked = []
        for marker, value in state.marker_to_value.items():
            candidates = state.marker_to_aliases.get(marker, {value})
            if any(
                _contains_known_value(validation_text, candidate) for candidate in candidates
            ):
                leaked.append(marker)
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
