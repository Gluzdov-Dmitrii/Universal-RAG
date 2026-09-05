from __future__ import annotations

import pytest

from secure_rag.domain.models import EntitySpan, MarkerState
from secure_rag.sanitization.core import PrivacyGateway, merge_spans
from secure_rag.sanitization.ner import EnsembleDetector, TransformersNerDetector
from secure_rag.sanitization.normalization import canonical_person_value
from secure_rag.sanitization.regex import RegexDetector


class LiteralDetector:
    name = "literal"

    def __init__(self, values: dict[str, str]) -> None:
        self.values = values

    def detect(self, text: str) -> list[EntitySpan]:
        spans = []
        for value, label in self.values.items():
            cursor = 0
            while (position := text.find(value, cursor)) >= 0:
                spans.append(
                    EntitySpan(
                        start=position,
                        end=position + len(value),
                        label=label,
                        score=1.0,
                        source=self.name,
                        priority=500,
                    )
                )
                cursor = position + len(value)
        return spans


def test_regex_baseline_covers_identifiers_and_money() -> None:
    text = (
        "Email test@example.org, телефон +7 (913) 123-45-67, "
        "ИНН 7707083893, СНИЛС 112-233-445 95, паспорт 45 08 123456, "
        "счёт 40702810900000000001 и доход 1000000 рублей."
    )
    labels = {span.label for span in RegexDetector().detect(text)}
    assert {"EMAIL", "PHONE", "INN", "SNILS", "PASSPORT", "BANK_ACCOUNT", "MONEY"} <= labels


def test_marker_roundtrip_and_unknown_marker_block() -> None:
    raw = "Анна Смирнова пишет Анне Смирновой на test@example.org."
    detector = EnsembleDetector(
        [RegexDetector(), LiteralDetector({"Анна Смирнова": "PER", "Анне Смирновой": "PER"})]
    )
    gateway = PrivacyGateway(detector)
    state = MarkerState()
    sanitized = gateway.sanitize_field("question", raw, state).text
    assert "Анна Смирнова" not in sanitized
    assert "test@example.org" not in sanitized
    assert gateway.restore(sanitized, state) == (
        "Анна Смирнова пишет Анна Смирнова на test@example.org."
    )
    with pytest.raises(ValueError, match="unknown"):
        gateway.restore(sanitized + " [[PER_9999]]", state)
    with pytest.raises(ValueError, match="unknown"):
        gateway.restore(sanitized + " [[per_0001]]", state)


def test_original_marker_is_not_treated_as_vault_marker() -> None:
    raw = "Документ содержит строку [[PER_0001]]."
    gateway = PrivacyGateway(RegexDetector())
    state = MarkerState()
    sanitized = gateway.sanitize_field("text", raw, state).text
    assert sanitized != raw
    assert gateway.restore(sanitized, state) == raw


def test_person_forms_share_one_surname_marker() -> None:
    raw = "Ерофеева. Ерофеев Максим Владимирович. Ерофеев М.В."
    detector = LiteralDetector(
        {
            "Ерофеева": "PER",
            "Ерофеев Максим Владимирович": "PER",
            "Ерофеев М.В.": "PER",
        }
    )
    gateway = PrivacyGateway(detector)
    state = MarkerState()

    sanitized = gateway.sanitize_field("question", raw, state).text

    assert sanitized == "[[PER_0001]]. [[PER_0001]]. [[PER_0001]]"
    assert state.marker_to_value == {"[[PER_0001]]": "Ерофеев"}
    assert canonical_person_value("Ерофееву") == "Ерофеев"
    assert gateway.restore(sanitized, state) == "Ерофеев. Ерофеев. Ерофеев"


def test_outbound_validation_does_not_match_short_value_inside_word() -> None:
    gateway = PrivacyGateway(RegexDetector())
    state = MarkerState()
    gateway.mark_literal("НТИ", "ORG", state)

    gateway.validate_outbound("PER-маркеры идентифицируют человека.", state)

    with pytest.raises(ValueError, match="known unmarked value"):
        gateway.validate_outbound("Организация НТИ указана без маркера.", state)


def test_known_value_propagation_respects_word_boundaries() -> None:
    gateway = PrivacyGateway(RegexDetector())
    text = "идентифицируют НТИ2 и НТИ"

    spans = gateway.propagate_known(
        text,
        [("ORG", "НТИ", 100)],
    )

    start = text.rindex("НТИ")
    assert [(span.start, span.end) for span in spans] == [(start, start + 3)]


def test_state_aliases_are_reapplied_without_rewriting_existing_markers() -> None:
    gateway = PrivacyGateway(RegexDetector())
    state = MarkerState()
    marker = gateway.mark_literal("Компания Альфа", "ORG", state)
    state.marker_to_aliases[marker].add("Альфа")

    sanitized = gateway.propagate_state_markers(
        f"{marker} подписала договор с Альфа.",
        state,
    )

    assert sanitized == f"{marker} подписала договор с {marker}."
    gateway.validate_outbound(sanitized, state)


def test_outbound_validation_ignores_numeric_values_inside_marker_tokens() -> None:
    gateway = PrivacyGateway(RegexDetector())
    state = MarkerState()
    marker = gateway.mark_literal("0001", "ID", state)

    gateway.validate_outbound(f"Значение скрыто как {marker}.", state)

    with pytest.raises(ValueError, match="known unmarked value"):
        gateway.validate_outbound("Значение 0001 осталось открытым.", state)


def test_transformer_policy_filters_disallowed_low_score_and_short_spans() -> None:
    detector = object.__new__(TransformersNerDetector)
    detector.name = "synthetic"
    detector.threshold = 0.85
    detector.priority = 100
    detector.allowed_labels = {"PER", "ORG"}
    detector.min_chars = 3
    detector._character_windows = lambda text: [(0, len(text))]
    detector._pipeline = lambda _text: [
        {"entity_group": "POSITION", "score": 0.99, "start": 0, "end": 7},
        {"entity_group": "ORG", "score": 0.99, "start": 8, "end": 10},
        {"entity_group": "ORG", "score": 0.84, "start": 11, "end": 24},
        {"entity_group": "PER", "score": 0.93, "start": 11, "end": 24},
    ]

    spans = detector.detect("инженер AC Анна Смирнова")

    assert [(span.label, span.start, span.end) for span in spans] == [("PER", 11, 24)]


def test_higher_priority_precise_collection3_boundary_wins_overlap() -> None:
    broad_legal = EntitySpan(0, 15, "ORG", 0.91, "legal", priority=100)
    precise_collection3 = EntitySpan(8, 15, "ORG", 0.99, "collection3", priority=200)

    assert merge_spans([broad_legal, precise_collection3]) == [precise_collection3]
