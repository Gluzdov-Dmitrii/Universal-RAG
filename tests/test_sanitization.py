from __future__ import annotations

import pytest

from secure_rag.models import EntitySpan, MarkerState
from secure_rag.sanitization.core import PrivacyGateway
from secure_rag.sanitization.ner import EnsembleDetector
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
    assert gateway.restore(sanitized, state) == raw
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
