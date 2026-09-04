from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from ..domain.models import EntitySpan

Validator = Callable[[str], bool]


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def valid_inn(value: str) -> bool:
    digits = _digits(value)
    if len(digits) == 10:
        weights = (2, 4, 10, 3, 5, 9, 4, 6, 8)
        check = (
            sum(
                int(digit) * weight
                for digit, weight in zip(digits[:-1], weights, strict=True)
            )
            % 11
            % 10
        )
        return check == int(digits[-1])
    if len(digits) == 12:
        weights_11 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
        weights_12 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
        check_11 = (
            sum(
                int(digit) * weight
                for digit, weight in zip(digits[:10], weights_11, strict=True)
            )
            % 11
            % 10
        )
        check_12 = (
            sum(
                int(digit) * weight
                for digit, weight in zip(digits[:11], weights_12, strict=True)
            )
            % 11
            % 10
        )
        return check_11 == int(digits[-2]) and check_12 == int(digits[-1])
    return False


def valid_snils(value: str) -> bool:
    digits = _digits(value)
    if len(digits) != 11:
        return False
    base = digits[:9]
    total = sum(
        int(digit) * weight
        for digit, weight in zip(base, range(9, 0, -1), strict=True)
    )
    if total < 100:
        expected = total
    elif total in (100, 101):
        expected = 0
    else:
        expected = total % 101
        if expected == 100:
            expected = 0
    return expected == int(digits[-2:])


@dataclass(frozen=True, slots=True)
class RegexRule:
    label: str
    pattern: re.Pattern[str]
    priority: int
    group: int = 0
    validator: Validator | None = None


RULES: tuple[RegexRule, ...] = (
    RegexRule(
        "RAW_MARKER",
        re.compile(r"\[\[[^\[\]\r\n]{1,80}\]\]"),
        2000,
    ),
    RegexRule(
        "EMAIL",
        re.compile(r"(?<![\w.+-])[\w.+-]+@(?:[\w-]+\.)+[\w-]{2,}(?![\w-])", re.UNICODE),
        1200,
    ),
    RegexRule(
        "SNILS",
        re.compile(r"(?<!\d)\d{3}[- ]?\d{3}[- ]?\d{3}[ ]?\d{2}(?!\d)"),
        1190,
        validator=valid_snils,
    ),
    RegexRule(
        "INN",
        re.compile(r"(?<!\d)(?:\d{12}|\d{10})(?!\d)"),
        1180,
        validator=valid_inn,
    ),
    RegexRule(
        "BANK_ACCOUNT",
        re.compile(
            r"(?i)(?:р(?:асч[её]тный)?\s*/?\s*с(?:ч[её]т)?|сч[её]т|account)"
            r"\s*(?:№|N|:)?\s*(\d{20})(?!\d)"
        ),
        1170,
        group=1,
    ),
    RegexRule(
        "BANK_ACCOUNT",
        re.compile(r"(?<!\d)\d{20}(?!\d)"),
        1135,
    ),
    RegexRule(
        "BANK_ACCOUNT",
        re.compile(r"(?<!\d)(\d{4}(?:[ -]\d{4}){4})(?!\d)"),
        1160,
        group=1,
    ),
    RegexRule(
        "PHONE",
        re.compile(
            r"(?<!\d)(?:\+7|8)[ \-]?\(?\d{3}\)?[ \-]?\d{3}"
            r"[ \-]?\d{2}[ \-]?\d{2}(?!\d)"
        ),
        1150,
    ),
    RegexRule(
        "PASSPORT",
        re.compile(r"(?<!\d)(?:\d{2}\s+\d{2}\s+\d{6}|\d{4}\s+\d{6})(?!\d)"),
        1140,
    ),
    RegexRule(
        "MONEY",
        re.compile(
            r"(?<!\w)(?:\d{1,15}|\d{1,3}(?:[ \u00a0]\d{3})+)(?:[.,]\d{1,2})?"
            r"\s*(?:₽|руб(?:л(?:ь|я|ей))?\.?)(?!\w)",
            re.IGNORECASE,
        ),
        1100,
    ),
)


class RegexDetector:
    name = "regex"

    def detect(self, text: str) -> list[EntitySpan]:
        spans: list[EntitySpan] = []
        for rule in RULES:
            for match in rule.pattern.finditer(text):
                start, end = match.span(rule.group)
                value = text[start:end]
                if rule.validator is not None and not rule.validator(value):
                    continue
                spans.append(
                    EntitySpan(
                        start=start,
                        end=end,
                        label=rule.label,
                        score=1.0,
                        source=self.name,
                        priority=rule.priority,
                    )
                )
        return spans
