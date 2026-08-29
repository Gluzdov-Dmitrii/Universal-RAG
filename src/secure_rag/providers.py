from __future__ import annotations

from typing import Protocol


class Provider(Protocol):
    name: str

    def answer(
        self,
        sanitized_question: str,
        sanitized_contexts: list[dict[str, str | float]],
    ) -> str: ...


class StubProvider:
    """Deterministic stand-in that proves the marker round trip without a cloud call."""

    name = "stub"

    def answer(
        self,
        sanitized_question: str,
        sanitized_contexts: list[dict[str, str | float]],
    ) -> str:
        lines = [
            "# Локальный mock-ответ",
            "",
            "Этот режим не генерирует выводы. Он показывает данные, которые получил бы Codex.",
            "",
            f"Запрос: {sanitized_question}",
            "",
            "Найденный контекст:",
        ]
        if not sanitized_contexts:
            lines.append("- Контекст не найден.")
        for item in sanitized_contexts:
            excerpt = str(item["text"])[:500].strip()
            lines.extend(
                [
                    "",
                    f"- Источник {item['source_ref']}, score={float(item['score']):.3f}",
                    f"  {excerpt}",
                ]
            )
        return "\n".join(lines).strip() + "\n"
