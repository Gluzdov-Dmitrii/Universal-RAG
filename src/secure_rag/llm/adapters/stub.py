from __future__ import annotations


class StubProvider:
    """Deterministic test provider that proves the marker round trip."""

    name = "stub"

    def answer(
        self,
        sanitized_question: str,
        sanitized_contexts: list[dict[str, object]],
    ) -> str:
        lines = [
            "# Тестовый mock-ответ",
            "",
            "Этот режим не генерирует выводы. Он показывает данные, которые получил бы provider.",
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
                    (
                        f"- Источник {item['source_ref']}, type={item['file_type']}, "
                        f"score={float(item['score']):.3f}"
                    ),
                    f"  {excerpt}",
                ]
            )
        return "\n".join(lines).strip() + "\n"
