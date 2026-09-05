from __future__ import annotations


class StubProvider:
    """Deterministic test provider that proves the marker round trip."""

    name = "stub"

    def answer(
        self,
        sanitized_question: str,
        sanitized_contexts: list[dict[str, object]],
    ) -> str:
        # Only echo already-validated dynamic fields. Static natural-language labels can
        # accidentally equal an entity learned from a previous turn and create a false leak.
        sections = [sanitized_question]
        for item in sanitized_contexts:
            excerpt = str(item["text"])[:500].strip()
            if excerpt:
                sections.append(excerpt)
        return "\n\n".join(sections).strip() + "\n"
