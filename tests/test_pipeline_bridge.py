from __future__ import annotations

from dataclasses import replace

from secure_rag.config import load_config
from secure_rag.models import EntitySpan, RetrievalHit
from secure_rag.pipeline import SecureRagPipeline
from secure_rag.sanitization.core import PrivacyGateway
from secure_rag.sanitization.ner import EnsembleDetector
from secure_rag.sanitization.regex import RegexDetector


class LiteralDetector:
    name = "literal"

    def detect(self, text: str) -> list[EntitySpan]:
        values = {"Анна Смирнова": "PER", "СибНИА": "ORG"}
        result = []
        for value, label in values.items():
            start = text.find(value)
            if start >= 0:
                result.append(
                    EntitySpan(
                        start=start,
                        end=start + len(value),
                        label=label,
                        score=1.0,
                        source=self.name,
                        priority=500,
                    )
                )
        return result


class FakeEmbedder:
    model_version = "fake@1"


class FakeRetriever:
    embedder = FakeEmbedder()

    def search(self, _query: str, top_k: int | None = None) -> list[RetrievalHit]:
        return [
            RetrievalHit(
                chunk_id="00000000-0000-0000-0000-000000000001",
                document_id="opaque-document",
                revision="revision",
                score=0.91,
                start=0,
                end=120,
                text=(
                    "Анна Смирнова работает в СибНИА. Email test@example.org. "
                    "Доход 1000000 рублей."
                ),
                source_name="Личное дело Анны.docx",
            )
        ]


class FakeManifest:
    def latest_build_id(self) -> str:
        return "build-test"


def test_full_stub_flow_keeps_raw_values_out_of_codex_file(tmp_path) -> None:
    base = load_config()
    config = replace(
        base,
        paths=replace(base.paths, runtime_root=(tmp_path / "runtime").resolve()),
    )
    config.ensure_runtime()
    gateway = PrivacyGateway(EnsembleDetector([RegexDetector(), LiteralDetector()]))
    pipeline = SecureRagPipeline(config, FakeRetriever(), gateway, FakeManifest())
    result = pipeline.run("Что известно про анна смирнова?", provider="stub")

    outbound = result.codex_input.read_text(encoding="utf-8")
    restored = result.restored_output.read_text(encoding="utf-8")
    assert "анна смирнова" not in outbound.casefold()
    assert "СибНИА" not in outbound
    assert "test@example.org" not in outbound
    assert "1000000 рублей" not in outbound
    assert "Личное дело Анны.docx" not in outbound
    assert "[[PER_0001]]" in outbound
    assert "Анна Смирнова" in restored
    assert "test@example.org" in restored
