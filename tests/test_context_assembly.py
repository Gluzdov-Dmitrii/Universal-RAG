from __future__ import annotations

from secure_rag.domain.models import MarkerState, RetrievalHit
from secure_rag.orchestration.context import ContextAssembler, ProcessMemorySpanCache
from secure_rag.sanitization.core import PrivacyGateway


class BatchRecordingDetector:
    name = "batch-recording"

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def detect(self, _text: str):
        raise AssertionError("bulk detection should be used")

    def detect_many(self, texts: list[str]):
        self.calls.append(tuple(texts))
        return [[] for _ in texts]


def test_context_detection_batches_fields_and_caches_immutable_document() -> None:
    detector = BatchRecordingDetector()
    assembler = ContextAssembler(
        PrivacyGateway(detector),
        span_cache=ProcessMemorySpanCache(max_entries=4, max_total_spans=100),
    )
    hit = RetrievalHit(
        chunk_id="chunk-1",
        document_id="a" * 32,
        revision="b" * 64,
        score=0.9,
        start=0,
        end=21,
        text="Полный текст документа",
        source_name="document.txt",
        context_scope="whole_document",
    )

    assembler.sanitize("первый вопрос", [hit], MarkerState(), None)
    assembler.sanitize("второй вопрос", [hit], MarkerState(), None)

    assert detector.calls == [
        ("первый вопрос", "Полный текст документа"),
        ("второй вопрос",),
    ]
