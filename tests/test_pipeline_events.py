from __future__ import annotations

import json
import uuid
from dataclasses import replace

import pytest

from secure_rag.config import load_config
from secure_rag.embeddings import HashingEmbedder
from secure_rag.events import JsonlEventLog, PipelineEvent
from secure_rag.pipeline import SecureRagPipeline
from secure_rag.sanitization.core import PrivacyGateway
from secure_rag.sanitization.ner import EnsembleDetector
from secure_rag.sanitization.regex import RegexDetector


class EmptyRetriever:
    def __init__(self) -> None:
        self.embedder = HashingEmbedder()

    def search(
        self,
        _query: str,
        top_k: int | None = None,
        *,
        on_event=None,
    ) -> list:
        return []


class FakeManifest:
    def latest_build_id(self) -> str:
        return "build-test"


def _config(tmp_path):
    base = load_config()
    source = (tmp_path / "source").resolve()
    source.mkdir()
    return replace(
        base,
        paths=replace(
            base.paths,
            source_root=source,
            runtime_root=(tmp_path / "runtime").resolve(),
        ),
    )


def test_attachment_pipeline_emits_safe_timed_events(tmp_path) -> None:
    config = _config(tmp_path)
    config.ensure_runtime()
    attachment = config.paths.source_root / "private-name.txt"
    attachment.write_text(
        "Контакт test@example.org отвечает за расчётную модель.",
        encoding="utf-8",
    )
    gateway = PrivacyGateway(EnsembleDetector([RegexDetector()]))
    events: list[PipelineEvent] = []

    result = SecureRagPipeline(
        config,
        EmptyRetriever(),
        gateway,
        FakeManifest(),
    ).run(
        "Кто отвечает за модель?",
        provider="stub",
        attachment_path=attachment,
        on_event=events.append,
    )

    outbound = result.codex_input.read_text(encoding="utf-8")
    restored = result.restored_output.read_text(encoding="utf-8")
    assert "test@example.org" not in outbound
    assert "private-name.txt" not in outbound
    assert "test@example.org" in restored
    completed = {event.stage for event in events if event.status == "completed"}
    assert {
        "request.validate",
        "retrieval.total",
        "attachment.read",
        "attachment.chunk",
        "attachment.embedding",
        "sanitizer.detect",
        "sanitizer.mark",
        "outbound.prepare",
        "provider.call",
        "demarker.restore",
    } <= completed
    assert all(event.duration_ms is not None for event in events if event.status == "completed")
    event_log = " ".join(str(dict(event.details)) for event in events)
    assert str(attachment) not in event_log
    assert "private-name.txt" not in event_log
    assert "test@example.org" not in event_log


def test_attachment_must_stay_inside_source_root(tmp_path) -> None:
    config = _config(tmp_path)
    config.ensure_runtime()
    outside = tmp_path / "outside.txt"
    outside.write_text("safe test", encoding="utf-8")
    events: list[PipelineEvent] = []
    pipeline = SecureRagPipeline(
        config,
        EmptyRetriever(),
        PrivacyGateway(EnsembleDetector([RegexDetector()])),
        FakeManifest(),
    )

    with pytest.raises(ValueError, match="inside the configured source root"):
        pipeline.run(
            "test",
            provider="stub",
            attachment_path=outside,
            on_event=events.append,
        )

    failed = [event for event in events if event.status == "failed"]
    assert failed[-1].stage == "request.validate"
    assert str(outside) not in str(dict(failed[-1].details))


def test_jsonl_event_log_drops_non_allowlisted_details(tmp_path) -> None:
    log = JsonlEventLog(tmp_path / "logs", str(uuid.uuid4()))
    log(
        PipelineEvent(
            stage="sanitizer.detect",
            label="dynamic label with private@example.org",
            status="completed",
            duration_ms=12.5,
            details={
                "detected_spans": 2,
                "raw_value": "private@example.org",
                "raw_path": r"D:\private\document.docx",
            },
        )
    )

    raw = log.path.read_text(encoding="utf-8")
    record = json.loads(raw)
    assert record["stage"] == "sanitizer.detect"
    assert record["details"] == {"detected_spans": 2}
    assert "private@example.org" not in raw
    assert "document.docx" not in raw
