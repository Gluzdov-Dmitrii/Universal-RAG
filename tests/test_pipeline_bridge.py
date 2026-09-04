from __future__ import annotations

import json
from dataclasses import replace

import pytest

from secure_rag.config import load_config
from secure_rag.domain.models import EntitySpan, MarkerState, RetrievalHit
from secure_rag.llm.adapters import OpenAIResponsesProvider
from secure_rag.llm.bridge import BridgeManager
from secure_rag.orchestration.pipeline import SecureRagPipeline
from secure_rag.sanitization.core import PrivacyGateway
from secure_rag.sanitization.ner import EnsembleDetector
from secure_rag.sanitization.regex import RegexDetector


class LiteralDetector:
    name = "literal"

    def detect(self, text: str) -> list[EntitySpan]:
        values = {
            "Анна Смирнова": "PER",
            "Анну Смирнову": "PER",
            "СибНИА": "ORG",
        }
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


def _bridge_config(tmp_path):
    base = load_config()
    config = replace(
        base,
        paths=replace(base.paths, runtime_root=(tmp_path / "runtime").resolve()),
    )
    config.ensure_runtime()
    return config


def _config_with_agent_workspace(tmp_path):
    base = load_config()
    workspace = (tmp_path / "agent-workspace").resolve()
    workspace.mkdir()
    (workspace / "instructions.md").write_text("trusted instructions", encoding="utf-8")
    config = replace(
        base,
        paths=replace(base.paths, runtime_root=(tmp_path / "runtime").resolve()),
        generation=replace(
            base.generation,
            agent_workspace_root=workspace,
            instruction_files=("instructions.md",),
        ),
    )
    config.ensure_runtime()
    return config


def test_bridge_ignores_known_value_occurring_only_in_trusted_prompt(tmp_path) -> None:
    config = _bridge_config(tmp_path)
    state = MarkerState()
    marker = PrivacyGateway(None).mark_literal("Codex", "ORG", state)

    result = BridgeManager(config).create(
        f"Что такое {marker}?",
        [],
        state,
        versions={},
        provider="manual",
        iterative_enabled=True,
        max_iterations=3,
    )

    outbound = result.codex_input.read_text(encoding="utf-8")
    assert "Правила для Codex" in outbound
    assert f"Что такое {marker}?" in outbound


def test_bridge_still_blocks_known_value_in_dynamic_context(tmp_path) -> None:
    config = _bridge_config(tmp_path)
    state = MarkerState()
    marker = PrivacyGateway(None).mark_literal("Codex", "ORG", state)
    contexts = [
        {
            "document_id": "opaque-document",
            "chunk_id": "opaque-chunk",
            "citation_ref": "R001",
            "score": 0.9,
            "source_ref": "opaque-document",
            "file_type": "txt",
            "text": "Немаркированное значение Codex.",
        }
    ]

    with pytest.raises(
        ValueError,
        match="^Outbound validation found a known unmarked value$",
    ):
        BridgeManager(config).create(
            f"Что такое {marker}?",
            contexts,
            state,
            versions={},
            provider="manual",
        )

    assert list(config.requests_path.iterdir()) == []


def test_canonical_surname_propagates_when_ner_misses_bare_form(tmp_path) -> None:
    class CompositeNameDetector:
        name = "composite-name"

        @staticmethod
        def detect(text: str) -> list[EntitySpan]:
            value = "О ? Егоров"
            start = text.find(value)
            if start < 0:
                return []
            return [
                EntitySpan(
                    start=start,
                    end=start + len(value),
                    label="PER",
                    score=1.0,
                    source="composite-name",
                    priority=500,
                )
            ]

    class SurnameRetriever(FakeRetriever):
        def search(self, _query: str, top_k: int | None = None, *, on_event=None):
            del top_k, on_event
            return [
                RetrievalHit(
                    chunk_id="00000000-0000-0000-0000-000000000002",
                    document_id="opaque-surname-document",
                    revision="revision",
                    score=0.9,
                    start=0,
                    end=30,
                    text="О ? Егоров указан в списке.",
                    source_name="список.txt",
                )
            ]

    base = load_config()
    config = replace(
        base,
        paths=replace(base.paths, runtime_root=(tmp_path / "runtime").resolve()),
    )
    config.ensure_runtime()
    gateway = PrivacyGateway(CompositeNameDetector())
    result = SecureRagPipeline(
        config,
        SurnameRetriever(),
        gateway,
        FakeManifest(),
    ).run("Что известно про Егоров?", provider="manual")

    outbound = result.codex_input.read_text(encoding="utf-8")
    assert "Егоров" not in outbound
    assert outbound.count("[[PER_0001]]") == 2


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
    assert "Сотрудник + PER-маркер + сказуемое" in outbound
    assert "анна смирнова" in restored
    assert "test@example.org" in restored


def test_provider_controlled_link_is_restored_only_to_plain_text(tmp_path, monkeypatch) -> None:
    config = _config_with_agent_workspace(tmp_path)
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
    assert "https://invalid.example/?value=Анну Смирнову" in restored


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
    assert manifest["automatic_send"] is False


def test_responses_provider_is_automatic_and_demarks_response(tmp_path, monkeypatch) -> None:
    config = _config_with_agent_workspace(tmp_path)
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

    assert "Анну Смирнову" in result.restored_output.read_text(encoding="utf-8")
    manifest = json.loads((result.request_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["automatic_send"] is True
    assert manifest["human_review_required"] is False
    assert manifest["provider_boundary"] == "no-tools-api"


def test_provider_response_can_be_restored_after_ui_interrupt(tmp_path, monkeypatch) -> None:
    class SyntheticUiInterrupt(BaseException):
        pass

    config = _config_with_agent_workspace(tmp_path)
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

    assert "Анну Смирнову" in restored.read_text(encoding="utf-8")


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
