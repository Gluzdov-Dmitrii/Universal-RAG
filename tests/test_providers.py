from __future__ import annotations

from types import SimpleNamespace

import openai
import pytest

from secure_rag.providers import (
    LocalCodexProvider,
    OpenAIResponsesProvider,
    resolve_provider_name,
)


class FakeResponses:
    def __init__(self, output_text: str = "Ответ с [[PER_0001]]") -> None:
        self.output_text = output_text
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


class FakeClient:
    def __init__(self, responses: FakeResponses) -> None:
        self.responses = responses


def test_auto_uses_responses_only_when_api_key_exists(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert resolve_provider_name("auto") == "stub"

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert resolve_provider_name("auto") == "responses"


def test_responses_call_omits_tools_and_disables_storage(monkeypatch) -> None:
    monkeypatch.setenv("SECURE_RAG_OPENAI_MODEL", "test-model")
    responses = FakeResponses()
    provider = OpenAIResponsesProvider(client=FakeClient(responses))

    answer = provider.answer_payload("Запрос с [[PER_0001]]", "opaque-request")

    assert answer == "Ответ с [[PER_0001]]\n"
    assert len(responses.calls) == 1
    call = responses.calls[0]
    assert call["model"] == "test-model"
    assert call["input"] == "Запрос с [[PER_0001]]"
    assert call["metadata"] == {"request_id": "opaque-request"}
    assert call["store"] is False
    assert "extra_headers" not in call
    assert "tools" not in call


def test_official_responses_client_disables_sdk_retries(monkeypatch) -> None:
    constructor_calls: list[dict[str, object]] = []

    def fake_openai(**kwargs):
        constructor_calls.append(kwargs)
        return FakeClient(FakeResponses())

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai, "OpenAI", fake_openai)

    OpenAIResponsesProvider()

    assert constructor_calls == [
        {
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1",
            "max_retries": 0,
        }
    ]


def test_explicit_responses_needs_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="^openai_api_key_missing$"):
        OpenAIResponsesProvider()


def test_responses_provider_hides_upstream_error_content() -> None:
    class FailingResponses:
        @staticmethod
        def create(**_kwargs):
            raise RuntimeError(r"secret text at D:\private\document.docx")

    provider = OpenAIResponsesProvider(client=FakeClient(FailingResponses()))

    with pytest.raises(RuntimeError) as error:
        provider.answer_payload("sanitized payload", "opaque-request")

    assert str(error.value) == "openai_responses_provider_failed"
    assert error.value.__cause__ is None


def test_codex_local_needs_explicit_opt_in(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("SECURE_RAG_ALLOW_UNSAFE_CODEX_LOCAL", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))

    with pytest.raises(RuntimeError, match="^codex_local_requires_explicit_unsafe_opt_in$"):
        LocalCodexProvider(forbidden_roots=(tmp_path / "repo",))


def test_codex_local_workspace_root_stays_outside_private_roots(monkeypatch, tmp_path) -> None:
    repo_root = (tmp_path / "repo").resolve()
    repo_root.mkdir()
    local_app_data = (tmp_path / "appdata").resolve()
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.setenv("SECURE_RAG_ALLOW_UNSAFE_CODEX_LOCAL", "1")
    monkeypatch.delenv("SECURE_RAG_CODEX_SANDBOX_ROOT", raising=False)

    provider = LocalCodexProvider(forbidden_roots=(repo_root,))

    assert provider.sandbox_root.is_relative_to(local_app_data)
    assert not provider.sandbox_root.is_relative_to(repo_root)


def test_codex_local_persists_separate_thread_when_configured(
    monkeypatch, tmp_path
) -> None:
    import sys

    calls: list[tuple[str, object]] = []

    class FakeThread:
        @staticmethod
        def set_name(name):
            calls.append(("name", name))

        @staticmethod
        def run(payload):
            calls.append(("run", payload))
            return SimpleNamespace(final_response="Ответ с [[PER_0001]]")

    class FakeCodex:
        def __init__(self, config):
            calls.append(("config", config))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def thread_start(self, **kwargs):
            calls.append(("start", kwargs))
            return FakeThread()

    fake_module = SimpleNamespace(
        ApprovalMode=SimpleNamespace(deny_all="deny_all"),
        Codex=FakeCodex,
        CodexConfig=lambda **kwargs: kwargs,
        Sandbox=SimpleNamespace(read_only="read_only"),
    )
    monkeypatch.setitem(sys.modules, "openai_codex", fake_module)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("SECURE_RAG_ALLOW_UNSAFE_CODEX_LOCAL", "1")
    monkeypatch.setenv("SECURE_RAG_CODEX_PERSIST_THREADS", "1")

    answer = LocalCodexProvider().answer_payload(
        "Запрос с [[PER_0001]]", "00000000-0000-4000-8000-000000000002"
    )

    assert answer == "Ответ с [[PER_0001]]\n"
    options = next(value for name, value in calls if name == "start")
    assert options["ephemeral"] is False
    assert options["sandbox"] == "read_only"
    assert options["approval_mode"] == "deny_all"
    assert ("name", "Secure RAG 00000000") in calls
    assert calls[-1] == ("run", "Запрос с [[PER_0001]]")
