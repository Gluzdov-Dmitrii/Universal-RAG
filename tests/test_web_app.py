from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from secure_rag.api import web
from secure_rag.api.chat_state import StateMessage
from secure_rag.domain.models import CitationLocation, DocumentSource
from secure_rag.orchestration.events import PipelineEvent


def _client(monkeypatch, tmp_path: Path) -> TestClient:
    monkeypatch.setenv("SECURE_RAG_API_KEY", "test-backend-key")
    monkeypatch.setenv("SECURE_RAG_CHAT_STATE_DB", str(tmp_path / "chat-state.sqlite"))
    monkeypatch.setenv("SECURE_RAG_PIPELINE_EVENT_DIR", str(tmp_path / "events"))
    web._state_stores.clear()
    return TestClient(web.app)


def _jwt(user_id: str, *, secret: str = "test-backend-key") -> str:
    def encode(value: object) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    now = int(time.time())
    header = encode({"alg": "HS256", "typ": "JWT"})
    payload = encode(
        {"sub": user_id, "iss": "open-webui", "iat": now, "exp": now + 300}
    )
    signed = f"{header}.{payload}"
    signature = hmac.new(secret.encode(), signed.encode(), hashlib.sha256).digest()
    encoded_signature = base64.urlsafe_b64encode(signature).decode().rstrip("=")
    return f"{signed}.{encoded_signature}"


def _headers(user_id: str = "user-1", chat_id: str = "chat-1") -> dict[str, str]:
    return {
        "Authorization": "Bearer test-backend-key",
        "X-OpenWebUI-User-Jwt": _jwt(user_id),
        "X-OpenWebUI-Chat-Id": chat_id,
    }


def _result(tmp_path: Path, answer: str = "Готовый ответ"):
    restored = tmp_path / "restored.txt"
    restored.write_text(answer, encoding="utf-8")
    return SimpleNamespace(
        request_id="request-1",
        restored_output=restored,
        sources=(
            DocumentSource(
                citation_refs=("R001",),
                document_id="opaque-document-id",
                path=Path(r"D:\Nextcloud\Проекты\расчёт.xlsx"),
                file_type="xlsx",
                best_score=0.9,
                locations=(
                    CitationLocation(
                        citation_ref="R001",
                        kind="sheet",
                        start="Расчёты",
                    ),
                ),
            ),
        ),
    )


def test_health_does_not_load_models_or_require_auth(monkeypatch) -> None:
    monkeypatch.delenv("SECURE_RAG_API_KEY", raising=False)

    response = TestClient(web.app).get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "universal-rag"}


def test_history_uses_answer_but_not_progress_or_failed_attempts() -> None:
    completed = StateMessage(
        "assistant",
        "Ход выполнения:\n- поиск\n"
        f"{web._PROGRESS_END_MARKER}\n\nОтвет:\n\nготово"
        "\n\nИсточники на сервере:\n    R001 · private.docx",
    )
    failed = StateMessage(
        "assistant",
        f"Ход выполнения:\n{web._PIPELINE_ERROR_MARKER}\nошибка",
    )

    assert web._history_content(completed) == "готово"
    assert web._history_content(failed) == ""


def test_models_require_shared_backend_key(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)

    assert client.get("/v1/models").status_code == 401
    response = client.get("/v1/models", headers=_headers())

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["data"]] == ["universal-rag"]


def test_completion_uses_history_and_returns_plain_text_sources(
    tmp_path,
    monkeypatch,
) -> None:
    client = _client(monkeypatch, tmp_path)
    calls: list[tuple[str, str]] = []

    def run(question: str, *, conversation_context: str, on_event=None):
        calls.append((question, conversation_context))
        return _result(tmp_path, '**private**\n<img src="https://example.test/leak">')

    monkeypatch.setattr(web.runtime, "run", run)
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "universal-rag",
            "messages": [
                {"role": "user", "content": "старый вопрос"},
                {"role": "assistant", "content": "старый ответ"},
                {"role": "user", "content": "текущий вопрос"},
            ],
        },
    )

    assert response.status_code == 200
    assert calls == [
        (
            "текущий вопрос",
            "Пользователь:\nстарый вопрос\n\nАссистент:\nстарый ответ",
        )
    ]
    content = response.json()["choices"][0]["message"]["content"]
    assert content.startswith("    **private**")
    assert '    <img src="https://example.test/leak">' in content
    assert "Источники на сервере:" in content
    assert r"R001 · xlsx · лист Расчёты · D:\Nextcloud\Проекты\расчёт.xlsx" in content


def test_streaming_completion_uses_openai_sse_shape(tmp_path, monkeypatch) -> None:
    client = _client(monkeypatch, tmp_path)

    def run(_question: str, *, conversation_context: str, on_event=None):
        assert on_event is not None
        on_event(
            PipelineEvent(
                stage="retrieval.total",
                label="Поиск контекста",
                status="started",
            )
        )
        on_event(
            PipelineEvent(
                stage="retrieval.total",
                label="Поиск контекста",
                status="completed",
                details={"index_hits": 3},
            )
        )
        return _result(tmp_path)

    monkeypatch.setattr(web.runtime, "run", run)

    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "universal-rag",
            "stream": True,
            "messages": [{"role": "user", "content": "вопрос"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"
    assert '"object": "chat.completion.chunk"' in response.text
    assert "Ход выполнения" in response.text
    assert "Ищу релевантные фрагменты" in response.text
    assert "найдено фрагментов — 3" in response.text
    assert "Ответ:" in response.text
    assert "data: [DONE]" in response.text


def test_completion_rejects_unknown_model_and_missing_user_message(
    monkeypatch, tmp_path
) -> None:
    client = _client(monkeypatch, tmp_path)

    unknown = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "other", "messages": [{"role": "user", "content": "x"}]},
    )
    missing_user = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "universal-rag",
            "messages": [{"role": "system", "content": "instructions"}],
        },
    )

    assert unknown.status_code == 404
    assert missing_user.status_code == 400


def test_pipeline_failure_does_not_expose_exception_text(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)

    def fail(_question: str, *, conversation_context: str, on_event=None):
        raise RuntimeError(r"secret from D:\Nextcloud\private.docx")

    monkeypatch.setattr(web.runtime, "run", fail)
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "universal-rag",
            "messages": [{"role": "user", "content": "вопрос"}],
        },
    )

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["code"] == "pipeline_failed"
    assert detail["stage"] == "Обрабатываю запрос"
    assert detail["diagnostic_id"]
    assert "private.docx" not in response.text


def test_streaming_failure_reports_safe_stage_and_diagnostic_id(
    monkeypatch,
    tmp_path,
) -> None:
    client = _client(monkeypatch, tmp_path)

    def fail(_question: str, *, conversation_context: str, on_event=None):
        assert on_event is not None
        on_event(
            PipelineEvent(
                stage="provider.initialize",
                label="Подготовка выбранной модели",
                status="started",
            )
        )
        on_event(
            PipelineEvent(
                stage="provider.initialize",
                label="Подготовка выбранной модели",
                status="failed",
                details={"error_type": "RuntimeError"},
            )
        )
        raise RuntimeError(r"secret from D:\Nextcloud\private.docx")

    monkeypatch.setattr(web.runtime, "run", fail)
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "universal-rag",
            "stream": True,
            "messages": [{"role": "user", "content": "вопрос"}],
        },
    )

    assert response.status_code == 200
    assert "Подготавливаю выбранную модель" in response.text
    assert "Код диагностики" in response.text
    assert "private.docx" not in response.text
    assert "data: [DONE]" in response.text
    event_files = list((tmp_path / "events").glob("*.jsonl"))
    assert len(event_files) == 1
    assert "private.docx" not in event_files[0].read_text(encoding="utf-8")


def test_completion_requires_signed_user_and_chat_identity(monkeypatch, tmp_path) -> None:
    client = _client(monkeypatch, tmp_path)
    payload = {
        "model": "universal-rag",
        "messages": [{"role": "user", "content": "вопрос"}],
    }

    missing = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-backend-key"},
        json=payload,
    )
    forged = client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer test-backend-key",
            "X-OpenWebUI-User-Jwt": _jwt("user-1", secret="wrong-key"),
            "X-OpenWebUI-Chat-Id": "chat-1",
        },
        json=payload,
    )

    assert missing.status_code == 401
    assert missing.json() == {"detail": "openwebui_identity_required"}
    assert forged.status_code == 401
    assert forged.json() == {"detail": "invalid_user_identity"}


def test_persisted_context_is_isolated_by_user_and_reused_for_incremental_requests(
    monkeypatch,
    tmp_path,
) -> None:
    client = _client(monkeypatch, tmp_path)
    contexts: list[str] = []

    def run(question: str, *, conversation_context: str, on_event=None):
        contexts.append(conversation_context)
        return _result(tmp_path, answer=f"ответ на {question}")

    monkeypatch.setattr(web.runtime, "run", run)
    first = client.post(
        "/v1/chat/completions",
        headers=_headers("user-a", "shared-chat-id"),
        json={
            "model": "universal-rag",
            "messages": [{"role": "user", "content": "первый вопрос"}],
        },
    )
    second = client.post(
        "/v1/chat/completions",
        headers=_headers("user-a", "shared-chat-id"),
        json={
            "model": "universal-rag",
            "messages": [{"role": "user", "content": "продолжение"}],
        },
    )
    other_user = client.post(
        "/v1/chat/completions",
        headers=_headers("user-b", "shared-chat-id"),
        json={
            "model": "universal-rag",
            "messages": [{"role": "user", "content": "чужой вопрос"}],
        },
    )

    assert first.status_code == second.status_code == other_user.status_code == 200
    assert contexts[0] == ""
    assert "первый вопрос" in contexts[1]
    assert "ответ на первый вопрос" in contexts[1]
    assert contexts[2] == ""
