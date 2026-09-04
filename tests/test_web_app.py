from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from secure_rag.api import web
from secure_rag.domain.models import CitationLocation, DocumentSource


def _client(monkeypatch) -> TestClient:
    monkeypatch.setenv("SECURE_RAG_API_KEY", "test-backend-key")
    return TestClient(web.app)


def _headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-backend-key"}


def _result(tmp_path: Path, answer: str = "Готовый ответ"):
    restored = tmp_path / "restored.txt"
    restored.write_text(answer, encoding="utf-8")
    return SimpleNamespace(
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


def test_models_require_shared_backend_key(monkeypatch) -> None:
    client = _client(monkeypatch)

    assert client.get("/v1/models").status_code == 401
    response = client.get("/v1/models", headers=_headers())

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["data"]] == ["universal-rag"]


def test_completion_uses_latest_user_message_and_returns_plain_text_sources(
    tmp_path,
    monkeypatch,
) -> None:
    client = _client(monkeypatch)
    questions: list[str] = []

    def run(question: str):
        questions.append(question)
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
    assert questions == ["текущий вопрос"]
    content = response.json()["choices"][0]["message"]["content"]
    assert content.startswith("    **private**")
    assert '    <img src="https://example.test/leak">' in content
    assert "Источники на сервере:" in content
    assert r"R001 · xlsx · лист Расчёты · D:\Nextcloud\Проекты\расчёт.xlsx" in content


def test_streaming_completion_uses_openai_sse_shape(tmp_path, monkeypatch) -> None:
    client = _client(monkeypatch)
    monkeypatch.setattr(web.runtime, "run", lambda _question: _result(tmp_path))

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
    assert '"object": "chat.completion.chunk"' in response.text
    assert "data: [DONE]" in response.text


def test_completion_rejects_unknown_model_and_missing_user_message(monkeypatch) -> None:
    client = _client(monkeypatch)

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


def test_pipeline_failure_does_not_expose_exception_text(monkeypatch) -> None:
    client = _client(monkeypatch)

    def fail(_question: str):
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
    assert response.json() == {"detail": "pipeline_failed"}
    assert "private.docx" not in response.text
