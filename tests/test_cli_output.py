from __future__ import annotations

import json
import os

from secure_rag.api.cli import (
    WEB_API_IMPORT,
    _load_repo_environment,
    _safe_jsonl,
    build_parser,
)


def test_safe_jsonl_emits_exactly_one_parseable_record(capsys) -> None:
    value = {"phase": "complete_with_errors", "failed": 2}

    _safe_jsonl(value)

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == value


def test_cli_web_api_entrypoint_is_available() -> None:
    assert WEB_API_IMPORT == "secure_rag.api.web:app"
    args = build_parser().parse_args(["serve", "--host", "0.0.0.0", "--port", "8000"])

    assert args.command == "serve"
    assert args.host == "0.0.0.0"
    assert args.port == 8000


def test_prepare_models_command_is_available_without_a_document_query() -> None:
    args = build_parser().parse_args(["prepare-models", "--ner", "all"])

    assert args.command == "prepare-models"
    assert args.ner == "all"


def test_repo_environment_does_not_override_service_environment(tmp_path, monkeypatch) -> None:
    environment = tmp_path / ".env"
    loaded_name = "UNIVERSAL_RAG_TEST_FROM_DOTENV"
    environment.write_text(
        f"{loaded_name}=loaded\nSECURE_RAG_API_PORT=9000\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SECURE_RAG_API_PORT", "8000")
    os.environ.pop(loaded_name, None)

    try:
        _load_repo_environment(environment)

        assert os.environ[loaded_name] == "loaded"
        assert os.environ["SECURE_RAG_API_PORT"] == "8000"
    finally:
        os.environ.pop(loaded_name, None)
