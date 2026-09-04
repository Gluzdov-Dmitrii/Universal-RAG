from __future__ import annotations

import json

from secure_rag.api.cli import WEB_APP_PATH, _safe_jsonl, build_parser


def test_safe_jsonl_emits_exactly_one_parseable_record(capsys) -> None:
    value = {"phase": "complete_with_errors", "failed": 2}

    _safe_jsonl(value)

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == value


def test_cli_streamlit_entrypoint_exists() -> None:
    assert WEB_APP_PATH.is_file()
    assert WEB_APP_PATH.name == "web.py"


def test_prepare_models_command_is_available_without_a_document_query() -> None:
    args = build_parser().parse_args(["prepare-models", "--ner", "all"])

    assert args.command == "prepare-models"
    assert args.ner == "all"
