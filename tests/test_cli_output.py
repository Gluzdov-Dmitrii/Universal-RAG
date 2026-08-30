from __future__ import annotations

import json

from secure_rag.cli import _safe_jsonl


def test_safe_jsonl_emits_exactly_one_parseable_record(capsys) -> None:
    value = {"phase": "complete_with_errors", "failed": 2}

    _safe_jsonl(value)

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == value
