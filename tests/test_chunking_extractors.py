from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from secure_rag.chunking import chunk_text
from secure_rag.config import load_config
from secure_rag.extractors import extract_document


def test_chunking_is_deterministic_and_overlaps() -> None:
    text = ("Первое предложение. Второе предложение с данными.\n\n" * 20).strip()
    kwargs = {
        "text": text,
        "document_id": "doc",
        "revision": "rev",
        "chunk_chars": 180,
        "overlap_chars": 30,
        "min_chunk_chars": 30,
    }
    first = chunk_text(**kwargs)
    second = chunk_text(**kwargs)
    assert first == second
    assert len(first) > 1
    assert all(chunk.text == text[chunk.start : chunk.end] for chunk in first)
    assert all(first[index + 1].start < first[index].end for index in range(len(first) - 1))


def test_html_extractor_removes_script(tmp_path) -> None:
    path = tmp_path / "sample.htm"
    path.write_text(
        "<html><body><h1>Полезный текст</h1><script>secret_script()</script></body></html>",
        encoding="utf-8",
    )
    config = load_config().ingestion
    result = extract_document(path, config)
    assert "Полезный текст" in result.text
    assert "secret_script" not in result.text


def test_source_and_runtime_may_not_overlap(tmp_path) -> None:
    default_path = Path(__file__).resolve().parents[1] / "config" / "pilot.yaml"
    raw = yaml.safe_load(default_path.read_text(encoding="utf-8"))
    raw["paths"]["source_root"] = "source"
    raw["paths"]["runtime_root"] = "source/runtime"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    path = config_dir / "pilot.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="must not overlap"):
        load_config(path)
