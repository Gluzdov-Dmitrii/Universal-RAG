from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import secure_rag.ingestion.signatures as signatures_module
from secure_rag.config import load_config
from secure_rag.domain.models import TextLocation
from secure_rag.ingestion.chunking import chunk_text
from secure_rag.ingestion.extractors import extract_document, iter_source_files, normalize_text
from secure_rag.ingestion.signatures import compute_index_signature


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


def test_chunking_maps_character_ranges_to_page_range() -> None:
    text = "[PAGE 1]\nпервая страница\n\n[PAGE 2]\nвторая страница"
    page_two_start = text.index("[PAGE 2]")

    chunks = chunk_text(
        text=text,
        document_id="doc",
        revision="rev",
        chunk_chars=len(text),
        overlap_chars=0,
        min_chunk_chars=1,
        text_locations=(
            TextLocation(0, page_two_start, "page", "1"),
            TextLocation(page_two_start, len(text), "page", "2"),
        ),
    )

    assert len(chunks) == 1
    assert chunks[0].location_kind == "page"
    assert chunks[0].location_start == "1"
    assert chunks[0].location_end == "2"


def test_normalize_text_replaces_lone_surrogates_without_shifting_offsets() -> None:
    source = "до\ud800после"

    normalized = normalize_text(source)

    assert normalized == "до\ufffdпосле"
    assert len(normalized) == len(source)
    normalized.encode("utf-8")


def test_html_extractor_reads_valid_document(tmp_path) -> None:
    path = tmp_path / "sample.htm"
    path.write_text(
        "<html><body><h1>Полезный текст</h1><p>Второй абзац</p></body></html>",
        encoding="utf-8",
    )
    config = load_config().ingestion
    result = extract_document(path, config)
    assert "Полезный текст" in result.text
    assert "Второй абзац" in result.text


def test_html_extractor_recovers_malformed_document(tmp_path) -> None:
    path = tmp_path / "malformed.html"
    path.write_text(
        "<html><body><h1>Заголовок<p>Первый <b>жирный<p>Второй",
        encoding="utf-8",
    )

    result = extract_document(path, load_config().ingestion)

    assert "Заголовок" in result.text
    assert "Первый" in result.text
    assert "жирный" in result.text
    assert "Второй" in result.text


def test_source_scan_excludes_generated_webhelp_indexes(tmp_path) -> None:
    generated = tmp_path / "docs" / "whgdata" / "whlstf163.htm"
    generated.parent.mkdir(parents=True)
    generated.write_text("generated search index", encoding="utf-8")
    document = tmp_path / "docs" / "manual.htm"
    document.write_text("useful document", encoding="utf-8")

    discovered = list(iter_source_files(tmp_path, (".htm",)))

    assert discovered == [document.resolve()]


def test_html_extractor_excludes_active_and_fallback_content(tmp_path) -> None:
    path = tmp_path / "excluded.html"
    path.write_text(
        """<html><head><style>hidden_style</style></head><body>
        До<script>hidden_script</script>после
        <noscript>hidden_fallback</noscript>видимый хвост
        </body></html>""",
        encoding="utf-8",
    )

    result = extract_document(path, load_config().ingestion)

    assert "До" in result.text
    assert "после" in result.text
    assert "видимый хвост" in result.text
    assert "hidden_style" not in result.text
    assert "hidden_script" not in result.text
    assert "hidden_fallback" not in result.text


def test_index_signature_changes_with_extractor_policy(monkeypatch) -> None:
    config = load_config()
    current = compute_index_signature(config, "test-embedding")

    monkeypatch.setattr(
        signatures_module,
        "EXTRACTOR_VERSION",
        "test-only-different-extractor-policy",
    )

    assert current != compute_index_signature(config, "test-embedding")


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
